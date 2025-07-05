import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from torch.utils.data import Sampler
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np

def build_sequence_data(root_npz_folder, 
                        save_npz_folder,
                        shift=5,
                        n_epoch=10,
                        ):

    total_files = len(glob(os.path.join(root_npz_folder, "*.npz")))
    for f in tqdm(glob(os.path.join(root_npz_folder, "*.npz")), total=total_files):
        data = np.load(f)
        raw = data["epochs"]
        label = data["labels"]

        all_epochs = raw.shape[0] // n_epoch

        raw = raw[:all_epochs * n_epoch]
        label = label[:all_epochs * n_epoch]

        raw = sliding_window_view(raw, window_shape=n_epoch, axis=0)
        raw = raw[::shift]
        raw = np.transpose(raw, (0, -1, 1, 2)) 

        label = sliding_window_view(label, window_shape=n_epoch, axis=0)
        label = label[::shift]

        np.savez_compressed(os.path.join(save_npz_folder, os.path.split(f)[-1][:7]+".npz"), epochs=raw, labels=label)
        print(os.path.join(save_npz_folder, os.path.split(f)[-1][:7]+".npz"), "saved.")

def build_database_info(root_npz_folder, max_files=None):
    """
    Args:
        root_folder (str): folder containing .npz and labels_*.csv files
        transform: optional transform to apply to data
        max_files: if set, only load this many files (for debugging)
    """
    npz_files = sorted(glob.glob(os.path.join(root_npz_folder, "*.npz")))
    if max_files:
        npz_files = npz_files[:max_files]

    print(f"Found {len(npz_files)} npz files.")

    f_name = []
    f_idx = []
    s_name = []
    with tqdm(total=len(npz_files)) as bar:
        for nf in npz_files:
            data = np.load(nf)
            raw = data["epochs"]
            f_name.extend([os.path.split(nf)[-1] for i in range(raw.shape[0])])
            f_idx.extend(list(range(raw.shape[0])))
            s_name.extend([os.path.split(nf)[-1][3:5] for i in range(raw.shape[0])])
            bar.update(1)

    info_csv = pd.DataFrame(np.array([f_name, f_idx, s_name]).T, columns=["file_name", "index", "subject_id"])
    with open(os.path.join(root_npz_folder, "info_csv.csv"), "w") as f:
        info_csv.to_csv(f)

    return info_csv

class NormalDataset(Dataset):
    def __init__(self, root_npz_folder, transform=None, max_files=None):
        data_list = glob(os.path.join(root_npz_folder, "/*.npz"))
        self.X = None
        self.Y = None

        with tqdm(total=len(data_list), desc="Loading data") as bar:
            for i, d_l in enumerate(data_list):
                d = np.load(d_l)
                if self.X is None:
                    self.X = d["epochs"]
                    self.Y = d["labels"]
                else:
                    self.X = np.stack((self.X, d["epochs"]), axis=0)
                    self.Y = np.stack((self.Y, d["labels"]), axis=0)
                
                bar.update(1)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

class SubjectObjectDataset(Dataset):
    def __init__(self, data_info, data_dir, dtype=torch.float32):
        self.dtype = dtype
        self.data_info = data_info
        self.data_dir = data_dir
        self.loaded_files = {}  # optional cache

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        file_name = row['file_name']
        data_index = row['index']
        subject_id = row['subject_id']

        if file_name not in self.loaded_files:
            file_path = os.path.join(self.data_dir, file_name)
            self.loaded_files[file_name] = np.load(file_path, allow_pickle=True)

        npz_data = self.loaded_files[file_name]
        data_array = npz_data['epochs']
        label = npz_data['labels']
        sample = data_array[data_index]
        sample = torch.tensor(sample, dtype=self.dtype)
        labels = label[data_index]
        labels = torch.tensor(labels, dtype=self.dtype)

        return sample, labels

class SubjectBatchSampler(Sampler):
    def __init__(self, data_info, batch_size):
        """
        data_info: DataFrame with 'subject_id'
        batch_size: number of samples per batch
        """
        self.batch_size = batch_size
        self.subject_to_indices = defaultdict(list)

        # group indices by subject
        for idx, subject in enumerate(data_info['subject_id']):
            self.subject_to_indices[subject].append(idx)

        # flatten list of batches
        self.batches = []
        for subject, indices in self.subject_to_indices.items():
            random.shuffle(indices)
            # split into batches
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i+batch_size])

        random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def plot_hypnogram(pred_labels, true_labels, epoch_duration=30, event_id=None, title="Hypnogram", save_name="hypnogram.png"):
    """
    Plot hypnogram: predicted vs ground truth sleep stages.
    
    pred_labels: array-like, predicted class indices
    true_labels: array-like, ground truth class indices
    epoch_duration: seconds per epoch (default 30)
    event_id: dict mapping stage names to integers, e.g., {'W':3, 'N1':0, 'N2':1, 'N3':2, 'R':4}
    title: plot title
    """
    assert len(pred_labels) == len(true_labels), "Predictions and ground truth must have same length"
    
    # If event_id not given, define default
    if event_id is None:
        event_id = {'N1': 0, 'N2': 1, 'N3': 2, 'W': 3, 'R': 4}
    
    # Reverse mapping
    idx_to_stage = {v:k for k,v in event_id.items()}
    
    # Convert indices to stage names
    pred_stages = [idx_to_stage[i] for i in pred_labels]
    true_stages = [idx_to_stage[i] for i in true_labels]
    
    # Make Y numeric: use the same mapping
    y_pred = pred_labels
    y_true = true_labels

    # Time axis in hours
    times = np.arange(len(y_pred)) * epoch_duration / 3600  # in hours

    plt.figure(figsize=(12, 4))
    
    plt.step(times, y_true, where='post', label='Ground Truth', color='black', linewidth=1.5)
    plt.step(times, y_pred, where='post', label='Prediction', color='red', alpha=0.7)
    
    plt.yticks(list(event_id.values()), list(event_id.keys()))
    plt.ylim([max(event_id.values())+0.5, -0.5])  # invert y-axis: deeper stages lower
    plt.xlabel("Time (hours)")
    plt.ylabel("Sleep Stage")
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_name)




if __name__=="__main__":
    build_database_info("sleep-edf-database-expanded-1.0.0\\sleep-edf-database-expanded-1.0.0\\sleep-cassette\\sleep_epochs_windowed_v3")
