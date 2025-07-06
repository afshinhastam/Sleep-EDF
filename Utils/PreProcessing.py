import mne
import os
import numpy as np
import pickle
from time import time
import torch
import collections
import Utils

import argparse
parser = argparse.ArgumentParser(
    description="Build MNE epochs from EDF files and log statistics to TensorBoard."
)
parser.add_argument(
    "--epoch_size", 
    type=float, 
    default=29.99,
    help="Epoch length in seconds (default: 29.99)"
)
parser.add_argument(
    "--save_epochs_dir", 
    type=str, 
    default="epochs",
    help="Directory to save .npz files containing epochs and labels (default: epochs)"
)
parser.add_argument(
    "--save_checkpoint_dir", 
    type=str, 
    default="checkpoints",
    help="Directory for TensorBoard logs and checkpoints (default: checkpoints)"
)
parser.add_argument(
    "--root_edf_path", 
    type=str, 
    default="sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette/sleep_epochs_windowed_v3",
    help="Root path where EDF files are stored (default: sleep-edf-database-expanded-1.0.0/...)"
)
parser.add_argument(
    "--save_npz_path", 
    type=str, 
    default="epochs",
    help="Save path where npz epochs will be save"
)
parser.add_argument(
    "--save_seq_path", 
    type=str, 
    default="seq_epochs",
    help="Save path where sequentialed epochs will be save"
)
parser.add_argument(
    "--seq_len", 
    type=int, 
    default=10,
    help="Seq. length"
)
parser.add_argument(
    "--seq_shift", 
    type=int, 
    default=5,
    help="Epoch shift in the Sqe."
)
args = parser.parse_args()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=args.save_checkpoint_dir)

def build_epochs(local_edf_path=args.root_edf_path, save_npz_path=args.save_epochs_dir):
  # Load all datas and their annotationas.
  all_file_name = os.listdir(local_edf_path)

  PSG_list = [f for f in all_file_name if f.split('-')[-1]=='PSG.edf']
  Hypnogram_list = [f for f in all_file_name if f.split('-')[-1]=='Hypnogram.edf']

  PSG_list.sort()
  Hypnogram_list.sort()

  map = { 'Sleep stage 1': 'N1',
          'Sleep stage 2': 'N2',
          'Sleep stage 3': 'N3',
          'Sleep stage 4': 'N3',}

  event_id = {'N1': 0,
              'N2': 1,
              'N3': 2,
              'Sleep stage W': 3,
              'Sleep stage R': 4}

  tmin = 0.0
  tmax = args.epoch_size
  baseline = (None, None)
  global_counter = collections.Counter()

  for i in range(len(PSG_list)):
    t_start = time()
    PSG_name = PSG_list[i]
    Hypnogram_name = Hypnogram_list[i]

    if os.path.exists(os.path.join(save_npz_path, PSG_name[:7]+".npz")):
      print("file is exist:", {os.path.join(save_npz_path, PSG_name[:7]+".npz")})
      continue

    if Hypnogram_name[:7] != PSG_name[:7]:
      print("Error and Continue: ->", PSG_name[:7], "!=", Hypnogram_name[:7])
      continue

    # assert Hypnogram_name[:7] == PSG_name[:7]

    annotation = mne.read_annotations(os.path.join(local_edf_path, Hypnogram_name))
    annotation = annotation.crop(annotation[1]['onset'] - 30 * 60,
                    annotation[-2]['onset'] + 30 * 60)

    existing_labels = set(annotation.description)
    map = {k: v for k, v in map.items() if k in existing_labels}
    annotation.rename(map)

    data = mne.io.read_raw_edf(os.path.join(local_edf_path, PSG_name))
    data.set_annotations(annotation)

    data.drop_channels(data.ch_names[1:]) # Fpz-Cz channel

    events, events_id = mne.events_from_annotations(data, event_id=event_id, chunk_duration=30.)

    epochs = mne.Epochs(data,
                      events=events,
                      event_id=event_id,
                      tmin=tmin,
                      tmax=tmax,
                      baseline=baseline,
                      preload=True,
                      on_missing='warn')

    epochs_list = []
    epochs_label_list = []
    for j, epoch in enumerate(epochs):
      epochs_list.append(epoch)
      epochs_label_list.append(events[j][-1])
    
    epochs_data = np.stack(epochs_list, 0)
    epochs_data_label = np.stack(epochs_label_list, 0)
    event_labels_tensor = torch.tensor(epochs_data_label.reshape(-1), dtype=torch.int8)
    writer.add_histogram('Event_IDs_Histogram/{}'.format(PSG_name[:7]), event_labels_tensor, global_step=i)

    # Count per class
    counter = collections.Counter(epochs_data_label.reshape(-1))

    for class_name, class_idx in event_id.items():
      count = counter[class_idx]
      writer.add_scalar('Event_Counts/{}/{}'.format(PSG_name[:7], class_name), count, global_step=i)

    # Update global counter
    global_counter.update(counter)
    
    os.makedirs(save_npz_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_npz_path, PSG_name[:7]+".npz"), epochs=epochs_data, labels=epochs_data_label)

    print('\n {:d}/{:d} ---- {:s} Finished ---- Time:{:.2f}s\n'.format(i+1, len(PSG_list), PSG_name, time()-t_start))

  for class_name, class_idx in event_id.items():
    count = global_counter[class_idx]
    writer.add_scalar('Event_Counts_Global/{}'.format(class_name), count, global_step=0)

  writer.close()
  
if __name__=="__main__":
    build_epochs(args.root_edf_path, 
                 args.save_npz_path)
    print(f"-> Epochs generated and save into: {args.save_npz_path}")

    Utils.build_sequence_data(args.save_npz_path, 
                              args.save_seq_path, 
                              shift=args.seq_shift, 
                              n_epoch=args.seq_len)
    print(f"-> Sequence generated from epochs and save into: {args.save_npz_path}")

    Utils.build_database_info(args.save_seq_path)
    print(f"-> Database info generated and saved in: {args.save_seq_path}")

    print("-> Finished")

    # Command line
    # python build_epochs_from_edf.py \
    # --epoch_size 29.99 \
    # --save_epochs_dir epochs_v2 \
    # --save_checkpoint_dir tb_logs \
    # --root_edf_path "sleep-edf" \
    # --save_seq_path "seq_epochs"
