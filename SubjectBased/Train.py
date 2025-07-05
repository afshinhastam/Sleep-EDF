import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ..Utils import Utils
import os
import Model
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--continue_training", type=bool, default=False)
parser.add_argument("--continue_from_epoch", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--save_name", type=str, default="model.pth")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--root_path", type=str, default="sleep-edf-database-expanded-1.0.0\\sleep-edf-database-expanded-1.0.0\\sleep-cassette\\sleep_epochs_windowed_v3")
parser.add_argument("--info_path", type=str, default="info_csv.csv")

args = parser.parse_args()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=args.save_dir)

def train_and_plot(model, 
                   train_dataset, 
                   valid_dataset=None, 
                   n_epochs=10, 
                   lr=1e-3, 
                   save_dir="checkpoints", 
                   save_name="model.pth",
                   device="cuda" if torch.cuda.is_available() else "cpu",
                   continue_from_epoch=0,
                   continue_training=False,
                   ):
    if continue_training:
        model.load_state_dict(torch.load(os.path.join(save_dir, save_name.split(".")[0]+"_epoch_"+str(continue_from_epoch)+".pth")))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    valid_accuracies = []
    counter = 0
    for epoch in trange(continue_from_epoch, n_epochs + 1, desc="Training", disable=continue_training):
        model.train()
        running_loss = 0.0

        # Train
        with tqdm(total=len(train_dataset), desc="Training", disable=continue_training) as bar:
            for inputs, labels in train_dataset:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.ndim > 2:
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                writer.add_scalar('Loss/batch_train', loss.item(), counter)
                counter += 1

                bar.update(1)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Valid
        if valid_dataset is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                with tqdm(total=len(valid_dataset), desc="Validating", disable=continue_training) as bar:
                    for inputs, labels in valid_dataset:
                        inputs = inputs.to(device, dtype=torch.float32)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        if outputs.ndim > 2:
                            outputs = outputs.view(-1, outputs.size(-1))
                            labels = labels.view(-1)
                        preds = outputs.argmax(dim=-1)
                        correct += (preds == labels).sum().item()
                        total += labels.numel()

                        bar.update(1)

            valid_acc = correct / total
            valid_accuracies.append(valid_acc)

            writer.add_scalar('Accuracy/val', valid_acc, epoch)
            
            print(f"Epoch [{epoch}/{n_epochs}] Loss: {epoch_loss:.4f} Valid Acc: {valid_acc:.4f}")
        else:
            print(f"Epoch [{epoch}/{n_epochs}] Loss: {epoch_loss:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, save_name.split(".")[0]+"_epoch_"+str(epoch)+".pth"))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()

    if valid_dataset is not None:
        plt.subplot(1,2,2)
        plt.plot(valid_accuracies, label='Valid Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Valid Accuracy')
        plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.show()
    writer.add_figure("Training Plot", plt.gcf(), global_step=n_epochs, close=True)
    plt.close()

    print("Training complete. Plot saved as training_plot.png")
    return train_losses, valid_accuracies


if __name__=="__main__":    
    # data_info = Utils.build_database_info(args.root_path)
    with open(os.path.join(args.info_path), "r") as f:
        data_info = pd.from_csv(f)

    train_split = int(data_info["subject_id"].nunique()*0.7)
    valid_split = int(data_info["subject_id"].nunique()*0.2)
    test_split = data_info["subject_id"].nunique() - (train_split+valid_split)

    train_sub_idx = list(range(train_split))
    valid_sub_idx = list(range(train_split, train_split+valid_split))
    test_sub_idx = list(range(train_split+valid_split, train_split+valid_split+test_split))

    writer.add_text("Splits/train_subjects", str(data_info["subject_id"].values[train_sub_idx]))
    writer.add_text("Splits/valid_subjects", str(data_info["subject_id"].values[valid_sub_idx]))
    writer.add_text("Splits/test_subjects", str(data_info["subject_id"].values[test_sub_idx]))

    # Train
    info_idx = [i for i, sub in enumerate(data_info["subject_id"].values) if sub in train_sub_idx]
    sampler = Utils.SubjectBatchSampler(data_info.iloc[info_idx], args.batch_size)
    dataset = Utils.SubjectObjectDataset(data_info, args.root_path)
    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8, prefetch_factor=4, pin_memory=True)

    # Valid
    info_idx = [i for i, sub in enumerate(data_info["subject_id"].values) if sub in valid_sub_idx]
    sampler = Utils.SubjectBatchSampler(data_info.iloc[info_idx], args.batch_size)
    dataset = Utils.SubjectObjectDataset(data_info, args.root_path)
    valid_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=6, prefetch_factor=2, pin_memory=True)

    # Test
    info_idx = [i for i, sub in enumerate(data_info["subject_id"].values) if sub in test_sub_idx]
    sampler = Utils.SubjectBatchSampler(data_info.iloc[info_idx], args.batch_size)
    dataset = Utils.SubjectObjectDataset(data_info, args.root_path)
    test_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)

    print(f"Dataset split: {len(train_loader)} train, {len(valid_loader)} val, {len(test_loader)} test")

    eeg_model = Model.EEGClassifierModel()

    train_and_plot(eeg_model, 
                   train_loader, 
                   valid_loader, 
                   n_epochs=args.n_epochs, 
                   lr=args.lr, 
                   continue_training=args.continue_training, 
                   continue_from_epoch=args.continue_from_epoch,
                   device=args.device,
                   save_dir=args.save_dir,
                   save_name=args.save_name,
                   )

    # After training, you can evaluate on test_dataset
    eeg_model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name.split(".")[0]+"_epoch_"+str(args.n_epochs)+".pth")))
    eeg_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device, dtype=torch.float32)
            labels = labels.to(args.device)
            outputs = eeg_model(inputs)
            if outputs.ndim > 2:
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            writer.add_scalar('Accuracy/test_batch', correct / total, args.n_epochs)

    test_acc = correct / total
    writer.add_scalar('Accuracy/test_final', test_acc, args.n_epochs)
    print(f"Test Accuracy: {test_acc:.4f}")

    writer.close()
