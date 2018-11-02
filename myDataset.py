import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class myDataset(Dataset):
    def __init__(self, train_path, label_path):
        self.train_data = np.load(train_path, encoding="bytes")
        self.label = np.load(label_path, encoding="bytes")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        inputs = self.train_data[i]
        labels = self.label[i]
        labels = np.add(labels, 1)
        return torch.from_numpy(inputs).to(DEVICE), torch.from_numpy(labels).to(DEVICE)

def collate(utterance_list):
    batch_size = len(utterance_list)
    inputs, targets = zip(*utterance_list)
    lens = [len(utterance) for utterance in inputs]
    # index order
    utterance_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in utterance_order]
    targets = [targets[i] for i in utterance_order]

    out_targets = []
    targets_size = torch.IntTensor(batch_size)
    for i in range(batch_size):
        if i == 0:
            out_targets = targets[i]
        else:
            out_targets = torch.cat((out_targets, targets[i]))
        targets_size[i] = len(targets[i])

    return inputs, out_targets, targets_size

if __name__ == '__main__':
    # train_path = "./data/wsj0_train.npy"
    # label_path = "./data/wsj0_train_merged_labels.npy"
    # train_dataset = myDataset(train_path, label_path)
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate)
    # dev_path = "./data/wsj0_dev.npy"
    dev_label_path = "./data/wsj0_dev_merged_labels.npy"
    # dev_dataset = myDataset(dev_path, dev_label_path)

    test_path = "./data/wsj0_test.npy"
    test_dataset = myDataset(test_path, dev_label_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

    print(test_dataset.__len__())
    for i, (inputs, targets, targets_size) in enumerate(test_loader):
        if i == 0:
            print(len(inputs), inputs[0].shape)
