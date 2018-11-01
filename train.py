import numpy as np
import torch
from torch.utils.data import DataLoader

from myDataset import myDataset, collate
from model import PackedLanguageModel
from data.phoneme_list import N_PHONEMES
from config import MODEL_CONFIG as MC

def train(train_loader, model, optimizer, ctc):
    for i, (inputs, targets, targets_size) in enumerate(train_loader):
        print(i)
        optimizer.zero_grad()
        print("inputs:", len(inputs), inputs[0].shape, inputs[1].shape)
        outputs, outputs_size = model(inputs)
        print(outputs)
        print("targets", targets)
        loss = ctc(outputs, targets, outputs_size, targets_size)
        # loss.backward()
        # optimizer.step()



def main():

    train_path = "./data/wsj0_train.npy"
    label_path = "./data/wsj0_train_merged_labels.npy"
    train_dataset = myDataset(train_path, label_path)
    train_loader = DataLoader(train_dataset, batch_size=MC["batch_size"], shuffle=False, collate_fn=collate)

    model = PackedLanguageModel(N_PHONEMES, 40, 256, 3)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # ctc = CTCCriterion()
    ctc = 0
    start_epoch = 0

    nepochs = MC["nepochs"]
    for epoch in range(start_epoch, nepochs):
        model.train()
        train(train_loader, model, optimizer, ctc)

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()
