import numpy as np
import torch
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss

from myDataset import myDataset, collate
from model import PackedLanguageModel
from data.phoneme_list import N_PHONEMES
from config import MODEL_CONFIG as MC

def train(train_loader, model, optimizer, ctc, epoch):
    loss_sum = 0
    for i, (inputs, targets, targets_size) in enumerate(train_loader):
        print(i)
        optimizer.zero_grad()
        print("inputs:", len(inputs), inputs[0].shape, inputs[1].shape)
        # outputs: # longest_length * batch_size * nclasses
        outputs, outputs_size = model(inputs)
        print("outputs", outputs.shape)
        print("targets", targets.shape)
        targets = targets.to(torch.int32)
        print(targets.dtype, outputs_size.dtype, targets_size.dtype)
        loss = ctc(outputs, targets.to("cpu"), outputs_size.to("cpu"), targets_size.to("cpu"))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if i % 10 == 0:
            print("epoch {}, step {}, loss per step {}, finish {}".format(
                epoch, i, loss_sum/10, (i+1)*len(inputs)))
        if i % 100 == 0:
            save_model()

 main():

    train_path = "./data/wsj0_train.npy"
    label_path = "./data/wsj0_train_merged_labels.npy"
    train_dataset = myDataset(train_path, label_path)
    train_loader = DataLoader(train_dataset, batch_size=MC["batch_size"], shuffle=False, collate_fn=collate)

    model = PackedLanguageModel(N_PHONEMES, 40, 256, 3)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    ctc = CTCLoss()
    start_epoch = 0

    nepochs = MC["nepochs"]
    for epoch in range(start_epoch, nepochs):
        model.train()
        train(train_loader, model, optimizer, ctc)

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()
