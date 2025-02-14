import numpy as np
import torch
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss
import argparse
import torch.nn as nn
import os

from myDataset import myDataset, collate
from model import PackedLanguageModel, ER, ER_test
from data.phoneme_list import N_PHONEMES
from config import MODEL_CONFIG as MC

def train(train_loader, model, optimizer, ctc, epoch):
    loss_sum = 0
    for i, (inputs, targets, targets_size) in enumerate(train_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # print("inputs:", len(inputs), inputs[0].shape, inputs[1].shape)
        # outputs: # longest_length * batch_size * nclasses
        outputs, outputs_size = model(inputs)
        # print("outputs", outputs.shape)
        # print("targets", targets.shape)
        targets = targets.to(torch.int32)
        loss = ctc(outputs, targets.to("cpu"), outputs_size.to("cpu"), targets_size.to("cpu"))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if i % 10 == 0:
            print("epoch {}, step {}, loss per step {}, finish {}".format(
                epoch, i, loss_sum/10, (i+1)*len(inputs)))
            loss_sum = 0
        if i % args.checkpoint == 0:
            save_model(epoch, model, optimizer, loss, i, "./weights/")

def eval(dev_loader, model, epoch):
    error_rate_op = ER()
    for i, (inputs, targets, targets_size) in enumerate(dev_loader):
        outputs, outputs_size = model(inputs)
        targets = targets.to(torch.int32)
        error = error_rate_op(outputs.to("cpu"), targets.to("cpu"), outputs_size.to("cpu"), targets_size)
        print(error)
    return error

def test(test_loader, model, path_name):
    error_rate_op = ER_test()
    predictions = []
    for i, (inputs, targets, targets_size) in enumerate(test_loader):
        outputs, outputs_size = model(inputs)
        targets = targets.to(torch.int32)
        prediction = error_rate_op(outputs.to("cpu"), targets.to("cpu"), outputs_size.to("cpu"),targets_size)
        predictions.append(prediction)
        print(prediction)

    with open(path_name, 'w') as f:
        for i in range(len(predictions)):
            f.write(str(i) + "," + predictions[i] + "\n")


def save_model(epoch, model, optimizer, loss, step, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + "%.6f" % loss.item() + '.pth'
    print('Save model at Train Epoch: {} [Step: {}\tLoss: {:.12f}]'.format(
        epoch, step, loss.item()))
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

def load_model(epoch, step, loss, model, optimizer, save_path):
    filename = save_path + str(epoch) + '-' + str(step) + '-' + str(loss) + '.pth'
    if os.path.isfile(filename):
        print("######### loading weights ##########")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        print('########## loading weights done ##########')
        return model, optimizer, start_epoch, loss
    else:
        print("no such file: ", filename)

def main(args):
    train_path = "./data/wsj0_train.npy"
    label_path = "./data/wsj0_train_merged_labels.npy"
    train_dataset = myDataset(train_path, label_path)
    train_loader = DataLoader(train_dataset, batch_size=MC["batch_size"], shuffle=True, collate_fn=collate)

    dev_path = "./data/wsj0_dev.npy"
    dev_label_path = "./data/wsj0_dev_merged_labels.npy"
    dev_dataset = myDataset(dev_path, dev_label_path)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    test_path = "./data/wsj0_test.npy"
    test_dataset = myDataset(test_path, dev_label_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

    model = PackedLanguageModel(N_PHONEMES, 40, MC["hidden_size"], MC["nlayers"])
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MC["lr"], weight_decay=1e-6)
    ctc = CTCLoss()
    start_epoch = 0
    if args.resume is True:
        model, optimizer, start_epoch, loss = load_model(
            args.load_epoch,
            args.load_step,
            args.load_loss,
            model,
            optimizer,
            "./weights/"
        )

    nepochs = args.epochs
    for epoch in range(start_epoch, nepochs):
        model.train()
        train(train_loader, model, optimizer, ctc, epoch)
        # if (epoch+1) % 10 == 0:
            # model.eval()
            # with torch.no_grad():
                # error = eval(dev_loader, model, epoch)
    test(test_loader, model, "submission_10.csv")

def arguments():
    parser = argparse.ArgumentParser(description="Speaker Verificiation via CNN")
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='L2 regularization')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--checkpoint', type=int, default=100, metavar="R",
                        help='checkpoint to save model parameters')
    parser.add_argument('--resume', type=bool, default=False, metavar="R",
                        help='resume training from saved weight')
    parser.add_argument('--weights-path', type=str, default="./weights",
                        help='path to save weights')
    parser.add_argument('-load-epoch', type=str, default=0, metavar="LE",
                        help='number of epoch to be loaded')
    parser.add_argument('-load-step', type=str, default=0, metavar="LS",
                        help='number of step to be loaded')
    parser.add_argument('-load-loss', type=str, default=0, metavar="LL",
                        help='loss item to be loaded')
    return parser.parse_args()

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = arguments()
    main(args)
