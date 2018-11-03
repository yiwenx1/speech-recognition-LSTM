import torch
import torch.nn as nn
from torch.nn.utils import rnn
from ctcdecode import CTCBeamDecoder
import torch.nn.functional as F
import Levenshtein as L

from data.phoneme_list import PHONEME_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self):
        self.embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21,11), stride=(2,1), padding=(10,5)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=input_size)
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, padded_output):
        longest_length = padded_output.size(0)
        batch_size = padded_output.size(1)
        result = padded_output.view(longest_length*batch_size, -1)
        result = self.bn(result)
        result = self.fc(result)
        result = result.view(longest_length, batch_size, -1)
        return result


class PackedLanguageModel(nn.Module):

    def __init__(self, class_size, input_size, hidden_size, nlayers):
        super(PackedLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True)
        self.dense_layer = DenseLayer(hidden_size, class_size + 1)

    def forward(self, utterance_list):
        batch_size = len(utterance_list)
        inputs_length = [len(utterance) for utterance in utterance_list]
        inputs_length = torch.IntTensor(inputs_length)

        # Packs a list of variable length Tensors.
        packed_input = rnn.pack_sequence(utterance_list)
        # print("packed_input:", packed_input.data.shape)
        hidden = None
        packed_output, hidden = self.rnn(packed_input, hidden)
        # padded_output: longest_length * batch_size * (2*hidden_size)
        padded_output, _ = rnn.pad_packed_sequence(packed_output) # unpack output (padded)

        longest_length = padded_output.size(0)
        # padded_output: longest_length * batch_size * hidden_size
        padded_output = padded_output.view(longest_length, batch_size, 2, -1).sum(2).view(longest_length, batch_size, -1)
        # print("padded_output", padded_output.shape)

        # score: longest_length * batch_size * class_size
        score = self.dense_layer(padded_output)
        # print("score", score.shape)

        return score, inputs_length

class ER():
    def __init__(self):
        self.label_map = [' '] + PHONEME_MAP
        self.decoder = CTCBeamDecoder(labels=self.label_map, blank_id=0)

    def __call__(self, outputs, targets, outputs_size, targets_size):
        return self.forward(outputs, targets, outputs_size, targets_size)

    def forward(self, outputs, targets, outputs_size, targets_size):
        outputs = torch.transpose(outputs, 0, 1)
        probs = F.softmax(outputs, dim=2) # outputs: batch_size * seq_len * 47
        # print(outputs_size)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=outputs_size)
        pos = 0
        ls = 0
        print(targets_size)
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in targets[pos:pos + targets_size[i]])
            print("pred {}\ntarget {}".format(pred, true))
            pos += targets_size[i]
            print("pred {} true {}".format(len(pred),len(true)))
            ls += L.distance(pred, true)
        assert pos == targets.size(0)
        return ls / output.size(0)


class ER_test():
    def __init__(self):
        self.label_map = [' '] + PHONEME_MAP
        self.decoder = CTCBeamDecoder(labels=self.label_map, blank_id=0)

    def __call__(self, outputs, targets, outputs_size, targets_size):
        return self.forward(outputs, targets, outputs_size, targets_size)

    def forward(self, outputs, targets, outputs_size, targets_size):
        outputs = torch.transpose(outputs, 0, 1)
        probs = F.softmax(outputs, dim=2) # outputs: batch_size * seq_len * 47
        print(outputs_size)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=outputs_size)
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            print("pred {}".format(pred))
            print("pred {}".format(len(pred)))
        return pred
