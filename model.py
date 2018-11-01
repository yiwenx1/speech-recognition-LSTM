import torch
import torch.nn as nn
from torch.nn.utils import rnn


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
        self.dense_layer = DenseLayer(hidden_size, class_size)

    def forward(self, utterance_list):
        batch_size = len(utterance_list)
        inputs_length = [len(utterance) for utterance in utterance_list]

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



