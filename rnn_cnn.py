import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class rnn_cnn(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_classes, isbirnn, dropout, kernels_size, features):
        super(rnn_cnn, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.isbirnn = isbirnn

        self.rnn = nn.LSTM(
            input_size = embedding_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = isbirnn,
            dropout = dropout
        )

        features_num = features

        self.convs1 = nn.ModuleList([nn.Conv2d(1, features_num, (K, embedding_size)) for K in kernels_size])
        # Conv2d(in_channel, out_channel,kernel_size)

        self.dropout = nn.Dropout(dropout)

        if self.isbirnn:
            self.out = nn.Linear(hidden_size * 2 + len(kernels_size) * features_num, num_classes)
        else:
            self.out = nn.Linear(hidden_size + len(kernels_size) * features_num, num_classes)

    def forward(self, x):
        if self.isbirnn:
            h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0),self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        rnn_input = x

        rnn_out, _ = self.rnn(rnn_input, (h0, c0))

        cnn_input = torch.unsqueeze(x, 1)

        cnn_out = [F.relu(conv(cnn_input)).squeeze(3) for conv in self.convs1]

        cnn_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]

        cnn_out = torch.cat(cnn_out , 1)

        rnn_out = rnn_out[:, 8, :]

        out = torch.cat((cnn_out, rnn_out), 1)

        out = self.out(out)

        return out
