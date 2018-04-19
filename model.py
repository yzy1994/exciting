import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class rnn_cnn(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_classes, isbirnn, dropout, kernels_size, features, use_rnn, use_cnn, vocab_size, word_embedding, add_size):
        super(rnn_cnn, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.embedding_layer.weight.data.copy_(word_embedding)

        self.use_rnn = use_rnn
        self.use_cnn = use_cnn
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.isbirnn = isbirnn
        self.add_size = add_size

        self.rnn = nn.LSTM(
            input_size = embedding_size+add_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = isbirnn,
            dropout = dropout
        )

        features_num = features

        self.convs1 = nn.ModuleList([nn.Conv2d(1, features_num, (K, embedding_size+add_size)) for K in kernels_size])
        # Conv2d(in_channel, out_channel,kernel_size)

        self.dropout = nn.Dropout(dropout)

        self.hidden_out_size = 0
        if self.use_rnn:
            if self.isbirnn:
                self.hidden_out_size += hidden_size*2
            else:
                self.hidden_out_size += hidden_size
        if self.use_cnn:
            self.hidden_out_size += len(kernels_size) * features_num

        self.out = nn.Linear(self.hidden_out_size, num_classes)

    def forward(self, x):
        x = x.long()

        x_add = x[:,:,1:]
        x = x[:,:,:1]
        x = torch.squeeze(x, 2)

        x = self.embedding_layer(x)
        if self.isbirnn:
            h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0),self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        x_add = x_add.float()
        x = torch.cat((x, x_add), 2)

        rnn_input = x

        if self.use_rnn:
            rnn_out, _ = self.rnn(rnn_input, (h0, c0))
            rnn_out = rnn_out[:, 8, :]

        if self.use_cnn:
            cnn_input = torch.unsqueeze(x, 1)

            cnn_out = [F.relu(conv(cnn_input)).squeeze(3) for conv in self.convs1]

            cnn_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]

            cnn_out = torch.cat(cnn_out , 1)

        if self.use_rnn and self.use_cnn:
            out = torch.cat((cnn_out, rnn_out), 1)
        elif self.use_rnn:
            out = rnn_out
        else:
            out = cnn_out

        out = self.out(out)
        return out
