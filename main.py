import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import datahelper
import numpy as np
import rnn_cnn
from logger import Logger

#Hyper Parameters###########
EMBEDDING_SIZE = 46 #32+5+9
SEQENCE_LENGTH = 17

EPOCH = 100
LR = 0.005
BATCH_SIZE = 64
W2V_PATH = "data/cec.vector"
TRIANSET_PATH = 'data/cec.train'
TESTSET_PATH = 'data/cec.test'
DROPOUT_RATE = 0.5

RNN_HIDDEN_SIZE = 64
NUM_CLASSES = 8
NUM_LAYERS = 2  #LSTM-unit
IS_BIRNN = True

USE_CNN = False
KERNELS_SIZE = [3,4,5]
FEATURES_PER_FILTER = 3

############################

#loading pre-trained word2vec
dict, embd = datahelper.loadWord2Vec(W2V_PATH)


raw_x, raw_y = datahelper.load_file(TRIANSET_PATH,dict,embd,SEQENCE_LENGTH,EMBEDDING_SIZE)
np_x = raw_x.astype(np.float32)
np_y = np.array(raw_y).astype(np.int64)
input_y = torch.from_numpy(np_y)
input_x = torch.from_numpy(np_x)

torch_dataset = Data.TensorDataset(data_tensor=input_x, target_tensor=input_y)

test_raw_x, test_raw_y = datahelper.load_file(TESTSET_PATH,dict,embd, SEQENCE_LENGTH,EMBEDDING_SIZE)
test_np_x = test_raw_x.astype(np.float32)
test_np_y = np.array(test_raw_y).astype(np.int64)
test_input_y = torch.from_numpy(test_np_y)
test_input_x = torch.from_numpy(test_np_x)
test_dataset = Data.TensorDataset(data_tensor=test_input_x, target_tensor=test_input_y)


train_loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False
)

#model = rnn.RNN(input_size=EMBEDDING_SIZE, hidden_size=RNN_HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,isbirnn=IS_BIRNN, dropout=DROPOUT_RATE)
model = rnn_cnn.rnn_cnn(EMBEDDING_SIZE, RNN_HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, IS_BIRNN, DROPOUT_RATE, KERNELS_SIZE, FEATURES_PER_FILTER)

logger = Logger('./logs')

optimizer = optim.Adam(model.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # gives batch data
        b_x = Variable(x)  # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)  # batch y

        #changguicaozuo
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    count = 0
    total = 0
    r_count = 0
    r_total = 0
    reco_count = 0
    for test_input_x,test_input_y in test_loader:

        t_x = Variable(test_input_x)
        test_output = model(t_x)
        pred = torch.max(test_output, 1)[1].data.numpy().squeeze()

        if test_input_y.numpy() != 0:
            r_total += 1.0
            if pred == test_input_y.numpy():
                r_count += 1.0
            if pred != 0:
                reco_count += 1.0

        if pred == test_input_y.numpy():
            count += 1.0

        total += 1.0

    accuracy = count/total
    recall = r_count/r_total
    rec_rate = reco_count/r_total
    print ('Epoch :', epoch, 'Loss : %.4f' % loss.data[0] ," Accuracy: %.4f" % accuracy,
           " Recall: %.4f " % recall, ' Rec_rate: %.4f ' % rec_rate)

    info = {
        'loss' : loss.data[0],
        'accuracy' : accuracy,
        'recall' : recall,
        'rec_rate' : rec_rate,
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)



