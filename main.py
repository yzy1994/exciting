import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import rnn_cnn
from datahelper import loadData, adjust_learning_rate
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import sys

#Parameters################33
WES = 64       #word_embedding_size
ADD_SIZE = 14
VOCAB_SIZE = 7959 #word_embedding_num
SEQUENCE_LENGTH = 17

EPOCH = 30
LR = 0.02 #learning rate
LR_DECAY_RATE = 0.5
BATCH_SIZE = 400
VOCAB_PATH = './data/vocab.pkl' #list [word2idx(Dictionary),embedding([[]])]
DATA_PATH = './data/event_input'
DROPOUT_RATE = 0.4
NUM_CLASSES = 7

#RNN Parameters
USE_RNN = True
RNN_HIDDEN_SIZE = 80
NUM_LAYERS = 1
IS_BIRNN = True

#CNN Parameters
USE_CNN = True
KERNELS_SIZE = [3,4,5]
FEATURES_PER_FILTER = 10

#
TARGET_SET = ['statement', 'operation', 'emergency','perception', 'stateChange', 'movement', 'None']

if __name__=="__main__":
    foutput = None
    if len(sys.argv)!=1 :
        FEATURES_PER_FILTER = int(sys.argv[1].strip())
        EXP_NO = sys.argv[2]
        print sys.argv[1]
        print sys.argv[2]
        foutput = open('./log/fn_'+str(FEATURES_PER_FILTER)+"expn_"+EXP_NO, 'w')

    #LOAD VOCAB##################################
    datalist = pickle.load(open(VOCAB_PATH, 'r'))
    word2idx = datalist[0]
    vector_list = datalist[1]
    w_embedding = torch.from_numpy(np.array(vector_list))
    ################################################

    train_x, train_y, test_x, test_y = loadData(DATA_PATH)

    torch_trainset = Data.TensorDataset(data_tensor=train_x, target_tensor=train_y)

    train_loader = Data.DataLoader(dataset=torch_trainset, batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    model = rnn_cnn(WES, RNN_HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, IS_BIRNN, DROPOUT_RATE, KERNELS_SIZE, FEATURES_PER_FILTER, USE_RNN, USE_CNN, VOCAB_SIZE, w_embedding, ADD_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)

            ###
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%5 == 0:
            adjust_learning_rate(optimizer, LR_DECAY_RATE)
            b_x = Variable(test_x)
            output = model(b_x)
            _, pred = torch.max(output, 1)

            print '-----eval-------'
            pred_numpy = pred.data.numpy()
            y_numpy = test_y.numpy()
            print classification_report(y_numpy, pred_numpy, target_names=TARGET_SET, digits=4)
            print confusion_matrix(y_numpy, pred_numpy)
            if foutput is not None:
                foutput.write(classification_report(y_numpy, pred_numpy, target_names=TARGET_SET, digits=4))



