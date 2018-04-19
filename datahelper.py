import pickle
import torch
import random
import numpy as np
VOCAB_DIR = './data/vocab.pkl'
datalist = pickle.load(open(VOCAB_DIR, 'r'))
word2idx = datalist[0]
CATEGORY_DICT = {'statement':0, 'action':1, 'emergency':2,'perception':3,
                 'stateChange':4, 'movement':5, 'None':6}
TEST_RATIO = 0.2

def loadData(filedir):
    fopen = open(filedir, 'r')
    data_list = []
    for line in fopen.readlines():
        contents = line.split('\t')
        category = contents[0].strip()
        event_content = contents[1].strip()
        #x = [word2idx[w] for w in event_content.split(' ')]
        y = CATEGORY_DICT[category]

        addition_seq = contents[2].strip()
        x_addition_list = []
        words = event_content.split(' ')
        additions = addition_seq.split(' ')

        for i in range(len(words)):
            x_addition = [word2idx[words[i]]]
            for j in range(len(additions[i])):
                x_addition.append(int(additions[i][j:j+1]))
            x_addition_list.append(x_addition)

        data_list.append([x_addition_list, y])

    random.shuffle(data_list)
    test_num = int(len(data_list)*TEST_RATIO)
    train_list = data_list[:-test_num]
    test_list = data_list[-test_num:]
    train_x = [x_y[0] for x_y in train_list]
    train_y = [x_y[1] for x_y in train_list]

    train_x_tensor = torch.from_numpy(np.array(train_x))
    train_y_tensor = torch.from_numpy(np.array(train_y))
    test_x = [x_y[0] for x_y in test_list]
    test_y = [x_y[1] for x_y in test_list]
    test_x_tensor = torch.from_numpy(np.array(test_x))
    test_y_tensor = torch.from_numpy(np.array(test_y))

    return train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor

def adjust_learning_rate(optimizer, decay_rate= 0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
