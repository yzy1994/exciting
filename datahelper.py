import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def lookup_word(self, word):
        if word not in self.word2idx:
            return self.word2idx['unk']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def loadWord2Vec(filename):
    dict = Dictionary()
    embd = []
    fr = open(filename, 'r')
    line = fr.readline().decode('utf-8').strip()
    # print line
    word_dim = int(line.split(' ')[1])
    dict.add_word("unk")
    embd.append([0] * word_dim)
    for line in fr:
        row = line.strip().split(' ')
        dict.add_word(row[0])
        vec = []
        for v in row[1:]:
            vec.append(np.float32(v))
        embd.append(vec)
    print "loaded word2vec"
    fr.close()
    return dict, embd

def load_file(filename, dict, embd, seq_len, embedding_size):
    input_x = []
    input_y = []
    fr = open(filename,'r')
    line_num = 0
    for line in fr.readlines():
        line_num = line_num +1
        x_y = line.split('\t')
        x_raw = x_y[0].strip()
        y = float(x_y[1][0])
        input_y.append(y)
        line_list = []
        words_raw = x_raw.split(' ')
        for word_raw in words_raw:
            word_raw = word_raw.strip()
            word_fix = word_raw.split('/')
            if len(word_fix)!=2:
                continue

            real_word = word_fix[0].strip()
            suffix = word_fix[1].strip()
            word_embd = []
            word_embd.extend(embd[dict.lookup_word(real_word)])
            fixnum = 0
            for ch in suffix:
                fixnum=fixnum+1
                if fixnum ==15:
                    break
                if ch=='1':
                    word_embd.append(1.0)
                else:
                    word_embd.append(0.0)
            line_list.extend(word_embd)
        input_x.extend(line_list)
    input_x = np.array(input_x).reshape(line_num, seq_len, embedding_size)


    return input_x, input_y
