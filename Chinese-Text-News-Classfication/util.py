from threading import main_thread
from numpy.lib.twodim_base import tri
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch._C import dtype
import jieba
import pickle as pkl
import numpy as np
import torch
import json
import torch.nn as nn
from importlib import import_module

max_size = 32
emb_dim = 300
vocab_dir = "./data/vocab.pkl"
filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
UNK, PAD = '<UNK>', '<PAD>'


# x = import_module('model.TextCNN')
# Config = x.Config
# config = Config()
# TextCnn = x.Model
# textcnn = TextCnn(config=config)

class Dic():
    def __init__(self, dic=None) -> None:
        if dic:
            self.dic = dic
            self.count = len(dic)
        else:
            self.dic = dict()
            self.count = -1
        self.add_word('PAD')
        self.add_word('UNK')

    def add_word(self, word):
        if word in self.dic.keys():
            return self.dic[word]
        else:
            self.count += 1
            self.dic[word] = self.count
            return self.dic[word]

    def get_word(self, word):
        if word in self.dic.keys():
            return self.dic[word]
        else:
            return self.dic['UNK']

    def return_len(self):
        return self.count


def make_dic():
   
    pass


def load_sougou(config):

    word_to_id = pkl.load(open(vocab_dir, 'rb'))
    embeddings = np.random.rand(len(word_to_id), emb_dim)

    with open(config.SouGou, 'r', encoding=' utf-8')as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if line[0] in word_to_id:
                idx = word_to_id[line[0]]
                emb = [float(x) for x in line[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        print(embeddings.shape)
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

# load_sougou()


def tokenizer(line, word):
  
    if word:
        return [y for y in line]
    else:
        return [y for y in jieba.cut(line)]


def load_data(data_name, word, config):

    contents = []
    data = []
    label = []
    embedding = pkl.load(open(vocab_dir, 'rb'))
    path = f'./data/{data_name}.txt'
    with open(path, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            s = line.split('\t')
            line = tokenizer(s[0], word)
            seq_len = len(line)
            if len(line) < max_size:
                line.extend([PAD]*(max_size-len(line)))
            else:
                line = line[:max_size]
                seq_len = max_size
            em_word = []
            for word in line:
             
                em_word.append(embedding.get(word, embedding.get(UNK)))
            # print(em_word)
            # em_word = torch.tensor(em_word)
            contents.append([em_word, int(s[1]), seq_len])
            data.append([em_word])
            label.append(int(s[1]))
    data = torch.tensor(data)
    label = torch.tensor(label)
    data = torch.squeeze(data, dim=1)
    label.view(-1, 1)
    if config.cuda_is_aviable:
        data = data.cuda(device=config.cuda_device)
        label = label.cuda(device=config.cuda_device)
    return contents, data, label


def save_FastText_dict(word_is_true_or_flase):
    def token(line, num):
 
        new = []
        for i in range(len(line)):
            if i+1 >= num:
                word = ''
                for j in range(1, num+1):
                    word = word+line[i-num+j]
                new.append(word)
        return new

    def n_grame(dic, line):
 
        biline = token(line, 2)
        for word in biline:
            dic.add_word(word)

    dic = Dic()
    with open('./data/train.txt', 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            s = line.split('\t')
            line = tokenizer(s[0], word_is_true_or_flase)
            n_grame(dic, line)

    with open('./data/test.txt', 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            s = line.split('\t')
            line = tokenizer(s[0], word_is_true_or_flase)
            n_grame(dic, line)

    with open('./data/dev.txt', 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            s = line.split('\t')
            line = tokenizer(s[0], word_is_true_or_flase)
            n_grame(dic, line)

    print(dic.return_len())
    with open('./data/fast_text_dict.txt', 'w', encoding='utf-8')as f:
        f.write(json.dumps(dic.dic))


def FastText_dataloader(data_name, word_is_true_or_flase, config):

    contents = []
    data = []
    bi_data = []
    tri_data = []
    label = []
    all_dic = ''
    embedding = pkl.load(open(vocab_dir, 'rb'))
    path = f'./data/{data_name}.txt'

    def token(line, num):

        new = []
        for i in range(len(line)):
            if i+1 >= num:
                word = ''
                for j in range(1, num+1):
                    word = word+line[i-num+j]
                new.append(word)
        return new

    with open('./data/fast_text_dict.txt', 'r', encoding='utf-8')as f:
        s = f.read()
        dic = json.loads(s)
        all_dic = Dic(dic)

    with open(path, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip().replace('\n', '')
            s = line.split('\t')
            line = tokenizer(s[0], word_is_true_or_flase)
            origin_line = line
            if len(line) < max_size:
                line.extend([PAD]*(max_size-len(line)))
            else:
                line = line[:max_size]
            em_word = []
            for word in line:

                em_word.append(int(embedding.get(word, embedding.get(UNK))))
            origin_emb = em_word


            biline = token(origin_line, 2)
            em_word = []
            for word in biline:
                idx_word = all_dic.get_word(word)
                em_word.append(idx_word)
            if len(em_word) < max_size:
                em_word.extend([all_dic.get_word('PAD')]
                               * (max_size-len(biline)))
            else:
                em_word = em_word[:max_size]
            bi_emb = em_word

         

            data.append(origin_emb)
            bi_data.append(bi_emb)
            # tri_data.append(tri_emb)
            label.append(int(s[1]))
    data = torch.tensor(data).int()
    bi_data = torch.tensor(bi_data).long()
    # tri_data = torch.tensor(tri_data).long()

    label = torch.tensor(label)
    data = torch.squeeze(data, dim=1)
    bi_data = torch.squeeze(bi_data)
    # tri_data = torch.squeeze(tri_data)
    label.view(-1, 1)
    m, n = data.shape
    d = torch.zeros((3, m, n), dtype=torch.long)

    if config.cuda_is_aviable:
        data = data.cuda(device=config.cuda_device)
        label = label.cuda(device=config.cuda_device)
        bi_data = bi_data.cuda(device=config.cuda_device)
        # tri_data = tri_data.cuda(device=config.cuda_device)
        d = d.cuda(device=config.cuda_device)
    # data = [data, bi_data, tri_data]


    d[0] = data
    d[1] = bi_data
    # d[2] = tri_data
    data = d.permute(1, 0, 2)
    return data, label


def data_loader(data, label, config):

    Data = TensorDataset(data, label)
    train_loader = DataLoader(
        dataset=Data, batch_size=config.batch_size, shuffle=config.shuffle)
    return train_loader


if __name__ == "__main__":

    from model.TextRNN import Config
    config = Config()
    load_sougou(config=config)
