import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm


peer_num = 15000
min_length = 80
max_length = 100


def get_data():
    path = '../data/'
    L = os.listdir(path)
    L.remove('.DS_Store')
    labels = dict()
    index = 0
    pool = ThreadPoolExecutor(max_workers=2)
    for item in L:
        labels[item] = index
        inner = os.listdir(path+item)
        # parse_txt(inner, item, index)
        pool.submit(parse_txt, inner, item, index)
        index += 1
    print(labels)


def parse_txt(inner, item, index):

    s = ""
    count = 0
    for inner_item in tqdm(inner):
        with open(f'../data/{item}/{inner_item}', 'r', encoding='utf-8')as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace(' ', '').replace('\t', '')
                line = line.replace(' ', '')
                if (min_length < len(line) < max_length) and (count < peer_num):
                    line = line+'\t'+str(index)
                    s = s+line+'\n'
                    count += 1
        # break
    path = f'../save_information/{item}.txt'
    save_data(s, path)


def save_data(data, path):
    with open(path, 'w')as f:
        f.write(data)


def save_dataset(datas, name):
    s = ""
    for line in datas:
        s = s + line
    path = f"../THUCNews/{name}.txt"
    save_data(s, path)


def train_test_split():
    datas = os.listdir('../save_information/')
    # datas.remove('.DS_Store')
    all_data = []
    for item in tqdm(datas):
  
        with open(f'../save_information/{item}', 'r', encoding='utf-8')as f:
            for line in f.readlines():
                all_data.append(line)

    import random
    random.shuffle(all_data)
    train_data = all_data[:len(all_data)-20000]
    test_data = all_data[len(all_data)-20000:len(all_data)-10000]
    eval_data = all_data[len(all_data)-10000:]

    save_dataset(train_data, 'train')
    save_dataset(test_data, 'test')
    save_dataset(eval_data, 'dev')


if __name__ == "__main__":
    # get_data()
    train_test_split()
