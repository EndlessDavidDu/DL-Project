import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Config():
    def __init__(self) -> None:
        self.model_name = 'TextRNN'
        self.data_path = './data/'
        self.embedding_path = "./data/embedding_SougouNews.npz"
        self.model_save_path = './save/TextRnnModel.pkl'
        self.SouGou = './data/sgns.sogou.char'
        self.log_path = './log/SummaryWriter'
        self.dropout = 0.5
        self.num_classes = 10
        self.batch_size = 10000
        self.shuffle = True
        self.cuda_is_aviable = True
        self.cuda_device = 1
        self.learning_rate = 1e-4
        self.epoch = 500
        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding_path)['embeddings'], dtype=torch.float)
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.out = 256
        self.hidden_num = 2                                                 


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(input_size=config.embed,
                            hidden_size=config.out, num_layers=config.hidden_num, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(32 * config.out, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.embeddings(x)
        output, (hn, cn) = self.lstm(out)
        out = output.reshape(-1, output.size(1) * output.size(2))
        out = self.fc(out)
        out = self.softmax(out)
        return out
