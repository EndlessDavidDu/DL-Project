import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Config():
    def __init__(self) -> None:
        self.model_name = 'TextRCNN'
        self.data_path = './data/'
        self.embedding_path = "./data/embedding_SougouNews.npz"
        self.model_save_path = './save/TextRCNNModel.pkl'
        self.SouGou = './data/sgns.sogou.char'
        self.log_path = './log/SummaryWriter'
        self.dropout = 0.5
        self.num_classes = 10
        self.batch_size = 5000
        self.shuffle = True
        self.cuda_is_aviable = True
        self.cuda_device = 2
        self.learning_rate = 1e-4
        self.epoch = 500
        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding_path)['embeddings'], dtype=torch.float)
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.out = 256
        self.hidden_num = 2
        self.sequence_length = 32


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(input_size=config.embed,
                            hidden_size=config.out, num_layers=config.hidden_num, batch_first=True, dropout=config.dropout, bidirectional=True)
 
        self.maxpool = nn.MaxPool1d(
            config.sequence_length)
    
        self.fc = nn.Linear(config.out * 2 + config.embed, config.num_classes)

    def forward(self, x):

        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
