import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Config():
    def __init__(self) -> None:
        self.model_name = 'TextCNN'
        self.data_path = './data/'
        self.embedding_path = "./data/embedding_SougouNews.npz"
        self.model_save_path = './save/TextCnnModel.pkl'
        self.SouGou = './data/sgns.sogou.char'
        self.log_path = './log/SummaryWriter'
        self.dropout = 0.5
        self.num_classes = 10
        self.batch_size = 10000
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


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=False)
        self.convs = nn.ModuleList(
            [nn.Conv1d(config.embed, config.num_filters, k) for k in config.filter_sizes])
     
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters *
                            len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze()
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
       
        return x

    def forward(self, x):
        out = self.embeddings(x)
      
        out = out.permute(0, 2, 1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
