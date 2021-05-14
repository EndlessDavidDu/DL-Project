import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable


class Config():
    def __init__(self) -> None:
        self.model_name = 'Transfermor'
        self.data_path = './data/'
        self.embedding_path = "./data/embedding_SougouNews.npz"
        self.model_save_path = './save/TextRCNNModel.pkl'
        self.SouGou = './data/sgns.sogou.char'
        self.log_path = './log/SummaryWriter'

        self.dropout = 0.5
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
        self.sequence_length = 32
        self.num_head = 6
        self.num_encoder = 3
        self.hidden = 1024
        self.last_hidden = 512


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=False)
        self.position = Positional_Encoding(
            config.sequence_length, config.embed, config.cuda_device, config.cuda_is_aviable, config.dropout)
        self.encoder = Encoder(config.embed, config.num_head, config.batch_size,
                               config.dropout, config.hidden, config.last_hidden)
        self.models = nn.ModuleList(
            [copy.deepcopy(self.encoder) for i in range(config.num_encoder)])
        self.fc = nn.Linear(
            config.embed * config.sequence_length, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position(out)
        for model in self.models:
            out = model(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Encoder(nn.Module):
    def __init__(self, embed, num_head, batch_size, dropout, hidden, last_hidden):
        super().__init__()
        self.multi_attention = MutiHeadSelfAttention(
            embed, num_head, batch_size, dropout)
        self.feedforword = FeadForward(embed, hidden, last_hidden, dropout)

    def forward(self, x):
        out = self.multi_attention(x)
        out = self.feedforword(out)
        return out


class Positional_Encoding(nn.Module):
    
    def __init__(self, sequence_length, embed, cuda_device, cuda_is_aviable, dropout):
        super().__init__()
        self.cuda_device = cuda_device
        self.cuda_is_aviable = cuda_is_aviable

    
        self.be = torch.tensor([[pos / math.pow(10000.0, 2.0 * i / embed) for i in range(embed)]
                                for pos in range(sequence_length)])
        self.be[:, 0::2] = torch.sin(self.be[:, 0::2])
        self.be[:, 1::2] = torch.cos(self.be[:, 1::2])
        self.embed = embed
        self.be = nn.Parameter(self.be, requires_grad=False)
     
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        input: [batch_Size, seq_len, word_embed]
        return: [batch_Size, seq_len, word_embed]
        '''
        out = x * math.sqrt(self.embed)
        out = out + self.be
        out = self.dropout(out)
        return out


class MutiHeadSelfAttention(nn.Module):
    def __init__(self, embed, num_head, batch_size, dropout):
        super().__init__()
        self.batch_size = batch_size
        self.num_head = num_head
        assert embed % num_head == 0
        self.dim_head = embed // num_head

        self.Q = nn.Linear(embed, num_head * self.dim_head)
        self.K = nn.Linear(embed, num_head * self.dim_head)
        self.V = nn.Linear(embed, num_head * self.dim_head)

        self.fc = nn.Linear(num_head * self.dim_head, embed)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x):
        '''
        input: [batch_Size, seq_len, word_embed]
        return: [batch_Size, seq_len, word_embed]
        '''
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = Q.view(self.batch_size * self.num_head, -1, self.dim_head)
        K = K.view(self.batch_size * self.num_head, -1, self.dim_head)
        V = V.view(self.batch_size * self.num_head, -1, self.dim_head)

        # [self.batch_size * self.num_head, seq_len, seq_len]
        attentions = torch.matmul(Q, K.permute(0, 2, 1))
        attentions = F.softmax(attentions, dim=-1)

        scale = K.size(-1) ** -0.5
        out = torch.matmul(attentions*scale, V)

        out = out.view(self.batch_size, -1,  self.num_head * self.dim_head)
        out = nn.ReLU()(self.fc(out))
        out = self.dropout(out)

        out = out+x  
        out = self.norm(out)
        return out


class FeadForward(nn.Module):
    def __init__(self, embed, hidden, last_hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed, hidden)
        self.fc2 = nn.Linear(hidden, last_hidden)
        self.fc3 = nn.Linear(last_hidden, embed)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x):
        '''
        input: [batch_Size, seq_len, word_embed]
        return: [batch_Size, seq_len, word_embed]
        '''
        out = nn.ReLU()(self.fc1(x))
        out = nn.ReLU()(self.fc2(out))
        out = nn.ReLU()(self.fc3(out))
        out = self.dropout(out)

        out = x + out
        out = self.norm(out)
        return x
