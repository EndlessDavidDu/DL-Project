import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


class Config():
    def __init__(self) -> None:
        self.name = 'ERNIE'
        self.train_path = 'THUCNews/train.txt'
        self.dev_path = 'THUCNews/dev.txt'
        self.test_path = 'THUCNews/test.txt'
        self.log_path = './log/SummaryWriter'
        self.model_save_path = f'./save/{self.name}.pkl'
        self.shuffle = True
        self.cuda_is_aviable = True
        self.cuda_device = 3
        self.learning_rate = 5e-5
        self.epoch = 3
        self.pad_size = 100
        self.batch_size = 100
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_classes = 14


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_config = BertConfig.from_pretrained(config.bert_path)
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        self.bert = BertModel.from_pretrained(
            config.bert_path, config=self.model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
       
        context = x[0]  
        mask = x[2]
        pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled[1])
        out = self.softmax(out)
        return out
