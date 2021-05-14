from torch.nn.modules import module
from train_iter import train, test
from util import load_data, data_loader, FastText_dataloader
from LoggerClass import Logging
import torch
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='Choose the model to train: TextCNN, TextRNN, TextRCNN, Transformer')
parser.add_argument('--word', default=False, type=bool,
                    help='Choose mode')
args = parser.parse_args()


if __name__ == '__main__':

    model = args.model
    word = args.word
 
    load = import_module(f'model.{model}')

    Logging.__init__(model, save_path='./log/', logerlevel='DEBUG')
    Logging.add_log(f'Read Parameter: model: {model}, word: {word}', 'debug')

    config = load.Config()
    text_model = load.Model(config)

    # 加载数据
    if model == 'FastText':
        train_data, train_label = FastText_dataloader('train', word, config)
        test_data, test_label = FastText_dataloader('test', word, config)
        eva_data, eva_label = FastText_dataloader('dev', word, config)
    else:
        _, train_data, train_label = load_data('train', word, config)
        _, test_data, test_label = load_data('test', word, config)
        _, eva_data, eva_label = load_data('dev', word, config)
    Logging.add_log(f'Completed loading data', 'debug')

    dataloader = data_loader(data=train_data, label=train_label, config=config)
    test_dataloader = data_loader(
        data=test_data, label=test_label, config=config)
    eva_dataloader = data_loader(data=eva_data, label=eva_label, config=config)

    Logging.add_log(f'Start Training', 'debug')
    train(config=config, model=text_model, data=dataloader, name=model,
          test_dataloader=test_dataloader, eva_dataloader=eva_dataloader, logger=Logging)


