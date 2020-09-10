import os
import json

seq2seq_config = os.path.dirname(__file__) + r'\seq2seq.json'
path = os.path.dirname(__file__)[:-6]


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def config(config_file=seq2seq_config):
    return get_config_json(config_file=config_file)


conf = {}

conf = config()

vocab_inp_size = conf['enc_vocab_size']
vocab_tar_size = conf['dec_vocab_size']
embedding_dim = conf['embedding_dim']
BATCH_SIZE = conf['batch_size']
units = conf['layer_size']
max_length_inp = 20
max_length_tar = 20
max_train_data_size = conf['max_train_data_size']
data = path + conf['tokenized_data']
train_data = path + conf['train_data']
resource_data = path + conf['resource_data']
tokenized_data = path + conf['tokenized_data']
