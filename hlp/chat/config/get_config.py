import os
import json

seq2seq_config = os.path.dirname(__file__) + r'\model_config.json'
path = os.path.dirname(__file__)[:-6]


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def config(config_file=seq2seq_config):
    return get_config_json(config_file=config_file)


conf = {}

conf = config()

# 公共配置
# vocab_inp_size = conf['enc_vocab_size']
# vocab_tar_size = conf['dec_vocab_size']
embedding_dim = conf['embedding_dim']
BATCH_SIZE = conf['batch_size']
BUFFER_SIZE = conf['buffer_size']
units = conf['layer_size']
vocab_size = conf['vocab_size']
max_length_inp = 40  # 最大文本长度
max_length_tar = 40  # 最大文本长度
max_train_data_size = conf['max_train_data_size']
data = path + conf['tokenized_data']  # 训练数据位置
resource_data = path + conf['resource_data']  # 原始数据位置
tokenized_data = path + conf['tokenized_data']  # 预处理之后数据位置
beam_size = conf['beam_size']  # beam_search大小
epochs = conf['epochs']  # 训练轮次

# seq2seq模型相关配置
seq2seq_train_data = path + conf['seq2seq']['train_data']  # 训练结果保存位置

# transformer模型相关配置
transformer_train_data = path + conf['transformer']['train_data']  # 训练结果保存位置
transformer_num_layers = conf['transformer']['num_layers']
transformer_d_model = conf['transformer']['d_model']
transformer_num_heads = conf['transformer']['num_heads']
transformer_units = conf['transformer']['units']
transformer_dropout = conf['transformer']['dropout']

# task模型相关配置
state_tracker_hidden_size = conf['task']['state_tracker_hidden_size']
task_train_data = path + conf['task']['train_data']
sent_groups = path + conf['task']['sent_groups']  # 含插槽的句子组合
database = path + conf['task']['database']
ontology = path + conf['task']['ontology']
semi_dict = path + conf['task']['semi_dict']
dialogues_train = path + conf['task']['dialogues_train']
dialogues_tokenized = path + conf['task']['tokenized_data']
