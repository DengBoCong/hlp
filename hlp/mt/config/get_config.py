import os
import json

json_path = os.path.join(os.path.dirname(__file__), 'transformer.json')  # 配置文件路径


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


conf = get_config_json(json_path)

# 对各变量赋值
path_to_file = conf['path_to_file']  # 用于训练的文本路径
path_to_eval_file = conf['path_to_eval_file']  # 用于评估计算指标的文本路径
num_eval = conf['num_eval']  # 用于计算指标的句子对数量
checkpoint_path = conf["checkpoint_path"]   # 检查点路径
BUFFER_SIZE = conf['BUFFER_SIZE']
BATCH_SIZE = conf['BATCH_SIZE']
test_size = conf['test_size']  # 训练数据中test数据占比
num_sentences = conf["num_sentences"]  # 用于训练的句子对数量
num_layers = conf["num_layers"]  # encoder 与 decoder 中包含的 encoder 与 decoder 层数
d_model = conf["d_model"]  # embedding 的维度
dff = conf["dff"]  # 点式前馈网络（Point wise feed forward network）第一层dense的维度
num_heads = conf["num_heads"]  # 多头注意力的头数
dropout_rate = conf["dropout_rate"]
EPOCHS = conf["EPOCHS"]  # 训练轮次
max_target_length = conf['max_target_length']  # 最大生成目标句子长度
