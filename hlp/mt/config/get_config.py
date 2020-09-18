import os
import json

path = os.path.dirname(os.path.dirname(__file__))  # mt目录
json_path = os.path.join(os.path.dirname(__file__), 'transformer.json')  # 配置文件路径
print(json_path)


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def config(config_file=json_path):
    return get_config_json(config_file=config_file)


conf = config()

NUM_EXAMPLES = conf['NUM_EXAMPLES']  # 用来训练测试的句子数
BUFFER_SIZE = conf['BUFFER_SIZE']
BATCH_SIZE = conf['BATCH_SIZE']
TEST_SIZE = conf['TEST_SIZE']  # test文本比例
EPOCHS = conf['EPOCHS']  # 训练轮次
TARGET_LENGTH = conf['TARGET_LENGTH']  # 生成目标文本最大长度
num_layers = conf['num_layers']
d_model = conf['d_model']
dff = conf['dff']
num_heads = conf['num_heads']
dropout_rate = conf['dropout_rate']
# path_to_file = os.path.join(os.path.dirname(__file__), conf['path_to_file'])
# path_to_eval_file = os.path.join(os.path.dirname(__file__), conf['path_to_eval_file'])
path_to_file = conf['path_to_file']
path_to_eval_file = conf['path_to_eval_file']
checkpoint_path = conf['checkpoint_path']
num_eval = conf['num_eval']
