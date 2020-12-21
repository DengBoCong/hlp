import os
import json

json_path = os.path.join(os.path.dirname(__file__), 'config.json')  # 配置文件路径


def get_config_json(config_file='main.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


conf = get_config_json(json_path)

# 对各变量赋值
num_validate_sentences = conf["num_validate_sentences"]  # 用来验证的句子数量
path_to_val_file = conf["path_to_val_file"]  # 验证集文本路径
validation_data = conf["validation_data"]  # 是否从给定文本数据集用来验证
checkpoint_name = conf['checkpoint_name']  # 检查点名字
validation_freq = conf['validation_freq']  # 验证频率，即每训练几个epoch进行验证
checkpoints_save_freq = conf['checkpoints_save_freq']  # 检查点保存频率
max_checkpoints_num = conf['max_checkpoints_num']  # 保存最大检查点数量
source_lang = conf['source_lang']  # 源语言
target_lang = conf['target_lang']  # 目标语言
reverse = conf['reverse']  # 是否对语料语言对翻转
en_tokenize_type = conf['en_tokenize_type']  # 英文分词类型，可选：BPE/TOKENIZE
zh_tokenize_type = conf['zh_tokenize_type']  # 中文分词类型，可选：TOKENIZE
tokenizer_path_prefix = conf["tokenizer_path_prefix"]  # 字典保存路径前缀
encoded_sequences_path_prefix = conf['encoded_sequences_path_prefix']  # 编码句子保存路径前缀
result_save_dir = conf['result_save_dir']  # 训练过程指标变化图保存路径
path_to_train_file = conf['path_to_train_file']  # 用于训练的文本路径
path_to_eval_file = conf['path_to_eval_file']  # 用于评估计算指标的文本路径
num_eval = conf['num_eval']  # 用于计算指标的句子对数量
checkpoint_path = os.path.join(conf["checkpoint_path_dir"], conf['source_lang']+'_'+conf['target_lang'])   # 检查点路径
BUFFER_SIZE = conf['BUFFER_SIZE']
BATCH_SIZE = conf['BATCH_SIZE']
train_size = conf['train_size']  # 训练数据中test数据占比
num_sentences = conf["num_sentences"]  # 用于训练的句子对数量
num_layers = conf["num_layers"]  # encoder 与 decoder 中包含的 encoder 与 decoder 层数
d_model = conf["d_model"]  # embedding 的维度
dff = conf["dff"]  # 点式前馈网络（Point wise feed forward network）第一层dense的维度
num_heads = conf["num_heads"]  # 多头注意力的头数
dropout_rate = conf["dropout_rate"]
EPOCHS = conf["EPOCHS"]  # 训练轮次
max_target_length = conf['max_target_length']  # 最大生成目标句子长度
target_vocab_size = conf["target_vocab_size"]  # 英语分词target_vocab_size
start_word = conf["start_word"]  # 句子开始标志
end_word = conf["end_word"]  # 句子结束标志
BEAM_SIZE = conf["BEAM_SIZE"]  # BEAM_SIZE
checkpoint_ensembling = conf["checkpoint_ensembling"]  # 是否采用checkpoint_ensembling

lm_path_to_train_file = conf["language_model"]["path_to_train_file_lm"]  # 语言模型训练文本路径
lm_language = conf["language_model"]["language"]
lm_tokenize_type = conf["language_model"]["tokenize_type"]
lm_EPOCHS = conf["language_model"]["EPOCHS"]
lm_num_sentences = conf["language_model"]["num_sentences"]
lm_BATCH_SIZE = conf["language_model"]["BATCH_SIZE"]
lm_train_size = conf["language_model"]["train_size"]
lm_checkpoint_path = os.path.join(conf["checkpoint_path_dir"], 'lm')
lm_d_embedding = conf["language_model"]["d_embedding"]
lm_d_rnn = conf["language_model"]["d_rnn"]
lm_max_checkpoints_num = conf["language_model"]["max_checkpoints_num"]
lm_checkpoints_save_freq = conf["language_model"]["checkpoints_save_freq"]
lm_validation_freq = conf["language_model"]["validation_freq"]

