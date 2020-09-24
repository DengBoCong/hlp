"""
对指定路径文档进行加载处理
主要方法：
- 处理中英句子
  - 英文文本：将单词将小写并用分开
  - 中文文本: 将汉字用空格分开
- 生成中英字典
- 使用字典将句子编码
- 检查检查点
- 加载检查点
"""

import config.get_config as _config
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


def preprocess_en_sentence(s):
    '''对英语句子进行处理'''
    s = s.lower().strip()
    s = re.sub(r'([?.!,])',r' \1',s)  # 在?.!,前添加空格
    s = re.sub(r'[^a-zA-Z?,!.]+'," ",s)  # 将除字母及标点外的字符变为空格
    s = re.sub(r'[" "]+'," ",s)  # 合并连续的空格
    s = s.strip()

    s = '<start> ' + s + ' <end>'  # 给句子加上开始结束标志
    return s


# temp_en = 'This is an example.'
# temp_en = preprocess_en_sentence(temp_en)
# print(temp_en)
# print('-'*20)


def preprocess_ch_sentence(s):
    s = s.strip()
    s = ' '.join(s)
    s = s.strip()
    s = '<start> ' + s + ' <end>'  # 给句子加上开始结束标志
    return s


# temp_ch = '这是一个例子。'
# temp_ch = preprocess_ch_sentence(temp_ch)
# print(temp_ch)
# print('-'*20)


def max_length(texts):
    '''返回给定句子列表中最大句子长度'''
    return max(len(t) for t in texts)


def create_dataset(path, num_sentences, en=True, ch=True):
    """
    Args:
        path:文本路径
        num_sentences:取出的句子对数量
        en:是否对英语句子进行预处理
        ch:是否对中文句子进行预处理

    Returns:英中文句子列表

    注意，文本应为 en  ch句子对
    将路径的文本
    """
    with open(path,encoding='UTF-8') as file:
        lines = file.read().strip().split('\n')

    en_sentences = [l.split('\t')[0] for l in lines[:num_sentences]]
    ch_sentences = [l.split('\t')[1] for l in lines[:num_sentences]]

    if en:
        en_sentences = [preprocess_en_sentence(s) for s in en_sentences]
    if ch:
        ch_sentences = [preprocess_ch_sentence(s) for s in ch_sentences]

    return en_sentences,ch_sentences


def tokenize(texts):
    """
    Args:
        texts:用于训练字典和编码的文本列表

    Returns:编码后的sequences和字典

    注意，编码是根据空格来分割的，所以需要先将句子进行预处理
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

    return sequences, tokenizer


# temp_texts = ['你 好 啊！','我爱你 ~']
# temp_sequences_ch,temp_tokenizer_ch = tokenize(temp_texts)
# print('temp_sequences_ch:')
# print(temp_sequences_ch)
# print('temp_tokenizer_ch:')
# print(len(temp_tokenizer_ch.word_index))
# print(temp_tokenizer_ch.word_index)
# temp_texts = ['I love you.','you are so beautiful !','i hate you .']
# temp_sequences_en,temp_tokenizer_en = tokenize(temp_texts)
# print('temp_sequences_en:')
# print(temp_sequences_en)
# print('temp_tokenizer_en:')
# print(temp_tokenizer_en.word_index)
# print('-'*20)


def get_tokenized_dataset(path=_config.path_to_file, num_sequences=_config.num_sentences):
    """
    Args:
        path:需要编码的文本路径
        num_sequences:

    Returns:input_sequences, target_sequences, input_tokenizer, target_tokenizer

    返回的为编码好的句子及字典
    """
    en_sentences, ch_sentences = create_dataset(path, num_sequences)
    input_sequences, input_tokenizer = tokenize(en_sentences)
    target_sequences, target_tokenizer = tokenize(ch_sentences)
    return input_sequences, target_sequences, input_tokenizer, target_tokenizer


# temp_input_sequences, temp_target_sequences, temp_input_tokenizer, temp_target_tokenizer = \
#     get_tokenized_dataset(_config.path_to_file,1000)
# print(temp_input_sequences)
# print(temp_target_sequences)
# print(temp_input_tokenizer.word_index)
# print(temp_target_tokenizer.word_index)
# print('-'*20)


def split_batch(input_tensor, target_tensor):
    '''将输入输出句子进行训练集及验证集的划分,返回张量'''
    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=_config.test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_dataset = train_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


# temp_train_dataset, temp_val_dataset = split_batch(temp_input_sequences, temp_target_sequences)
# example_input_batch, example_target_batch = next(iter(temp_train_dataset))
# print(example_input_batch.shape, example_target_batch.shape)
# print('-'*20)

def check_point():
    '''检测检查点目录下是否有文件'''
    checkpoint_dir = _config.checkpoint_path
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    if_ckpt = tf.io.gfile.listdir(checkpoint_dir)
    return if_ckpt

