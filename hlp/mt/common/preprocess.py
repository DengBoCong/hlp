"""
对指定路径文档进行加载处理

可在配置文件中对中英文分词方法进行选择配置

"""


import sys
sys.path.append('..')
import tensorflow_datasets as tfds
import config.get_config as _config
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import json


def load_sentences(path, num_sentences):
    """加载文本"""
    with open(path, encoding='UTF-8') as file:
        lines = file.read().strip().split('\n')
    en_sentences = [l.split('\t')[0] for l in lines[:num_sentences]]
    ch_sentences = [l.split('\t')[1] for l in lines[:num_sentences]]
    return en_sentences, ch_sentences


def preprocess_sentence_en_bpe(sentence, start_word=_config.start_word
                               , end_word=_config.end_word):
    """对BPE分词方法进行预处理"""
    sentence = start_word + ' ' + sentence + ' ' + end_word
    return sentence


def preprocess_sentence_en_tokenize(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对tokenize分词方法进行预处理"""
    s = sentence.lower().strip()
    s = re.sub(r'([?.!,])', r' \1', s)  # 在?.!,前添加空格
    s = re.sub(r'[^a-zA-Z?,!.]+', " ", s)  # 将除字母及标点外的字符变为空格
    s = re.sub(r'[" "]+', " ", s)  # 合并连续的空格
    s = s.strip()
    s = start_word + ' ' + s + ' ' + end_word  # 给句子加上开始结束标志
    return s


def preprocess_sentences_en(sentences, mode='BPE', start_word=_config.start_word
                               , end_word=_config.end_word):
    if mode == 'BPE':
        sentences = [preprocess_sentence_en_bpe(s, start_word, end_word) for s in sentences]
        return sentences
    elif mode == 'TOKENIZE':
        sentences = [preprocess_sentence_en_tokenize(s, start_word, end_word) for s in sentences]
        return sentences
    else:
        return ''


def preprocess_sentence_ch_tokenize(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对tokenize分词方法进行预处理"""
    s = sentence.strip()
    s = ' '.join(s)
    s = s.strip()
    s = start_word + ' ' + s + ' ' + end_word  # 给句子加上开始结束标志
    return s


def preprocess_sentences_ch(sentences, mode='TOKENIZE', start_word=_config.start_word
                               , end_word=_config.end_word):
    if mode == 'TOKENIZE':
        sentences = [preprocess_sentence_ch_tokenize(s, start_word, end_word) for s in sentences]
        return sentences
    else:
        return False


def create_tokenizer_bpe(sentences, save_path, start_word=_config.start_word
                         , end_word=_config.end_word, target_vocab_size=_config.target_vocab_size):
    """
    根据指定语料生成字典
    使用BPE分词
    传入经预处理的句子
    保存字典
    Returns:字典，字典词汇量
    """
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        sentences, target_vocab_size=target_vocab_size, reserved_tokens=[start_word, end_word])
    tokenizer.save_to_file(save_path)
    return tokenizer, tokenizer.vocab_size


def create_tokenizer_tokenize(sentences, save_path):
    """
    根据指定语料生成字典
    使用空格分词
    传入经预处理的句子
    保存字典
    Returns:字典，字典词汇量
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='UNK')
    tokenizer.fit_on_texts(sentences)
    json_string = tokenizer.to_json()
    with open(save_path, 'w') as f:
        json.dump(json_string, f)
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size


# 根据指定模型生成及保存字典
def create_tokenizer(sentences, mode, save_path):
    if mode == 'BPE':
        return create_tokenizer_bpe(sentences, save_path=save_path, start_word=_config.start_word
                             , end_word=_config.end_word, target_vocab_size=_config.target_vocab_size)
    elif mode == 'TOKENIZE':
        return create_tokenizer_tokenize(sentences, save_path)


def get_tokenizer_bpe(path):
    """从指定路径加载保存好的字典"""
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(path)
    return tokenizer, tokenizer.vocab_size


def get_tokenizer_tokenize(path):
    """从指定路径加载保存好的字典"""
    with open(path) as f:
        json_string = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size


# 取字典
def get_tokenizer(path, mode):
    if mode == 'BPE':
        return get_tokenizer_bpe(path)
    elif mode == 'TOKENIZE':
        return get_tokenizer_tokenize(path)


# 编码句子
def get_tokenized_tensor_bpe(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典

    Returns:
    """
    sequences = [tokenizer.encode(s) for s in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


def encode_sentences_bpe(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子列表
        tokenizer: 字典

    Returns:编码好的句子， 字典
    """
    sequences = [tokenizer.encode(s) for s in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


def encode_sentences_tokenize(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典

    Returns:编码好的句子， 字典
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


def encode_sentences(sentences, tokenizer, mode):
    if mode == 'BPE':
        return encode_sentences_bpe(sentences, tokenizer)
    elif mode == 'TOKENIZE':
        return encode_sentences_tokenize(sentences, tokenizer)
    else:
        return False


def decode_sentence_bpe(sequence, tokenizer):
    return tokenizer.decode(sequence)


def decode_sentence_tokenize(sequence, tokenizer):
    sentence = [tokenizer.index_word[idx.numpy()] for idx in sequence
                if idx != [tokenizer.word_index[_config.start_word]]]
    sentence = ''.join(sentence)
    return sentence


def decode_sentence(sentences, tokenizer, mode):
    if mode == 'BPE':
        return decode_sentence_bpe(sentences, tokenizer)
    elif mode == 'TOKENIZE':
        return decode_sentence_tokenize(sentences, tokenizer)
    else:
        return False


def split_batch(input_tensor, target_tensor):
    """
    将输入输出句子进行训练集及验证集的划分,返回张量
    """
    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=_config.test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


def check_point():
    """
    检测检查点目录下是否有文件
    """
    checkpoint_dir = _config.checkpoint_path
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    if_ckpt = tf.io.gfile.listdir(checkpoint_dir)
    return if_ckpt


def main():
    """
    模块方法测试
    """
    # 加载中英文字典
    tokenizer_en, vocab_size_en = get_tokenizer(path="../data/en_tokenizer"
                                                     , mode=_config.en_tokenize_type)
    tokenizer_ch, vocab_size_ch = get_tokenizer(path='../data/ch_tokenizer.json'
                                                     , mode=_config.ch_tokenize_type)
    print(vocab_size_en)
    print(vocab_size_ch)
    en = 'Transformer is good.'
    ch = '今天天气真好啊。'
    # 预处理句子
    en = preprocess_sentences_en([en], mode='BPE')
    ch = preprocess_sentences_ch([ch], mode='TOKENIZE')
    print("预处理后的句子")
    print(en)
    print(ch)
    # 编码句子
    print("编码后的句子")
    en, _ = encode_sentences(en, tokenizer_en, mode='BPE')

    ch, _ = encode_sentences(ch, tokenizer_ch, mode='TOKENIZE')
    print(en)
    for ts in en[0]:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))
    print(ch)
    for ts in ch[0]:
        print('{} ----> {}'.format(ts, tokenizer_ch.index_word[ts]))


if __name__ == '__main__':
    main()