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
import numpy


def load_single_sentences(path, num_sentences, column):
    """加载指定列文本，列计数从1开始"""
    sentences = []
    with open(path, encoding='UTF-8') as file:
        for i in range(num_sentences):
            line = file.readline()
            sentences.append(line.split('\t')[column - 1])
    return sentences


def load_sentences(path, num_sentences):
    """加载文本"""
    en_sentences = []
    ch_sentences = []
    with open(path, encoding='UTF-8') as file:
        for i in range(num_sentences):
            line = file.readline()
            en_sentences.append(line.split('\t')[0])
            ch_sentences.append(line.split('\t')[1])
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
    elif mode == 'WORD':
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
    if mode == 'WORD':
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
    elif mode == 'WORD':
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
    elif mode == 'WORD':
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
    elif mode == 'WORD':
        return encode_sentences_tokenize(sentences, tokenizer)
    else:
        return False


def create_encoded_sentences_bpe(sentences, tokenizer, path):
    """
    将编码好的句子保存至文件，返回最大句子长度
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典
        path:文件保存路径

    Returns:最大句子长度
    """
    sequences = [tokenizer.encode(s) for s in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    numpy.savetxt(path, sequences)
    return max_sequence_length


def create_encoded_sentences_tokenize(sentences, tokenizer, path):
    """
    将编码好的句子保存至文件，返回最大句子长度
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典

    Returns:最大句子长度
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    numpy.savetxt(path, sequences)
    return max_sequence_length


def create_encoded_sentences(sentences, tokenizer, mode, path):
    """
    将编码好的句子保存至文件，返回最大句子长度
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典
        mode:编码模式
        path:文件保存路径

    Returns:最大句子长度
    """
    if mode == 'BPE':
        return create_encoded_sentences_bpe(sentences, tokenizer, path)
    elif mode == 'WORD':
        return create_encoded_sentences_tokenize(sentences, tokenizer, path)
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
    elif mode == 'WORD':
        return decode_sentence_tokenize(sentences, tokenizer)
    else:
        return False


def split_batch(path_en, path_zh):
    """
    将输入输出句子进行训练集及验证集的划分,返回张量
    path_en:英文编码句子路径
    path_zh:中文编码句子路径
    """
    input_tensor = numpy.loadtxt(path_en, dtype='int32')
    target_tensor = numpy.loadtxt(path_zh, dtype='int32')
    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=_config.test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    train_dataset.cache()
    return train_dataset, val_dataset


def generate_batch_from_file(path_en, path_zh, num_steps, batch_size):
    """
    从编码文件中分batch读入数据集

    path_en：英文编码句子路径
    path_zh：中文编码句子路径
    num_steps：整个训练集的step数，即数据集中包含多少个batch
    batch_size:batch大小

    return:input_tensor shape=(batch_size, sentence_length), dtype=tf.int32
           , target_tensor shape=(batch_size, sentence_length), dtype=tf.int32
    """
    step = 0
    while step < num_steps:
        input_tensor = numpy.loadtxt(path_en, dtype='int32', skiprows=0 + step*batch_size, max_rows=batch_size)
        target_tensor = numpy.loadtxt(path_zh, dtype='int32', skiprows=0 + step*batch_size, max_rows=batch_size)
        step += 1
        yield tf.cast(input_tensor, tf.int32), tf.cast(target_tensor, tf.int32)


def count_word(sentences):
    """输入句子列表，使用空格分隔返回单词数"""
    count = 0
    for s in sentences:
        s = re.split(r' +', s)
        count += len(s)
    return count


def train_preprocess():
    """
    模型训练所需要的文本预处理
    - 加载句子
    - 预处理句子
    - 生成及保存字典
    - 编码句子

    """
    # 加载句子
    print('正在加载、预处理数据...')
    # en = _pre.load_single_sentences(_config.path_to_train_file_en, _config.num_sentences, column=1)
    # ch = _pre.load_single_sentences(_config.path_to_train_file_zh, _config.num_sentences, column=1)
    en, ch = load_sentences(_config.path_to_train_file, _config.num_sentences)

    # 计算语料词数
    num_words = count_word(en)
    print('英文语料单词数：%d' % num_words)

    # 预处理句子
    en = preprocess_sentences_en(en, mode=_config.en_tokenize_type)
    ch = preprocess_sentences_ch(ch, mode=_config.ch_tokenize_type)
    print('已加载句子数量:%d' % _config.num_sentences)
    print('数据加载、预处理完毕！\n')

    # 生成及保存字典
    print('正在生成、保存英文字典(分词方式:%s)...' % _config.en_tokenize_type)
    tokenizer_en, vocab_size_en = create_tokenizer(sentences=en, mode=_config.en_tokenize_type
                                                   , save_path=_config.en_bpe_tokenizer_path)
    print('生成英文字典大小:%d' % vocab_size_en)
    print('英文字典生成、保存完毕！\n')
    print('正在生成、保存中文字典(分词方式:%s)...' % _config.ch_tokenize_type)
    tokenizer_ch, vocab_size_ch = create_tokenizer(sentences=ch, mode=_config.ch_tokenize_type
                                                   , save_path=_config.ch_tokenizer_path)
    print('生成中文字典大小:%d' % vocab_size_ch)
    print('中文字典生成、保存完毕！\n')

    # 编码句子
    print("正在编码句子...")
    max_sequence_length_en = create_encoded_sentences(sentences=en, tokenizer=tokenizer_en
                                                      , mode=_config.en_tokenize_type
                                                      , path=_config.path_encoded_sequences_en)
    max_sequence_length_ch = create_encoded_sentences(sentences=ch, tokenizer=tokenizer_ch
                                                      , mode=_config.ch_tokenize_type
                                                      , path=_config.path_encoded_sequences_zh)
    print('最大中文句子长度:%d' % max_sequence_length_ch)
    print('最大英文句子长度:%d' % max_sequence_length_en)
    print("句子编码完毕！\n")

    return vocab_size_en, vocab_size_ch


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
    en = ['Transformer is good.', 'I am gg', 'q a a a']
    ch = '今天天气真好啊。'
    # 预处理句子
    en = preprocess_sentences_en([en], mode='BPE')
    ch = preprocess_sentences_ch([ch], mode='WORD')
    print("预处理后的句子")
    print(en)
    print(ch)
    # 编码句子
    print("编码后的句子")
    en, _ = encode_sentences(en, tokenizer_en, mode='BPE')

    ch, _ = encode_sentences(ch, tokenizer_ch, mode='WORD')
    print(en)
    for ts in en[0]:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))
    print(ch)
    for ts in ch[0]:
        print('{} ----> {}'.format(ts, tokenizer_ch.index_word[ts]))


if __name__ == '__main__':
    main()