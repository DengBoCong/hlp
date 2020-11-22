"""
对指定路径文档进行加载处理

可在配置文件中对中英文分词方法进行选择配置

"""
import sys

sys.path.append('..')
from sklearn.model_selection import train_test_split
from model import trainer
import config.get_config as _config
from model import nmt_model
from common import checkpoint
from pathlib import Path
from common import tokenize
import tensorflow as tf
import numpy
import re
import os
import jieba


def load_single_sentences(path, num_sentences, column):
    """加载指定列文本，列计数从1开始"""
    sentences = []
    with open(path, encoding='UTF-8') as file:
        for i in range(num_sentences):
            line = file.readline()
            sentences.append(line.split('\t')[column - 1])
    return sentences


def load_sentences(path, num_sentences, reverse=_config.reverse):
    """加载文本"""
    source_sentences = []
    target_sentences = []
    with open(path, encoding='UTF-8') as file:
        for i in range(num_sentences):
            line = file.readline()
            source_sentences.append(line.split('\t')[0])
            target_sentences.append(line.split('\t')[1])
    if reverse == 'True':
        return target_sentences, source_sentences
    else:
        return source_sentences, target_sentences


def _preprocess_sentence_en_bpe(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对BPE分词方法进行预处理"""
    sentence = start_word + ' ' + sentence + ' ' + end_word
    return sentence


def _preprocess_sentence_en_word(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对输入句子进行预处理"""
    s = sentence.lower().strip()
    s = re.sub(r'([?.!,])', r' \1', s)  # 在?.!,前添加空格
    s = re.sub(r'[^a-zA-Z?,!.]+', " ", s)  # 将除字母及标点外的字符变为空格
    s = re.sub(r'[" "]+', " ", s)  # 合并连续的空格
    s = s.strip()
    s = start_word + ' ' + s + ' ' + end_word  # 给句子加上开始结束标志
    return s


def _preprocess_sentences_en(sentences, mode=_config.en_tokenize_type, start_word=_config.start_word,
                             end_word=_config.end_word):
    """
    对英文句子列表进行指定mode的预处理
    返回处理好的句子列表
    """
    if mode == 'BPE':
        sentences = [_preprocess_sentence_en_bpe(s, start_word, end_word) for s in sentences]
        return sentences
    elif mode == 'WORD':
        sentences = [_preprocess_sentence_en_word(s, start_word, end_word) for s in sentences]
        return sentences
    else:
        return ''


def _preprocess_sentence_zh_char(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对输入句子(string)进行预处理"""
    s = sentence.strip()
    s = ' '.join(s)
    s = s.strip()
    s = start_word + ' ' + s + ' ' + end_word  # 给句子加上开始结束标志
    return s


def _preprocess_sentence_zh_word(sentences, start_word=_config.start_word, end_word=_config.end_word):
    """使用jieba进行分词前的预处理"""
    sentences_word = []
    for sentence in sentences:
        sentence = start_word + ' ' + ' '.join(jieba.cut(sentence.strip())) + ' ' + end_word
        sentences_word.append(sentence)
    return sentences_word


def _preprocess_sentences_zh(sentences, mode=_config.zh_tokenize_type, start_word=_config.start_word,
                             end_word=_config.end_word):
    """
    对中文句子列表进行指定mode的预处理
    返回处理好的句子列表
    """
    if mode == 'CHAR':
        sentences = [_preprocess_sentence_zh_char(s, start_word, end_word) for s in sentences]
        return sentences
    elif mode == 'WORD':
        return _preprocess_sentence_zh_word(sentences)


def preprocess_sentences(sentences, language):
    """通过language判断mode"""
    if language == "en":
        mode = _config.en_tokenize_type
        return _preprocess_sentences_en(sentences, mode)
    elif language == "zh":
        mode = _config.zh_tokenize_type
        return _preprocess_sentences_zh(sentences, mode)


def _split_batch(input_path, target_path, train_size=_config.train_size):
    """
    根据配置文件语言对来确定文件路径，划分训练集与验证集
    """

    input_tensor = numpy.loadtxt(input_path, dtype='int32')
    target_tensor = numpy.loadtxt(target_path, dtype='int32')
    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, train_size=train_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


def _generate_batch_from_file(input_path, target_path, num_steps, start_step, batch_size):
    """
    从编码文件中分batch读入数据集
    自动从配置文件设置确定input_path、target_path
    num_steps：整个训练集的step数，即数据集中包含多少个batch
    start_step:从哪个step开始读batch
    batch_size:batch大小

    return:input_tensor shape=(batch_size, sentence_length), dtype=tf.int32
           , target_tensor shape=(batch_size, sentence_length), dtype=tf.int32
    """

    step = int(start_step)
    while step < num_steps:
        input_tensor = numpy.loadtxt(input_path, dtype='int32', skiprows=0 + step * batch_size, max_rows=batch_size)
        target_tensor = numpy.loadtxt(target_path, dtype='int32', skiprows=0 + step * batch_size, max_rows=batch_size)
        step += 1
        yield tf.cast(input_tensor, tf.int32), tf.cast(target_tensor, tf.int32)


def get_dataset(steps, cache, train_size, validate_from_txt):
    """

    @param steps:训练集文本共含多少个batch
    @param cache:是否一次性加载入内存
    @param train_size:训练集比例
    @param validate_from_txt:是否从指定文本加载验证集

    返回训练可接收的训练集验证集
    """
    input_path = _config.encoded_sequences_path_prefix + _config.source_lang
    target_path = _config.encoded_sequences_path_prefix + _config.target_lang
    # 首先判断是否从指定文件读入，若为真，则从验证集文本读取验证集数据
    if validate_from_txt == 'True':
        train_size = 0.9999
        val_dataset, _ = _split_batch(input_path + '_val', target_path + '_val', train_size)
        # 判断训练数据是直接读取还是采用生成器读取
        if cache:
            train_dataset, _ = _split_batch(input_path, target_path, train_size)
        else:
            train_dataset = _generate_batch_from_file(input_path, target_path, steps, 0, _config.BATCH_SIZE)
        return train_dataset, val_dataset
    # 若为假，则从数据集中划分验证集
    else:
        if cache:
            train_dataset, val_dataset = _split_batch(input_path, target_path, train_size)
        else:
            train_dataset = _generate_batch_from_file(input_path, target_path
                                                      , steps * train_size, 0, _config.BATCH_SIZE)
            val_dataset = _generate_batch_from_file(input_path, target_path
                                                    , steps, steps * train_size, _config.BATCH_SIZE)
        return train_dataset, val_dataset


def _count_words(sentences):
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
    source_sentences, target_sentences = load_sentences(_config.path_to_train_file, _config.num_sentences)
    # 加载验证集
    if _config.validation_data == "True":
        source_sentences_val, target_sentences_val = load_sentences(_config.path_to_val_file
                                                                    , _config.num_validate_sentences)

    # 计算语料词数
    num_words = _count_words(source_sentences)
    print('源语料(%s)词数：%d' % (_config.source_lang, num_words))

    # 预处理句子
    source_sentences = preprocess_sentences(source_sentences, language=_config.source_lang)
    target_sentences = preprocess_sentences(target_sentences, language=_config.target_lang)
    if _config.validation_data == "True":
        source_sentences_val = preprocess_sentences(source_sentences_val, language=_config.source_lang)
        target_sentences_val = preprocess_sentences(target_sentences_val, language=_config.target_lang)
    print('已加载句子数量:%d' % _config.num_sentences)
    print('数据加载、预处理完毕！\n')

    # 生成及保存字典
    print('正在生成、保存源语言(%s)字典(分词方式:%s)...' % (_config.source_lang, _config.en_tokenize_type))
    tokenizer_source, vocab_size_source = tokenize.create_tokenizer(sentences=source_sentences
                                                                    , language=_config.source_lang)
    print('生成英文字典大小:%d' % vocab_size_source)
    print('源语言字典生成、保存完毕！\n')

    print('正在生成、保存目标语言(%s)字典(分词方式:%s)...' % (_config.target_lang, _config.zh_tokenize_type))
    tokenizer_target, vocab_size_target = tokenize.create_tokenizer(sentences=target_sentences
                                                                    , language=_config.target_lang)
    print('生成目标语言字典大小:%d' % vocab_size_target)
    print('目标语言字典生成、保存完毕！\n')

    # 编码句子
    print("正在编码训练集句子...")
    max_sequence_length_source = tokenize.create_encoded_sentences(sentences=source_sentences
                                                                   , tokenizer=tokenizer_source
                                                                   , language=_config.source_lang)
    max_sequence_length_target = tokenize.create_encoded_sentences(sentences=target_sentences
                                                                   , tokenizer=tokenizer_target
                                                                   , language=_config.target_lang)
    print('最大源语言(%s)句子长度:%d' % (_config.source_lang, max_sequence_length_source))
    print('最大目标语言(%s)句子长度:%d' % (_config.target_lang, max_sequence_length_target))
    if _config.validation_data == "True":
        print("正在编码验证集句子...")
        _ = tokenize.create_encoded_sentences(sentences=source_sentences_val
                                              , tokenizer=tokenizer_source
                                              , language=_config.source_lang
                                              , postfix='_val')
        _ = tokenize.create_encoded_sentences(sentences=target_sentences_val
                                              , tokenizer=tokenizer_target
                                              , language=_config.target_lang
                                              , postfix='_val')
    print("句子编码完毕！\n")

    return vocab_size_source, vocab_size_target


def load_model():
    """
    进行翻译或评估前数据恢复工作
    """
    # 加载源语言字典
    print("正在加载源语言(%s)字典..." % _config.source_lang)
    tokenizer_source, vocab_size_source = tokenize.get_tokenizer(language=_config.source_lang)
    print('源语言字典大小:%d' % vocab_size_source)
    print('源语言字典加载完毕！\n')

    # 加载目标语言字典
    print("正在加载目标语言(%s)字典..." % _config.target_lang)
    tokenizer_target, vocab_size_target = tokenize.get_tokenizer(language=_config.target_lang)
    print('目标语言字典大小:%d' % vocab_size_target)
    print('目标语言字典加载完毕！\n')

    # 创建模型及相关变量
    learning_rate = trainer.CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = nmt_model.get_model(vocab_size_source, vocab_size_target)

    # 加载检查点
    checkpoint.load_checkpoint(transformer, optimizer)

    return transformer, tokenizer_source, tokenizer_target


def check_point():
    """
    检测检查点目录下是否有文件
    """
    # 进行语言对判断从而确定检查点路径
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
    tokenizer_en, vocab_size_en = tokenize.get_tokenizer(path="../data/en_tokenizer"
                                                         , mode=_config.en_tokenize_type)
    tokenizer_ch, vocab_size_ch = tokenize.get_tokenizer(path='../data/tokenizer/ch_tokenizer.json'
                                                         , mode=_config.ch_tokenize_type)
    print(vocab_size_en)
    print(vocab_size_ch)
    en = 'Transformer is good.'
    ch = '今天天气真好啊。'

    # 预处理句子
    en = _preprocess_sentences_en([en], mode='BPE')
    ch = _preprocess_sentences_zh([ch], mode='WORD')
    print("预处理后的句子")
    print(en)
    print(ch)

    # 编码句子
    print("编码后的句子")
    en, _ = tokenize.encode_sentences(en, tokenizer_en, mode='BPE')

    ch, _ = tokenize.encode_sentences(ch, tokenizer_ch, mode='WORD')
    print(en)
    for ts in en[0]:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))
    print(ch)
    for ts in ch[0]:
        print('{} ----> {}'.format(ts, tokenizer_ch.index_word[ts]))


if __name__ == '__main__':
    main()
