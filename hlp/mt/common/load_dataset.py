import numpy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from hlp.mt.config import get_config as _config


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
