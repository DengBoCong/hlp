import tensorflow as tf
import numpy
from sklearn.model_selection import train_test_split

from hlp.mt.common import load_dataset, text_vectorize, text_split
from hlp.mt.config import get_config as _config


def get_tokenizer_path(language, mode):
    """合成字典保存路径

    @param language:语言
    @param mode:编码类型
    @return:字典保存路径
    """
    return _config.tokenizer_path_prefix + language + '_' + mode.lower()


def get_encoded_sequences_path(language, postfix=''):
    """根据语言获取已编码句子的保存路径

    @param language: 语言
    @param postfix: 保存路径的后缀
    @return:已编码句子的保存路径
    """
    return _config.encoded_sequences_path_prefix + language + postfix


def train_preprocess():
    language = _config.lm_language
    mode = _config.lm_tokenize_type
    tokenizer_path = get_tokenizer_path(language, mode)
    encoded_sequences_path_train = get_encoded_sequences_path(language, postfix='_train')
    # encoded_sequences_path_val = get_encoded_sequences_path(language, postfix='_val')

    # 文本加载及预处理
    print('正在加载、预处理数据...')
    sentences = load_dataset.load_single_sentences(_config.lm_path_to_train_file, _config.lm_num_sentences, column=2)
    sentences = text_split.preprocess_sentences(sentences, language, mode)
    print('已加载句子数量:%d' % _config.lm_num_sentences)
    print('数据加载、预处理完毕！\n')

    # 使用预处理的文本生成及保存字典
    tokenizer, vocab_size = text_vectorize.create_and_save_tokenizer(sentences, tokenizer_path, language, mode)
    print('生成字典大小:%d' % vocab_size)
    print('字典生成、保存完毕！\n')

    # 使用字典对文本进行编码并保存
    print("正在编码训练集句子...")
    max_sequence_length = text_vectorize.encode_and_save(sentences, tokenizer, encoded_sequences_path_train, language,
                                                         mode)
    print('最大句子长度:%d' % max_sequence_length)
    print("句子编码完毕！\n")

    return tokenizer, vocab_size, max_sequence_length


def get_dataset(sequences_path, train_size=_config.lm_train_size):
    """加载并划分数据集
    """
    tensor = numpy.loadtxt(sequences_path, dtype='int32')

    train_dataset, val_dataset = train_test_split(tensor, train_size=train_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(_config.lm_BATCH_SIZE).batch(_config.lm_BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)
    val_dataset = val_dataset.shuffle(_config.lm_BATCH_SIZE).batch(_config.lm_BATCH_SIZE, drop_remainder=True)

    return train_dataset, val_dataset

