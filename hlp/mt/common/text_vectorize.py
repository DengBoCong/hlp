import os

import tensorflow as tf
import numpy
import json
import tensorflow_datasets as tfds

from hlp.mt.config import get_config as _config


def _create_and_save_tokenizer_bpe(sentences, save_path, start_word=_config.start_word,
                                   end_word=_config.end_word, target_vocab_size=_config.target_vocab_size):
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


def _create_and_save_tokenizer_keras(sentences, save_path):
    """
    根据指定语料生成字典
    使用tf.keras.preprocessing.text.Tokenizer进行分词，即使用空格分词
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


def create_and_save_tokenizer(sentences, save_path, language, mode):
    """根据所给参数生成及保存字典
    @param sentences:生成字典所使用的句子列表
    @param save_path: 保存的路径
    @param language: 语言
    @param mode: 编码方法
    """
    # 若目录不存在，则创建目录
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if language == 'zh':
        return _create_and_save_tokenizer_keras(sentences, save_path)
    elif language == 'en':
        if mode == 'BPE':
            return _create_and_save_tokenizer_bpe(sentences, save_path)
        elif mode == 'WORD':
            return _create_and_save_tokenizer_keras(sentences, save_path)
        else:
            raise ValueError("语言(%s)暂不支持模式(%s)" % (language, mode))
    else:
        raise ValueError("暂不支持语言(%s)" % language)


def _load_tokenizer_bpe(path):
    """从指定路径加载保存好的字典"""
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(path)
    return tokenizer, tokenizer.vocab_size


def _load_tokenizer_keras(path):
    """从指定路径加载保存好的字典"""
    with open(path) as f:
        json_string = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size


# 取字典
def load_tokenizer(path, language, mode):
    """加载字典

    @param path:字典保存路径
    @param language: 语言
    @param mode: 模式
    @return: 字典及字典大小
    """
    if not os.path.exists(path):
        raise ValueError("路径(%s)不存在" % path)
    if language == 'zh':
        return _load_tokenizer_keras(path)
    elif language == 'en':
        if mode == 'BPE':
            return _load_tokenizer_bpe(path)
        elif mode == 'WORD':
            return _load_tokenizer_keras(path)
        else:
            raise ValueError("语言(%s)暂不支持模式(%s)" % (language, mode))


def _encode_sentences_bpe(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子列表
        tokenizer: 字典

    Returns:编码好的句子
    """
    sequences = [tokenizer.encode(s) for s in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


def _encode_sentences_keras(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子列表
        tokenizer: 字典

    Returns:编码好的句子
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


def encode_sentences(sentences, tokenizer, language, mode):
    """对句子进行编码
    @param sentences: 要编码的句子列表
    @param tokenizer: 使用的字典
    @param language: 语言
    @param mode: 模式
    @return: 编码好的句子列表，最大句子长度
    """
    if language == 'zh':
        return _encode_sentences_keras(sentences, tokenizer)
    elif language == 'en':
        if mode == 'BPE':
            return _encode_sentences_bpe(sentences, tokenizer)
        elif mode == 'WORD':
            return _encode_sentences_keras(sentences, tokenizer)
        else:
            raise ValueError("语言(%s)暂不支持模式(%s)" % (language, mode))


def get_start_token(start_word, tokenizer, language):
    """
    由于BPE分词的特殊性，在BPE获取start_token时需要在start_word后加一个空格
    故在此使用方法对mode(编码模式)进行判断
    返回 start_token  shape --> (1,) eg：[3]
    """
    if language == "en":
        mode = _config.en_tokenize_type
        if mode == 'BPE':
            start_word = start_word + ' '
            start_token, _ = _encode_sentences_bpe([start_word], tokenizer)
        elif mode == 'WORD':
            start_token, _ = _encode_sentences_keras([start_word], tokenizer)
    elif language == "zh":
        mode = _config.zh_tokenize_type
        if mode == 'CHAR':
            start_token, _ = _encode_sentences_keras([start_word], tokenizer)
        elif mode == 'WORD':
            start_token, _ = _encode_sentences_keras([start_word], tokenizer)

    start_token = [tf.squeeze(start_token)]
    return start_token


def _encode_and_save_bpe(sentences, tokenizer, path):
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


def _encode_and_save_keras(sentences, tokenizer, path):
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


def encode_and_save(save_path, sentences, tokenizer, language, mode):
    """编码并保存句子
    @param save_path:保存的路径
    @param sentences:需要进行编码并保存的句子
    @param tokenizer:使用的字典
    @param language:语言
    @param mode:模式
    @return:最大句子长度
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if language == 'zh':
        return _encode_and_save_keras(sentences, tokenizer, save_path)
    elif language == 'en':
        if mode == 'BPE':
            return _encode_and_save_bpe(sentences, tokenizer, save_path)
        elif mode == 'WORD':
            return _encode_and_save_keras(sentences, tokenizer, save_path)
    else:
        raise ValueError("语言(%s)暂不支持模式(%s)" % (language, mode))


def _decode_sentence_bpe(sequence, tokenizer):
    return tokenizer.decode(sequence)


def _decode_sentence_tokenizer(sequence, tokenizer, join_str=''):
    sentence = [tokenizer.index_word[idx.numpy()] for idx in sequence
                if idx != [tokenizer.word_index[_config.start_word]]]
    sentence = join_str.join(sentence)
    return sentence


def decode_sentence(sequence, tokenizer, language, mode):
    """对句子进行解码

    @param sequence: 需要解码的句子
    @param tokenizer: 使用的字典
    @param language: 语言
    @param mode: 模式
    """
    if language == 'zh':
        return _decode_sentence_tokenizer(sequence, tokenizer)
    elif language == 'en':
        if mode == 'BPE':
            return _decode_sentence_bpe(sequence, tokenizer)
        elif mode == 'WORD':
            return _decode_sentence_tokenizer(sequence, tokenizer, join_str=' ')
    else:
        raise ValueError("语言(%s)暂不支持模式(%s)" % (language, mode))
