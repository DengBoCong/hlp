"""
预处理中关于字典及编码解码部分
加入新语言时需要在下列方法中对新语言进行判断
主要方法包括：
- create_tokenizer(sentence, language): 生成及保存字典
- get_tokenizer(language): 获取字典
- encode_sentences(sentence, tokenizer, language): 编码句子
- get_start_token(start_word, tokenizer, language): 获取start_token
- create_encoded_sentences(sentence, tokenizer, language): 编码并保存句子列表
- decode_sentence(sentence, tokenizer, language): 解码句子
"""
import os

import tensorflow as tf
import numpy
import json
import tensorflow_datasets as tfds

from hlp.mt.config import get_config as _config


def _create_and_save_tokenizer_bpe(sentences, save_path, start_word=_config.start_word
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


def _get_mode_and_path_tokenize(language, model_type):
    """根据语言及模型确定字典的编码模式及保存路径"""
    if language == "en":
        if model_type == "nmt":
            mode = _config.en_tokenize_type
            path = _config.tokenizer_path_prefix + language + '_' + mode.lower()
        elif model_type == "lm":
            mode = _config.lm_en_tokenize_type
            path = _config.tokenizer_path_prefix + language + '_' + mode.lower()+"_lm"
    elif language == "zh":
        if model_type == "nmt":
            mode = _config.zh_tokenize_type
            path = _config.tokenizer_path_prefix + language + '_' + mode.lower()
        elif model_type == "lm":
            mode = _config.lm_zh_tokenize_type
            path = _config.tokenizer_path_prefix + language + '_' + mode.lower() + "_lm"
    return mode, path


# 根据指定模型生成及保存字典
def create_and_save_tokenizer(sentences, language, model_type="nmt"):
    """
    生成和保存指定语言的字典
    所使用的模式为配置文件中设置的该语言的模式
    支持的语言：en,zh
    @param sentences:
    @param language:
    @param model_type:
    @return:
    """
    # 生成英文字典
    mode, save_path = _get_mode_and_path_tokenize(language, model_type)
    if language == "en":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if mode == 'BPE':
            return _create_and_save_tokenizer_bpe(sentences, save_path=save_path)
        elif mode == 'WORD':
            return _create_and_save_tokenizer_keras(sentences, save_path=save_path)
    # 生成中文字典
    elif language == "zh":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if mode == 'CHAR':
            return _create_and_save_tokenizer_keras(sentences, save_path=save_path)
        elif mode == 'WORD':
            return _create_and_save_tokenizer_keras(sentences, save_path=save_path)


def _get_tokenizer_bpe(path):
    """从指定路径加载保存好的字典"""
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(path)
    return tokenizer, tokenizer.vocab_size


def _get_tokenizer_keras(path):
    """从指定路径加载保存好的字典"""
    with open(path) as f:
        json_string = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size


# 取字典
def load_tokenizer(language, model_type="nmt"):
    """
    根据语言获取保存的字典
    支持的语言：en、zh
    """
    # 根据所选语言确定mode、save_path
    mode, path = _get_mode_and_path_tokenize(language, model_type)
    if language == "en":
        if mode == 'BPE':
            return _get_tokenizer_bpe(path)
        elif mode == 'WORD':
            return _get_tokenizer_keras(path)
    elif language == "zh":
        if mode == 'CHAR':
            return _get_tokenizer_keras(path)
        elif mode == 'WORD':
            return _get_tokenizer_keras(path)


# 编码句子
def _get_tokenized_tensor_bpe(sentences, tokenizer):
    """
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典
    Returns:已编码填充的句子及句子长度
    """
    sequences = [tokenizer.encode(s) for s in sentences]
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    max_sequence_length = len(sequences[0])
    return sequences, max_sequence_length


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


def encode_sentences(sentences, tokenizer, language, model_type="nmt"):
    """
    sentence:要编码的句子列表
    tokenizer:字典
    language:句子的语言
    """
    mode, _ = _get_mode_and_path_tokenize(language, model_type)
    if language == "en":
        if mode == 'BPE':
            return _encode_sentences_bpe(sentences, tokenizer)
        elif mode == 'WORD':
            return _encode_sentences_keras(sentences, tokenizer)
    elif language == "zh":
        if mode == 'CHAR':
            return _encode_sentences_keras(sentences, tokenizer)
        elif mode == 'WORD':
            return _encode_sentences_keras(sentences, tokenizer)


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


def get_tokenizer_mode_and_path(language, model_type, postfix):
    """根据语言及模型确定编码句子的编码模式及保存路径"""
    if language == "en":
        if model_type == "nmt":
            mode = _config.en_tokenize_type
        elif model_type == "lm":
            mode = _config.lm_en_tokenize_type
    elif language == "zh":
        if model_type == "nmt":
            mode = _config.zh_tokenize_type
        elif model_type == "lm":
            mode = _config.lm_zh_tokenize_type
    path = _config.encoded_sequences_path_prefix + language + postfix

    return mode, path


def encode_and_save(sentences, tokenizer, language, postfix='', model_type='nmt'):
    """
    根据所选语言将编码好的句子保存至文件，返回最大句子长度
    Args:
        sentences: 需要编码的句子
        tokenizer: 字典
        language: 语言类型 （en/zh）
        postfix: 保存文本名字加标注后缀
        model_type: 模型

    Returns:最大句子长度
    """
    # 根据所选语言确定mode、save_path
    mode, save_path = get_tokenizer_mode_and_path(language, model_type, postfix)
    if language == "en":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if mode == 'BPE':
            return _encode_and_save_bpe(sentences, tokenizer, save_path)
        elif mode == 'WORD':
            return _encode_and_save_keras(sentences, tokenizer, save_path)
    elif language == "zh":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if mode == 'CHAR':
            return _encode_and_save_keras(sentences, tokenizer, save_path)
        elif mode == 'WORD':
            return _encode_and_save_keras(sentences, tokenizer, save_path)


def _decode_sentence_bpe(sequence, tokenizer):
    return tokenizer.decode(sequence)


def _decode_sentence_tokenizer(sequence, tokenizer, join_str=''):
    sentence = [tokenizer.index_word[idx.numpy()] for idx in sequence
                if idx != [tokenizer.word_index[_config.start_word]]]
    sentence = join_str.join(sentence)
    return sentence


def decode_sentence(sentence, tokenizer, language, model_type="nmt"):
    # 根据语言判断mode，支持语言：en、zh
    mode, _ = _get_mode_and_path_tokenize(language, model_type)
    if language == "en":
        if mode == 'BPE':
            return _decode_sentence_bpe(sentence, tokenizer)
        elif mode == 'WORD':
            return _decode_sentence_tokenizer(sentence, tokenizer, join_str=' ')
    elif language == "zh":
        if mode == 'CHAR':
            return _decode_sentence_tokenizer(sentence, tokenizer)
        elif mode == 'WORD':
            return _decode_sentence_tokenizer(sentence, tokenizer)
