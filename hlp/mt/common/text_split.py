from hlp.mt.config import get_config as _config
from hlp.utils import text_split


def _preprocess_sentence_en_bpe(sentence, start_word=_config.start_word, end_word=_config.end_word):
    sentence = start_word + ' ' + sentence + ' ' + end_word
    return sentence


def preprocess_sentences_en(sentences, mode=_config.en_tokenize_type,
                            start_word=_config.start_word, end_word=_config.end_word):
    """
    对英文句子列表进行指定mode的预处理
    返回处理好的句子列表，句子为添加开始介绍符的空格分隔的字符串
    """
    if mode == 'BPE':
        sentences = [_preprocess_sentence_en_bpe(s, start_word, end_word) for s in sentences]
        return sentences
    elif mode == 'WORD':
        sentences = [text_split.split_en_word(s) for s in sentences]
        sentences = [start_word + ' ' + ' '.join(s) + ' ' + end_word for s in sentences]
        return sentences
    else:
        return ''


def preprocess_sentences_zh(sentences, mode=_config.zh_tokenize_type,
                            start_word=_config.start_word, end_word=_config.end_word):
    """
    对中文句子列表进行指定mode的预处理
    返回处理好的句子列表，句子为添加开始介绍符的空格分隔的字符串
    """
    if mode == 'CHAR':
        sentences = [text_split.split_zh_char(s) for s in sentences]
        sentences = [start_word + ' ' + ' '.join(s) + ' ' + end_word for s in sentences]
        return sentences
    elif mode == 'WORD':
        sentences = [text_split.split_zh_word(s) for s in sentences]
        sentences = [start_word + ' ' + ' '.join(s) + ' ' + end_word for s in sentences]
        return sentences


def preprocess_sentences(sentences, language, mode):
    """

    :param sentences: 原始句子字符串列表
    :param language:
    :param mode:
    :return: 添加开始结束符的空格分隔的句子字符串构成的列表
    """
    if language == "en":
        return preprocess_sentences_en(sentences, mode)
    elif language == "zh":
        return preprocess_sentences_zh(sentences, mode)
