import re
import jieba

from hlp.mt.config import get_config as _config
from hlp.utils import text_split


def _preprocess_sentence_en_bpe(sentence, start_word=_config.start_word, end_word=_config.end_word):
    """对BPE分词方法进行预处理"""
    sentence = start_word + ' ' + sentence + ' ' + end_word
    return sentence


def preprocess_sentences_en(sentences, mode=_config.en_tokenize_type, start_word=_config.start_word,
                            end_word=_config.end_word):
    """
    对英文句子列表进行指定mode的预处理
    返回处理好的句子列表
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


def preprocess_sentences_zh(sentences, mode=_config.zh_tokenize_type, start_word=_config.start_word,
                            end_word=_config.end_word):
    """
    对中文句子列表进行指定mode的预处理
    返回处理好的句子列表
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
    """通过language判断mode"""
    if language == "en":
        return preprocess_sentences_en(sentences, mode)
    elif language == "zh":
        return preprocess_sentences_zh(sentences, mode)
