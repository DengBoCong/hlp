import re
import jieba

from hlp.mt.config import get_config as _config


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


def preprocess_sentences_zh(sentences, mode=_config.zh_tokenize_type, start_word=_config.start_word,
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
        return preprocess_sentences_en(sentences, mode)
    elif language == "zh":
        mode = _config.zh_tokenize_type
        return preprocess_sentences_zh(sentences, mode)