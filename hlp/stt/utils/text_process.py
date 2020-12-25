import tensorflow as tf
from hlp.utils import text_split


def tokenize_and_encode(texts: list, dict_path: str, max_len: int,
                        num_words: int, unk_token: str = "<unk>"):
    """
    用于将文本序列集合转化为token序列
    :param texts: 文本序列列表
    :param dict_path: 字典保存路径
    :param max_len: 文本最大长度
    :param num_words:最多保存词汇数量
    :param unk_token: 未登录词
    :return texts: 处理好的文本token序列
    :return tokenizer: tokenizer
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token=unk_token, num_words=num_words)
    tokenizer.fit_on_texts(texts)
    texts = tokenizer.texts_to_sequences(texts)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=max_len, padding="post")

    with open(dict_path, 'w', encoding="utf-8") as dict_file:
        dict_file.write(tokenizer.to_json())

    return texts, tokenizer

###################################################################
###################################################################


def split_and_encode(sentences, mode, word_index):
    """对文本进行切分和编码

    :param sentences: 文本列表
    :param mode: 切分模式
    :param word_index: 词典
    :return: 文本编码序列
    """
    splitted_sentences = split_sentences(sentences, mode)
    text_int_sequences_list = encode_texts(splitted_sentences, word_index)
    return text_int_sequences_list


# token转换成id
def encode_texts(splitted_sentences, word_index):
    text_int_sequences = []
    for splitted_sentence in splitted_sentences:
        text_int_sequences.append(encode_text(splitted_sentence, word_index))
    return text_int_sequences


# token转换成id
def encode_text(splitted_sentence, word_index):
    int_sequence = []
    for c in splitted_sentence.split(" "):
        int_sequence.append(int(word_index[c]))
    return int_sequence


def split_sentence(line, mode):
    """对转写文本进行切分

    :param line: 转写文本
    :param mode: 语料文本的切分方法
    :return: 切分后的文本，以空格分隔的字符串
    """
    if mode.lower() == "cn":
        return _split_sentence_cn(line)
    elif mode.lower() == "en_word":
        return _split_sentence_en_word(line)
    elif mode.lower() == "en_char":
        return _split_sentence_en_char(line)
    elif mode.lower() == "las_cn":
        return _split_sentence_las_cn_char(line)
    elif mode.lower() == "las_en_word":
        return _split_sentence_las_en_word(line)
    elif mode.lower() == "las_en_char":
        return _split_sentence_las_en_char(line)


def split_sentences(sentences, mode):
    """对文本进行切换

    :param sentences: 待切分文本序列
    :param mode: 切分模式
    :return: 空格分隔的token串的列表
    """
    text_list = []
    for text in sentences:
        text_list.append(split_sentence(text, mode))
    return text_list


def _split_sentence_en_word(s):
    result = text_split.split_en_word(s)
    return result


def _split_sentence_en_char(s):
    result = text_split.split_en_char(s)
    return result


def _split_sentence_las_en_char(s):
    s = text_split.split_en_char(s)

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s.insert(0, '<start>')
    s.append('<end>')

    return s


def _split_sentence_las_en_word(s):
    s = text_split.split_en_word(s)

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s.insert(0, '<start>')
    s.append('<end>')

    return s


def _split_sentence_cn(s):
    result = text_split.split_zh_char(s)
    return result


def _split_sentence_las_cn_char(s):
    s = text_split.split_zh_char(s)

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s.insert(0, '<start>')
    s.append('<end>')

    return s


# 获取最长的label_length
def get_max_label_length(text_int_sequences):
    return max(len(seq) for seq in text_int_sequences)


def get_label_and_length(text_int_sequences_list, max_label_length):
    target_length_list = []
    for text_int_sequence in text_int_sequences_list:
        target_length_list.append([len(text_int_sequence)])
    target_tensor_numpy = tf.keras.preprocessing.sequence.pad_sequences(text_int_sequences_list,
                                                                        maxlen=max_label_length,
                                                                        padding='post'
                                                                        )
    target_length = tf.convert_to_tensor(target_length_list)
    return target_tensor_numpy, target_length


# def tokenize_and_encode(texts):
#     """ 对文本进行tokenize和编码
#
#     :param texts: 已经用空格分隔的文本列表
#     :return: 文本编码序列, tokenizer
#     """
#     tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
#     tokenizer.fit_on_texts(texts)
#     text_int_sequences = tokenizer.texts_to_sequences(texts)
#     return text_int_sequences, tokenizer


# 将输出token id序列解码为token序列
def int_to_text_sequence(seq, index_word, mode):
    if mode.lower() == "cn":
        return int_to_text_sequence_cn(seq, index_word)
    elif mode.lower() == "en_word":
        return int_to_text_sequence_en_word(seq, index_word)
    elif mode.lower() == "en_char":
        return int_to_text_sequence_en_char(seq, index_word)


def int_to_text_sequence_cn(ids, index_word):
    result = []
    for i in ids:
        if 1 <= i <= len(index_word):
            word = index_word[str(i)]
            result.append(word)
    return "".join(result).strip()


def int_to_text_sequence_en_word(ids, index_word):
    result = []
    for i in ids:
        if 1 <= i <= (len(index_word)):
            word = index_word[str(i)]
            result.append(word)
            result.append(" ")
    return "".join(result).strip()


def int_to_text_sequence_en_char(ids, index_word):
    result = []
    for i in ids:
        if 1 <= i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<space>":
                result.append(word)
            else:
                result.append(" ")
    return "".join(result).strip()
