import re

import jieba


def split_en_word(s):
    """ 对英文文本小写并按词进行切换

    :param s: 待切分的英文文本
    :return: 用空格分隔的切分后的文本
    """
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    return s


def split_cn_char(s):
    """ 对中文按字进行切换

    :param s: 待切分的中文
    :return: 用空格分隔的切分后的文本
    """
    s = s.lower().strip()

    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = s.strip()

    return s


def split_cn_word(s):
    """ 对中文(含英文)按词进行切换

    :param s: 待切分的中文
    :return: 用空格分隔的切分后的文本
    """
    return ' '.join(jieba.cut(s.lower().strip()))


if __name__ == "__main__":
    en_txt1 = "I like NLP."
    print(split_en_word(en_txt1))

    en_txt1 = " I like  NLP.  "
    print(split_en_word(en_txt1))

    cn_txt1 = "我喜欢深度学习。"
    print(split_cn_char(cn_txt1))
    print(split_cn_word(cn_txt1))

    cn_txt2 = " 我喜欢深度学习。 "
    print(split_cn_char(cn_txt2))
    print(split_cn_word(cn_txt2))

    cn_txt3 = "我喜欢book."
    print(split_cn_word(cn_txt3))