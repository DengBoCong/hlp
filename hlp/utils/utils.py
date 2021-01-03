import os
import tensorflow as tf


def load_tokenizer(dict_path: str):
    """
    通过字典加载tokenizer
    :param dict_path: 字典路径
    :return tokenizer: 分词器
    """
    if not os.path.exists(dict_path):
        print("字典不存在，请检查之后重试")
        exit(0)
    with open(dict_path, 'r', encoding='utf-8') as dict_file:
        json_string = dict_file.read().strip().strip("\n")
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)

    return tokenizer
