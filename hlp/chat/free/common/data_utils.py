import os
import json
import jieba
import numpy as np
import tensorflow as tf
from pathlib import Path
import config.get_config as _config
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_sentence(start_sign, end_sign, w):
    """
    用于给句子首尾添加start和end
    :param w:
    :return: 合成之后的句子
    """
    w = start_sign + ' ' + w + ' ' + end_sign
    return w


def preprocess_request(sentence, token, max_length):
    sentence = " ".join(jieba.cut(sentence))
    sentence = preprocess_sentence(sentence, _config.start_sign, _config.end_sign)
    inputs = [token.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    dec_input = tf.expand_dims([token[_config.start_sign]], 0)

    return inputs, dec_input


def create_dataset(path, num_examples, start_sign, end_sign):
    """
    用于将分词文本读入内存，并整理成问答对
    :param path:
    :param num_examples:
    :return: 整理好的问答对
    """
    is_exist = Path(path)
    if not is_exist.exists():
        print('不存在已经分词好的文件，请先执行pre_treat模式')
        exit(0)
    with open(path, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split('\n')
        diag_weight = []
        word_pairs = []
        if num_examples == 0:
            for line in lines:
                temp = line.split("=")
                word_pairs.append([preprocess_sentence(start_sign, end_sign, w) for w in temp[0].split('\t')])
                if len(temp) == 1:
                    diag_weight.append(float(1))
                else:
                    diag_weight.append(float(temp[1]))
        else:
            for line in lines[:num_examples]:
                temp = line.split("=")
                word_pairs.append([preprocess_sentence(start_sign, end_sign, w) for w in temp[0].split('\t')])
                if len(temp) == 1:
                    diag_weight.append(float(1))
                else:
                    diag_weight.append(float(temp[1]))

    return zip(*word_pairs), diag_weight


def read_data(path, num_examples, start_sign, end_sign, max_length):
    """
    读取数据，将input和target进行分词后返回
    :param path: Tokenizer文本路径
    :param num_examples: 最大序列长度
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    (input_lang, target_lang), diag_weight = create_dataset(path, num_examples, start_sign, end_sign)
    input_tensor, target_tensor, lang_tokenizer = tokenize(input_lang, target_lang, max_length)
    return input_tensor, target_tensor, lang_tokenizer, diag_weight


def tokenize(input_lang, target_lang, max_length):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    :param input_lang: 输入
    :param target_lang: 目标
    :return: input_tensor, target_tensor, lang_tokenizer
    """
    lang = np.hstack((input_lang, target_lang))
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    input_tensor = lang_tokenizer.texts_to_sequences(input_lang)
    target_tensor = lang_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length,
                                                                  padding='post')

    return input_tensor, target_tensor, lang_tokenizer


def create_padding_mask(input):
    """
    对input中的padding单位进行mask
    :param input:
    :return:
    """
    mask = tf.cast(tf.math.equal(input, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(input):
    seq_len = tf.shape(input)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(input)
    return tf.maximum(look_ahead_mask, padding_mask)


def load_data(dict_fn, data_fn, start_sign, end_sign, checkpoint_dir, max_length, max_train_data_size=0):
    """
    数据加载方法，含四个元素的元组，包括如下：
    :return:input_tensor, input_token, target_tensor, target_token
    """
    input_tensor, target_tensor, lang_tokenizer, diag_weight = read_data(data_fn, max_train_data_size, start_sign,
                                                                         end_sign, max_length)

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(lang_tokenizer.word_index, indent=4, ensure_ascii=False))

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor, diag_weight)).cache().shuffle(
        _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    steps_per_epoch = len(input_tensor) // _config.BATCH_SIZE

    return input_tensor, target_tensor, lang_tokenizer, dataset, steps_per_epoch, checkpoint_prefix


def load_token_dict(dict_fn):
    """
    加载字典方法
    :return:input_token, target_token
    """
    is_exits = Path(dict_fn)
    if not is_exits.exists():
        print("不存在字典文件，请先执行train模式并生成字典文件")
        exit(0)

    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    inv = {}
    for key, value in token_dict.items():
        inv[value] = key

    result = []
    for text in sequences:
        temp = ''
        for token in text:
            temp = temp + ' ' + inv[token]
        result.append(temp)
    return result


def dict_texts_to_sequences(texts, token_dict):
    """
    将text转换成序列
    Args:
        texts: 文本列表
        token_dict: 字典
    Returns: 序列列表
    """
    result = []
    for text in texts:
        result.append([token_dict.get(element, 1) for element in text.split(" ")])

    return result


def smn_load_train_data(dict_fn, data_fn, checkpoint_dir, max_utterance, max_sentence, max_train_data_size=0):
    is_exist = os.path.exists(data_fn)
    if not is_exist:
        print('不存在训练数据集，请添加数据集之后重试')
        exit(0)

    print('正在读取文本数据...')
    history = []  # 用于保存每轮对话历史语句
    response = []  # 用于保存每轮对话的回答
    label = []  # 用于保存每轮对话的标签
    store = []  # 作为中间介质保存所有语句，用于字典处理
    count = 0  # 用于处理数据计数

    with open(data_fn, 'r', encoding='utf-8') as file:

        if max_train_data_size == 0:
            lines = file.read().strip().split('\n')
        else:
            lines = file.read().strip().split('\n')[:max_train_data_size]

        for line in lines:
            count += 1
            apart = line.split('\t')
            store.extend(apart[0:])
            label.append(int(apart[0]))
            response.append(apart[-1])
            del apart[0]
            del apart[-1]
            history.append(apart)
            if count % 100 == 0:
                print('已读取 {} 轮对话'.format(count))

    print('数据读取完成，正在生成字典并保存...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<UNK>')
    tokenizer.fit_on_texts(store)

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(tokenizer.word_index, indent=4, ensure_ascii=False))
    print('字典已保存，正在整理数据，生成训练数据...')
    response = tokenizer.texts_to_sequences(response)
    response = tf.keras.preprocessing.sequence.pad_sequences(response, maxlen=max_sentence, padding="post")

    count = 0
    utterances = []
    for utterance in history:
        count += 1
        pad_sequences = [0] * max_sentence
        # 注意了，这边要取每轮对话的最后max_utterances数量的语句
        utterance_padding = tokenizer.texts_to_sequences(utterance)[-max_utterance:]
        utterance_len = len(utterance_padding)
        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != 10:
            utterance_padding += [pad_sequences] * (max_utterance - utterance_len)
        utterances.append(tf.keras.preprocessing.sequence.pad_sequences(utterance_padding, maxlen=max_sentence,
                                                                        padding="post").tolist())

        if count % 100 == 0:
            print('已生成 {} 轮训练数据'.format(count))

    print('数据生成完毕，正在转换为Dataset...')
    dataset = tf.data.Dataset.from_tensor_slices((utterances, response, label)).cache().shuffle(
        _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    steps_per_epoch = len(utterances) // _config.BATCH_SIZE
    print('训练数据处理完成，正在进行训练...')

    return dataset, tokenizer, checkpoint_prefix, steps_per_epoch


def load_smn_valid_data(data_fn, max_sentence, max_utterance, max_valid_data_size,
                        token_dict=None, tokenizer=None, max_turn_utterances_num=10):
    """
    用于单独加载smn的评价数据，这个方法设计用于能够同时在train时进行评价，以及单独evaluate模式中使用
    注意了，这里token_dict和必传其一，同时传只使用tokenizer
    Args:
        data_fn: 评价数据地址
        max_sentence: 最大句子长度
        max_utterance: 最大轮次语句数量
        token_dict: 字典地址
        tokenizer: 分词器实例
        max_turn_utterances_num: dataset的批量，最好取单轮对话正负样本数总和的倍数
    Returns: dataset
    """
    if not os.path.exists(data_fn):
        return

    history = []
    response = []
    label = []
    with open(data_fn, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split("\n")[:max_valid_data_size]
        for line in lines:
            apart = line.split("\t")
            label.append(int(apart[0]))
            response.append(apart[-1])
            del apart[0]
            del apart[-1]
            history.append(apart)

    if tokenizer is not None:
        response = tokenizer.texts_to_sequences(response)
    else:
        response = dict_texts_to_sequences(response, token_dict)
    response = tf.keras.preprocessing.sequence.pad_sequences(response, maxlen=max_sentence, padding="post")

    utterances = []
    for utterance in history:
        pad_sequences = [0] * max_sentence
        if tokenizer is not None:
            utterance_padding = tokenizer.texts_to_sequences(utterance)[-max_utterance:]
        else:
            utterance_padding = dict_texts_to_sequences(utterance, token_dict)[-max_utterance:]

        utterance_len = len(utterance_padding)
        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != max_utterance:
            utterance_padding += [pad_sequences] * (max_utterance - utterance_len)
        utterances.append(tf.keras.preprocessing.sequence.pad_sequences(utterance_padding, maxlen=max_sentence,
                                                                        padding="post").tolist())

    # 在这里不对数据集进行打乱，方便用于指标计算
    dataset = tf.data.Dataset.from_tensor_slices((utterances, response, label)).prefetch(
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(max_turn_utterances_num, drop_remainder=True)

    return dataset


def get_tf_idf_top_k(history, k=5):
    """
    使用tf_idf算法计算权重最高的k个词，并返回
    Args:
        history: 上下文语句
        k: 返回词数量
    Returns: top_5_key
    """
    tf_idf = {}

    vectorizer = TfidfVectorizer(analyzer='char_wb')
    weights = vectorizer.fit_transform(history).toarray()[-1]
    key_words = vectorizer.get_feature_names()

    for i in range(len(weights)):
        tf_idf[key_words[i]] = weights[i]

    top_k_key = []
    tf_idf_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:k]
    for element in tf_idf_sorted:
        top_k_key.append(element[0])

    return top_k_key


def creat_index_dataset(data_fn, database_fn, max_database_size):
    """
    生成轮次tf-idf为索引的候选回复
    Args:
        data_fn: 文本数据路径
        database_fn: 保存候选数据路径
        max_database_size: 从文本中读取最大数据量
    Returns:
    """
    if not os.path.exists(data_fn):
        print("没有找到对应的文本数据，请确认文本数据存在")
        exit(0)

    tf_idf = {}
    count = 0

    print("检测到对应文本，正在处理文本数据...")
    with open(data_fn, 'r', encoding='utf-8') as file:
        if max_database_size == 0:
            lines = file.read().strip().split("\n")
        else:
            lines = file.read().strip().split("\n")[:max_database_size]

        for line in lines:
            count += 1
            apart = line.split("\t")[1:]
            for i in range(len(apart) - 1):
                key_words = get_tf_idf_top_k(apart[:i + 1], 5)
                if tf_idf.get('-'.join(key_words), '[NONE]') is '[NONE]':
                    tf_idf['-'.join(key_words)] = [apart[i + 1]]
                else:
                    tf_idf['-'.join(key_words)].append(apart[i + 1])

            if count % 100 == 0:
                print("已处理了 {} 轮次对话".format(count))

    with open(database_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(tf_idf, indent=4, ensure_ascii=False))

    print("文本处理完毕，已保存tf-idf候选回复集")
