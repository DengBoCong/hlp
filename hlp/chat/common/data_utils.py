import os
import json
import jieba
import pysolr
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_sentence(start_sign: str, end_sign: str, sentence: str):
    """
    用于给句子首尾添加start和end
    Args:
        start_sign: 开始标记
        end_sign: 结束标记
        sentence: 待处理句子
    Returns:
        sentence: 合成之后的句子
    """
    sentence = start_sign + ' ' + sentence + ' ' + end_sign
    return sentence


def preprocess_request(sentence: str, token: dict, max_length: int, start_sign: str, end_sign: str):
    """
    用于处理回复功能的输入句子，返回模型使用的序列
    Args:
        sentence: 待处理句子
        token: 字典
        max_length: 单个句子最大长度
        start_sign: 开始标记
        end_sign: 结束标记
    Returns:
        inputs: 处理好的句子
        dec_input: decoder输入
    """
    sentence = " ".join(jieba.cut(sentence))
    sentence = preprocess_sentence(sentence, start_sign, end_sign)
    inputs = [token.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    dec_input = tf.expand_dims([token[start_sign]], 0)

    return inputs, dec_input


def create_dataset(path: str, num_examples: int, start_sign: str, end_sign: str):
    """
    用于将分词文本读入内存，并整理成问答对
    Args:
        path: 分词文本路径
        num_examples: 读取的数据量大小
        start_sign: 开始标记
        end_sign: 结束标记
    Returns:
        zip(*word_pairs): 整理好的问答对
        diag_weight: 样本权重
    """
    is_exist = Path(path)
    if not is_exist.exists():
        print('不存在已经分词好的文件，请先执行pre_treat模式')
        exit(0)
    with open(path, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split('\n')
        diag_weight = []
        word_pairs = []
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            # 文本数据中的问答对权重通过在问答对尾部添加“<|>”配置
            temp = line.split("<|>")
            word_pairs.append([preprocess_sentence(start_sign, end_sign, w) for w in temp[0].split('\t')])
            # 如果没有配置对应问答对权重，则默认为1.
            if len(temp) == 1:
                diag_weight.append(float(1))
            else:
                diag_weight.append(float(temp[1]))

    return zip(*word_pairs), diag_weight


def read_data(path: str, num_examples: int, start_sign: str, end_sign: str, max_length: int,
              tokenizer: tf.keras.preprocessing.text.Tokenizer = None):
    """
    读取数据，将input和target进行分词后返回
    Args:
        path: 分词文本路径
        num_examples: 读取的数据量大小
        start_sign: 开始标记
        end_sign: 结束标记
        max_length: 最大序列长度
        tokenizer: 传入现有的分词器，默认重新生成
    Returns:
        input_tensor: 输入序列张量
        target_tensor: 目标序列张量
        lang_tokenizer: 分词器
    """
    (input_lang, target_lang), diag_weight = create_dataset(path, num_examples, start_sign, end_sign)
    input_tensor, target_tensor, lang_tokenizer = tokenize(input_lang, target_lang, max_length, tokenizer)
    return input_tensor, target_tensor, lang_tokenizer, diag_weight


def tokenize(input_lang: list, target_lang: list, max_length: int,
             tokenizer: tf.keras.preprocessing.text.Tokenizer = None):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    Args:
        input_lang: 输入序列
        target_lang: 目标序列
        max_length: 最大序列长度
        tokenizer: 传入现有的分词器，默认重新生成
    Returns:
        input_tensor: 输入序列张量
        target_tensor: 目标序列张量
        lang_tokenizer: 分词器
    """
    lang = np.hstack((input_lang, target_lang))
    if tokenizer is not None:
        lang_tokenizer = tokenizer
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    input_tensor = lang_tokenizer.texts_to_sequences(input_lang)
    target_tensor = lang_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length,
                                                                  padding='post')

    return input_tensor, target_tensor, lang_tokenizer


def load_data(dict_fn: str, data_fn: str, start_sign: str, end_sign: str, buffer_size: int,
              batch_size: int, checkpoint_dir: str, max_length: int, valid_data_split: float = 0.0,
              valid_data_fn: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    数据加载方法，含四个元素的元组，包括如下：
    Args:
        dict_fn: 字典路径
        data_fn: 文本数据路径
        start_sign: 开始标记
        end_sign: 结束标记
        buffer_size: Dataset加载缓存大小
        batch_size: Dataset加载批大小
        checkpoint_dir: 检查点保存路径
        max_length: 单个句子最大长度
        valid_data_split: 用于从训练数据中划分验证数据
        valid_data_fn: 验证数据文本路径
        max_train_data_size: 最大训练数据量
        max_valid_data_size: 最大验证数据量
    Returns:
        train_dataset: 训练Dataset
        valid_dataset: 验证Dataset
        steps_per_epoch: 训练数据总共的步数
        valid_steps_per_epoc: 验证数据总共的步数
        checkpoint_prefix: 检查点前缀
    """
    train_input, train_target, lang_tokenizer, diag_weight = read_data(data_fn, max_train_data_size,
                                                                       start_sign, end_sign, max_length)
    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0

    if valid_data_fn != "":
        valid_input, valid_target, _, _ = read_data(valid_data_fn, max_valid_data_size, start_sign,
                                                    end_sign, max_length, tokenizer=lang_tokenizer)
    elif valid_data_split != 0.0:
        train_size = int(len(train_input) * (1.0 - valid_data_split))
        valid_input = train_input[train_size:]
        valid_target = train_target[train_size:]
        train_input = train_input[:train_size]
        train_target = train_target[:train_size]
        diag_weight = diag_weight[:train_size]
    else:
        valid_flag = False

    with open(dict_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(lang_tokenizer.word_index, indent=4, ensure_ascii=False))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target, diag_weight)).cache().shuffle(
        buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    if valid_flag:
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_target)).cache().shuffle(
            buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
        valid_steps_per_epoch = len(valid_input) // batch_size
    else:
        valid_dataset = None

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    steps_per_epoch = len(train_input) // batch_size

    return train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch, checkpoint_prefix


def load_token_dict(dict_fn: str):
    """
    加载字典方法
    Args:
        dict_fn: 字典路径
    Returns:
        token: 字典
    """
    if not os.path.exists(dict_fn):
        print("不存在字典文件，请先执行train模式并生成字典文件")
        exit(0)

    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


def sequences_to_texts(sequences: list, token_dict: dict):
    """
    将序列转换成text
    Args:
        sequences: 待处理序列
        token_dict: 字典文本路径
    Returns:
        result: 处理完成的序列
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


def dict_texts_to_sequences(texts: list, token_dict: dict):
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


def smn_load_train_data(dict_fn: str, data_fn: str, checkpoint_dir: str, buffer_size: int,
                        batch_size: int, max_utterance: int, max_sentence: int, max_train_data_size: int = 0):
    """
    用于SMN的训练数据加载
    Args:
        dict_fn: 字典文本路径
        data_fn: 数据文本路径
        buffer_size: Dataset加载缓存大小
        batch_size: Dataset加载批大小
        checkpoint_dir: 检查点保存路径
        max_utterance: 每轮对话最大对话数
        max_sentence: 单个句子最大长度
        max_train_data_size: 最大训练数据量
    Returns:
        dataset: TensorFlow的数据处理类
        tokenizer: 分词器
        checkpoint_prefix: 检查点前缀
        steps_per_epoch: 总的步数
    """
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
        buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    steps_per_epoch = len(utterances) // batch_size
    print('训练数据处理完成，正在进行训练...')

    return dataset, tokenizer, checkpoint_prefix, steps_per_epoch


def load_smn_valid_data(data_fn: str, max_sentence: int, max_utterance: int, max_valid_data_size: int,
                        token_dict: dict = None, tokenizer: tf.keras.preprocessing.text.Tokenizer = None,
                        max_turn_utterances_num: int = 10):
    """
    用于单独加载smn的评价数据，这个方法设计用于能够同时在train时进行评价，以及单独evaluate模式中使用
    注意了，这里token_dict和必传其一，同时传只使用tokenizer
    Args:
        data_fn: 评价数据地址
        max_sentence: 最大句子长度
        max_utterance: 最大轮次语句数量
        max_valid_data_size: 最大验证数据量
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


def get_tf_idf_top_k(history: list, k: int = 5):
    """
    使用tf_idf算法计算权重最高的k个词，并返回
    Args:
        history: 上下文语句
        k: 返回词数量
    Returns: top_5_key
    """
    tf_idf = {}

    vectorizer = TfidfVectorizer(analyzer='word')
    weights = vectorizer.fit_transform(history).toarray()[-1]
    key_words = vectorizer.get_feature_names()

    for i in range(len(weights)):
        tf_idf[key_words[i]] = weights[i]

    top_k_key = []
    tf_idf_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:k]
    for element in tf_idf_sorted:
        top_k_key.append(element[0])

    return top_k_key


def creat_index_dataset(data_fn: str, solr_sever: str, max_database_size: int):
    """
    生成轮次tf-idf为索引的候选回复
    Args:
        data_fn: 文本数据路径
        solr_sever: solr服务的地址
        max_database_size: 从文本中读取最大数据量
    Returns:
    """
    if not os.path.exists(data_fn):
        print("没有找到对应的文本数据，请确认文本数据存在")
        exit(0)

    responses = []
    count = 0
    solr = pysolr.Solr(url=solr_sever, always_commit=True)
    solr.ping()

    print("检测到对应文本，正在处理文本数据...")
    with open(data_fn, 'r', encoding='utf-8') as file:
        if max_database_size == 0:
            lines = file.read().strip().split("\n")
        else:
            lines = file.read().strip().split("\n")[:max_database_size]
        lines = lines[::2]

        for line in lines:
            count += 1
            apart = line.split("\t")[1:]
            for i in range(len(apart)):
                responses.append({
                    "utterance": apart[i]
                })

            if count % 100 == 0:
                print("已处理了 {} 轮次对话".format(count))
    solr.delete(q="*:*")
    solr.add(docs=responses)

    print("文本处理完毕，已更新候选回复集")
