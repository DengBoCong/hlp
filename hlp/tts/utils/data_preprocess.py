import re
import os
import json
import inflect
import numpy as np
import tensorflow as tf
from unidecode import unidecode
from hlp.tts.utils.utils import get_phoneme_dict_symbols


# dict_fn: str, data_fn: str, start_sign: str, end_sign: str, buffer_size: int,
#               batch_size: int, checkpoint_dir: str, max_length: int, valid_data_split: float = 0.0,
#               valid_data_fn: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0

def load_data(data_path: str, max_len: int, vocab_size: int, tokenized_type: str = "phoneme", dict_path: str = ""):
    """
    加载训练验证数据方法，非phoneme的方法将会保存字典
    :param data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param dict_path: 字典路径，若使用phoneme则不用传
    :return: 返回dataset
    """
    if not os.path.exists(data_path):
        print("加载的训练验证数据文件不存在，请先执行pre_treat模式后重试")
        exit(0)

    audio_data_pair = []
    sentence_data = []
    print("正在加载训练验证数据...")
    with open(data_path, 'r', encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip().strip("\n").replace("/", " ").split("\t")
            sentence_data.append(line[-1])
            line.pop(-1)
            audio_data_pair.append(line)

    if tokenized_type == "phoneme":
        sentence_sequences = text_to_sequence_phoneme(texts=sentence_data, max_len=max_len)
    else:
        if dict_path == "":
            print("请在加载数据时，传入字典保存路径")
            exit(0)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>", num_words=vocab_size)
        tokenizer.fit_on_texts(sentence_data)
        sentence_sequences = tokenizer.texts_to_sequences(sentence_data)
        with open(dict_path, 'w', encoding="utf-8") as dict_file:
            dict_file.write(json.dumps(tokenizer.to_json(), indent=4, ensure_ascii=False))

    dataset = tf.data.Dataset.from_tensor_slices((audio_data_pair, sentence_sequences)).cache().shuffle(
        20000).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(deal_audio_sentence_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1, drop_remainder=True)
    print("训练验证数据加载完毕")

    for (i, (mel, mag, stop_token, sentence)) in enumerate(dataset.take(6)):
        print("哈哈：")
        print(mel)
    exit(0)
    return dataset


def read_npy_file(filename):
    data = np.load(filename.numpy().decode())
    return data.astype(np.float32)


def deal_audio_sentence_pairs(audio_data_pair: tf.Tensor, sentence: tf.Tensor):
    """
    用于处理音频句子对，将其转化为张量
    :param audio_data_pair: 音频相关数据对，mel、mag、stop_token保存文件
    :param sentence: 音频句子对
    :return: mel, mag, stop_token, sentence
    """
    [mel, ] = tf.py_function(read_npy_file, [audio_data_pair[0]], [tf.float32, ])
    [mag, ] = tf.py_function(read_npy_file, [audio_data_pair[1]], [tf.float32, ])
    [stop_token, ] = tf.py_function(read_npy_file, [audio_data_pair[2]], [tf.float32, ])

    return mel, mag, stop_token, sentence


def text_to_phonemes_converter(text: str, cmu_dict_path: str):
    """
    将句子按照CMU音素字典进行分词切分
    :param text: 单个句子文本
    :param cmu_dict_path: cmu音素字典路径
    :return: 按照音素分词好的数组
    """
    _, symbols_set = get_phoneme_dict_symbols()

    alt_re = re.compile(r'\([0-9]+\)')
    cmu_dict = {}
    text = _clean_text(text)
    text = re.sub(r"([?.!,])", r" \1", text)

    # 文件是从官网下载的，所以文件编码格式要用latin-1
    with open(cmu_dict_path, 'r', encoding='latin-1') as cmu_file:
        for line in cmu_file:
            if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
                parts = line.split('  ')
                word = re.sub(alt_re, '', parts[0])

                # 这里要将非cmu音素的干扰排除
                pronunciation = " "
                temps = parts[1].strip().split(' ')
                for temp in temps:
                    if temp not in symbols_set:
                        pronunciation = None
                        break
                if pronunciation:
                    pronunciation = ' '.join(temps)
                    if word in cmu_dict:
                        cmu_dict[word].append(pronunciation)
                    else:
                        cmu_dict[word] = [pronunciation]

    cmu_result = []
    for word in text.split(' '):
        # 因为同一个单词，它的发音音素可能不一样，所以存在多个
        # 音素分词，我这里就单纯的取第一个，后面再改进和优化
        cmu_word = cmu_dict.get(word.upper(), [word])[0]
        if cmu_word != word:
            cmu_result.append("{" + cmu_word + "}")
        else:
            cmu_result.append(cmu_word)

    return " ".join(cmu_result)


def text_to_word_converter(text: str):
    """
    按照单词进行切分
    :param text: 单个句子文本
    :return: 处理好的文本序列
    """
    text = _clean_text(text)
    text = text.lower().strip()
    text = re.sub(r"([?.!,])", r" \1 ", text)  # 切分断句的标点符号
    text = re.sub(r'[" "]+', " ", text)  # 合并多个空格
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = text.strip()
    return text


def text_to_char_converter(text: str):
    """
    按照字符进行切分
    :param text: 单个句子文本
    :return: 处理好的文本序列
    """
    text = _clean_text(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = re.sub(r'[" "]+', " ", text)
    sentences = text.strip()

    result = ""
    for sentence in sentences:
        result = result + ' ' + sentence

    return result


def text_to_sequence_phoneme(texts, max_len: int):
    """
    专用于phoneme的text转序列的方法
    :param texts: 文本序列列表
    :param max_len: 文本序列最大长度
    :return: 转换后的id序列
    """
    dict_set, _ = get_phoneme_dict_symbols()

    sequences = []
    for text in texts:
        sequence = []
        while len(text):
            # 判断有没有由花括号的音素，没有就直接按照字典转换
            m = re.compile(r'(.*?)\{(.+?)\}(.*)').match(text)
            if not m:
                sequence += [dict_set[s] for s in _clean_text(text)
                             if s in dict_set and s is not 'unk' and s is not '~']
                break
            sequence += [dict_set[s] for s in _clean_text(m.group(1))
                         if s in dict_set and s is not 'unk' and s is not '~']
            sequence += [dict_set[s] for s in ['@' + s for s in m.group(2).split()]
                         if s in dict_set and s is not 'unk' and s is not '~']
            text = m.group(3)
        sequence.append(dict_set['~'])
        sequences.append(sequence)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post")
    return sequences


def _clean_text(text: str):
    """
    用于对句子进行整理，将美元、英镑、数字、小数点、序
    数词等等转化为单词，同时对部分缩写进行扩展
    :param text: 单个句子文本
    :return: 处理好的文本序列
    """
    text = unidecode(text)
    text = text.lower()
    text = _clean_number(text=text)
    text = _abbreviations_to_word(text=text)
    text = re.sub(r"\s+", " ", text)

    return text


def _clean_number(text: str):
    """
    对句子中的数字相关进行统一单词转换
    :param text: 单个句子文本
    :return: 转换后的句子文本
    """
    comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
    decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
    dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
    ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    number_re = re.compile(r"[0-9]+")

    text = re.sub(comma_number_re, lambda m: m.group(1).replace(',', ''), text)
    text = re.sub(pounds_re, r"\1 pounds", text)
    text = re.sub(dollars_re, _dollars_to_word, text)
    text = re.sub(decimal_number_re, lambda m: m.group(1).replace('.', ' point '), text)
    text = re.sub(ordinal_re, lambda m: inflect.engine().number_to_words(m.group(0)), text)
    text = re.sub(number_re, _number_to_word, text)

    return text


def _number_to_word(number_re: str):
    """
    将数字转为单词
    :param number_re: 数字匹配式
    :return:
    """
    num = int(number_re.group(0))
    tool = inflect.engine()

    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + tool.number_to_words(num % 100)
        elif num % 100 == 0:
            return tool.number_to_words(num // 100) + " hundred"
        else:
            return tool.number_to_words(num, andword="", zero='oh', group=2).replace(", ", " ")
    else:
        return tool.number_to_words(num, andword="")


def _dollars_to_word(dollars_re: str):
    """
    将美元转为单词
    :param dollars_re: 美元匹配式
    :return:
    """
    match = dollars_re.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        # 注意力，不符合格式的要直接返回
        return match + ' dollars'
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _abbreviations_to_word(text: str):
    """
    对句子中的压缩次进行扩展成单词
    :param text: 单个句子文本
    :return: 转换后的句子文本
    """
    abbreviations = [
        (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort')
        ]
    ]

    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)

    return text
