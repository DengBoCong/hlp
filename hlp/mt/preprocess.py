import tensorflow as tf
import re
import io
from sklearn.model_selection import train_test_split


# 对英文句子：小写化，切分句子，添加开始和结束标记
def preprocess_en_sentence(s):
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格

    # 例如： "he is a boy." => "he is a boy ."
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s


# 对中文句子预处理：按字切分句子，添加开始和结束标记
def preprocess_ch_sentence(s):
    s = s.lower().strip()

    s = [c for c in s]
    s = ' '.join(s)  # 给字之间加上空格
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s


# en_sentence = u"May I borrow this book?"   # <start> may i borrow this book ? <end>
# print(preprocess_en_sentence(en_sentence))
#
# ch_sentence = '这个对吗？ 试试看。'  # <start> 这 个 对 吗 ？ 试 试 看 。 <end>
# print(preprocess_ch_sentence(ch_sentence))


def max_length(texts):
    return max(len(t) for t in texts)




def create_dataset(path, num_examples):
    """
    :param path:文本路径
    :param num_examples:取出文本数量
    :return: 处理后的英中句子列表

    输入文本格式为：每行为对应中英句子对，以\t间隔
    返回为预处理后的中英句子列表
    处理后的英文句子每个词改成小写，词间添加空格，并添加开始结束标记，例：<start> may i borrow this book ? <end>
    处理后的英文句子每个字间添加空格，并添加开始结束标记，例：<start> 这 个 对 吗 ？ 试 试 看 。 <end>

    """
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    en_sentences = [l.split('\t')[0] for l in lines[:num_examples]]
    ch_sentences = [l.split('\t')[1] for l in lines[:num_examples]]

    en_sentences = [preprocess_en_sentence(s) for s in en_sentences]
    ch_sentences = [preprocess_ch_sentence(s) for s in ch_sentences]

    print("导入文本的英中句子数：",len(en_sentences), len(ch_sentences))  # 句子对总数
    # print("导入文本的英中句子最大长度：",print(max_length(en_sentences), max_length(ch_sentences)))  # 句子最大长度

    return en_sentences, ch_sentences


def tokenize(texts):
    """
    :param texts: 用以训练词典的文本列表
    :return:编码后的文本列表、字典

    此函数用于将文本进一步编码，token数字化，后补齐
    并得到文本的字典

    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                         padding='post')

    return sequences, tokenizer


# sequences, tokenizer = tokenize(["<start> may i borrow this book ? <end>", "<start> i love you ! <end>"])
# print("sequences:",sequences)  # [[ 1  4  2  5  6  7  8  3] [ 1  2  9 10 11  3  0  0]]
# print(tokenizer.word_index)
# # {'<start>': 1, 'i': 2, '<end>': 3,  # 'may': 4,
# # 'borrow': 5, 'this': 6, 'book': 7, '?': 8, 'love': 9, 'you': 10, '!': 11}


def load_dataset(path, num_examples=None):
    """
    :param path: 文本路径
    :param num_examples: 取出的句子数量
    :return: input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    返回编码后的英中句子对及英中字典

    """
    # 创建清理过的输入输出对
    inp_lang, targ_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    print('英文词典大小：', len(inp_lang_tokenizer.word_index))  # 英文词典大小
    print('中文词典大小：', len(targ_lang_tokenizer.word_index))  # 中文字典大小

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def encode(inp_sentence):
    inp_sentence = '<start> ' + inp_sentence + ' <end>'
    inp_sentence = [inp_lang_tokenizer.word_index[i] for i in inp_sentence.split(' ')]  # token编码
    encoder_input = tf.expand_dims(inp_sentence, 0)
    return  encoder_input


if __name__ == '__main__':
    # 生成编码后的中英句子及字典
    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset('en-ch.txt', 20000)

    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2, shuffle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_dataset = train_dataset.shuffle(10000).padded_batch(batch_size=64,
                                                              padded_shapes=((len(x_train[0]),), (len(y_train[0]),)))

    print(train_dataset)

    en_batch, ch_batch = next(iter(train_dataset))
    print(en_batch, ch_batch)

    print('-' * 50)






def create_eval_dataset(path, num_examples):
    """
    :param path:文本路径
    :param num_examples:取出文本数量
    :return: 处理后的英中句子列表

    输入文本格式为：每行为对应中英句子对，以\t间隔
    返回为预处理后的中英句子列表
    处理后的英文句子每个词改成小写，词间添加空格，并添加开始结束标记，例：<start> may i borrow this book ? <end>
    中文句子不做处理

    """
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    en_sentences = [l.split('\t')[0] for l in lines[:num_examples]]
    ch_sentences = [l.split('\t')[1] for l in lines[:num_examples]]

    en_sentences = [preprocess_en_sentence(s) for s in en_sentences]


    print("导入文本的英中句子数：",len(en_sentences), len(ch_sentences))  # 句子对总数
    # print("导入文本的英中句子最大长度：",print(max_length(en_sentences), max_length(ch_sentences)))  # 句子最大长度

    return en_sentences, ch_sentences
