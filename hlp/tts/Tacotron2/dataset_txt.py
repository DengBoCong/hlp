import csv
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import re
import io
# 将csv数据写到txt中
with open('E:\LJSpeech-1.1\LJSpeech-1.1\metadata.csv',
              'r', encoding='UTF-8') as f:
    reader = csv.reader(f)
    j = 0
    for row in reader:
        row[0] = row[0][11:]
        str = ''.join(row)
        print(str)
        with open('文字预处理.txt', 'a', encoding='UTF-8') as w:
                w.write(str + '\n')
    w.close()
    f.close()

# 因为有很多异常数据所以要对写好的txt处理
with open('文字预处理.txt', 'r', encoding='UTF-8') as w:
    with open('文字.txt', 'a', encoding='UTF-8') as f:
        for row in w:
            i = 0
            str = []
            if row[0] == 'L':
                for i in range(len(row) - 11):
                    str = row[i + 11]
                f.write(str)
            else:
                f.write(row)
        f.write('\n')
w.close()
f.close()
path_to_file = "文字.txt"
#对英文进行编码
def preprocess_en_sentence(s):
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    en_sentences = [l.split('\t')[0] for l in lines[:num_examples]]
    en_sentences = [preprocess_en_sentence(s) for s in en_sentences]
    return en_sentences

en = create_dataset(path_to_file, None)

def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                              padding='post')
    return sequences, tokenizer

en_seqs, en_tokenizer = tokenize(en)
print(en_seqs.shape)
print(en[0])
print(en_seqs[0])
print('英文词典大小：', len(en_tokenizer.word_index))  # 英文词典大小