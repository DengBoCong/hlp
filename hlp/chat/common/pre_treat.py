import os
import re
import jieba
import pickle
from model.task.kb import load_kb
import config.get_config as _config


'''
对话数据集预处理模块
raw_data保存原数据集路径
tokenized_data保存分词后的数据集路径
'''


def preprocess_raw_data():
    """
    用来对原始文本进行预处理的方法，主要是将原
    始文本进行分词后，保存在一个新的文本中，供后继使用
    :return:
    """
    raw_data = _config.resource_data
    tokenized_data = _config.tokenized_data

    # 首先判断原数据集是否存在，不存在则退出
    if not os.path.exists(raw_data):
        print('数据集不存在，请添加数据集!')
        exit()

    pairs = []

    # 对每一轮对话上下文进行配对，形成一问一答两个部分，如果遇到
    # 下一轮对话，直接跳过
    with open(raw_data, encoding='utf-8') as raw_file:
        one_pair = []
        pair_count = 0
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            # line = re.sub(r"[%s]+" % punctuation, "", line)
            # 因为原始数据集中，是一轮一轮的对话排列的，所以需要注意的是
            # 在一轮对话结束之后，最后一句不能作为问句，需要跳到下一轮进行处理
            if line == '':
                one_pair = []
                continue
            elif len(one_pair) == 1:
                one_pair.append(line)
                pairs.append(one_pair)
                one_pair = [line]
                pair_count += 1
                if pair_count % 10000 == 0:
                    print('已处理：', pair_count, '个问答对')
            else:
                one_pair.append(line)

    print('读取完毕，处理中...')
    results = []
    # 接下来，我们把上面的对话内容进行分词，并存入train_tokenized文本中
    for pair in pairs:
        if len(pair) != 2:
            continue

        # 使用jieba分词器进行分词
        pair[0] = " ".join(jieba.cut(pair[0]))
        pair[1] = " ".join(jieba.cut(pair[1]))
        results.append(pair[0] + '\t' + pair[1])

    # 将处理之后存在内存中的数据写入到新文本中
    train_tokenized = open(tokenized_data, 'w', encoding='utf-8')
    for i in range(len(results)):
        train_tokenized.write(results[i] + '\n')
        if i % 10000 == 0:
            print(len(range(len(results))), '处理进度：', i)

    train_tokenized.close()

def load_embed_model(fn, embed_dim=300):
    with open(fn + '.w2i', 'rb') as file:
        word2idx = pickle.load(file)

def load_embed(**kargs):
    if not kargs["generate_embed"]:
        word2idx, embed = loa

def load_data(**kargs):
    kb = load_kb(kargs["kb"], "name")
