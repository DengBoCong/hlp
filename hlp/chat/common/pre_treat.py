import os
import re
import jieba
import config.getConfig as gConfig
from zhon.hanzi import punctuation

'''
对话数据集预处理模块
raw_data保存原数据集路径
tokenized_data保存分词后的数据集路径
'''


def preprocess_raw_data():  # tokenizer
    raw_data = gConfig.resource_data
    tokenized_data = gConfig.tokenized_data

    # 首先判断原数据集是否存在，不存在则退出
    if not os.path.exists(raw_data):
        exit()

    pairs = []

    # 对每一轮对话上下文进行配对，形成一问一答两个部分，如果遇到
    # 下一轮对话，直接跳过
    with open(raw_data, encoding='utf-8') as raw_file:
        one_pair = []
        pair_count = 0
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            line = re.sub(r"[%s]+" % punctuation, "", line)
            if line == '':
                one_pair = []
                continue
            elif len(one_pair) == 1:
                one_pair.append(line)
                pairs.append(one_pair)
                one_pair = [line]
                pair_count += 1
                print('已处理：', pair_count, '个问答对')
            else:
                one_pair.append(line)

    results = []
    # 接下来，我们把上面的对话内容进行分词，并存入train_tokenized文本中
    for pair in pairs:
        if len(pair) != 2:
            continue

        # 使用jieba分词器进行分词
        pair[0] = " ".join(jieba.cut(pair[0]))
        pair[1] = " ".join(jieba.cut(pair[1]))
        results.append(pair[0] + '\t' + pair[1])

    train_tokenized = open(tokenized_data, 'w', encoding='utf-8')
    for i in range(len(results)):
        train_tokenized.write(results[i] + '\n')

        if i % 1000 == 0:
            print(len(range(len(results))), '处理进度：', i)

    train_tokenized.close()

