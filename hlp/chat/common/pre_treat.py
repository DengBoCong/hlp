import os
import re
import json
import jieba
from collections import defaultdict

'''
对话数据集预处理模块
raw_data保存原数据集路径
tokenized_data保存分词后的数据集路径
'''


def preprocess_raw_data(raw_data, tokenized_data):
    """
    用来对原始文本进行预处理的方法，主要是将原
    始文本进行分词后，保存在一个新的文本中，供后继使用
    :param raw_data:  原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :return:
    """

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


class Delexicalizer:
    """
    去词化器
    """

    def __init__(self, info_slots, semi_dict, values, replaces):
        self.info_slots = info_slots  # 所有informable槽位信息
        self.semi_dict = semi_dict  # 语句中槽位同义词替换字典
        self.values = values  # 数据库中所有requestable槽位信息
        self.replaces = replaces

        self.inv_info_slots = self._inverse_dict(self.info_slots, '%s')  # informable槽值对字典
        self.inv_values = self._inverse_dict(self.values, '<v.%s> ',
                                             func=lambda x: x.upper())  # requestable槽值对字典，槽位已同义化
        self.inv_semi_dict = self._inverse_dict(self.semi_dict, '%s')

        self.inv_semi_dict = {k: "<v.%s> " % self.inv_info_slots[v].upper()
        if v in self.inv_info_slots else "<s.%s> " % v.upper() for k, v in self.inv_semi_dict.items()}

        self.num_matcher = re.compile(r' \d{1,2}([., ])')
        self.post_matcher = re.compile(
            r'( [.]?c\.b[.]?[ ]?\d[ ]?[,]?[ ]?\d[.]?[ ]?[a-z][\.]?[ ]?[a-z][\.]?)|( cb\d\d[a-z]{2})')
        self.phone_matcher = re.compile(r'[ (](#?0)?(\d{10}|\d{4}[ ]\d{5,6}|\d{3}-\d{3}-\d{4})[ ).,]')
        self.street_matcher = re.compile(
            r' (([a-z]+)?\d{1,3}([ ]?-[ ]?\d+)? )?[a-z]+ (street|road|avenue)(, (city [a-z]+))?')

    def _inverse_dict(self, d, fmt="%s ", func=str):
        """
        将字典中key和value转换工具
        """
        inv = {}
        for k, vs in d.items():
            for v in vs:
                inv[v.lower()] = fmt % (func(k))
        return inv

    def delex(self, sent):
        """
        将句子去词化
        """
        sent = ' ' + sent.lower()
        sent = self.post_matcher.sub(' <v.POSTCODE> ', sent)
        sent = " , ".join(sent.split(","))

        # for r, v in self.replaces:
        #     sent = sent.replace(" " + r + " ", " " + v + " ")

        sent = sent.replace('  ', ' ')

        sent = self.phone_matcher.sub(' <v.PHONE> ', sent)
        for v in sorted(self.inv_values.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_values[v])

        sent = self.street_matcher.sub(' <v.ADDRESS> ', sent)
        for v in sorted(self.inv_semi_dict.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_semi_dict[v])

        sent = self.num_matcher.sub(' <COUNT> ', sent)

        sent = sent.replace('  ', ' ')

        return sent.strip()


def create_delexicaliser(semi_dict_fn, kb_fn, onto_fn, req_slots=["address", "phone", "postcode", "name"]):
    """
    去词化器创建工具
    """
    semi_dict = defaultdict(list)
    values = defaultdict(list)

    with open(kb_fn) as file:
        kb = json.load(file)

    with open(semi_dict_fn) as file:
        semi_dict = json.load(file)

    with open(onto_fn) as file:
        onto_data = json.load(file)

    for entry in kb:
        for slot in req_slots:
            if slot in entry:
                values[slot].append(entry[slot])

    # slots = ["area", "food", "pricerange", "address", "phone", "postcode", "name"]
    return Delexicalizer(onto_data['informable'], semi_dict, values, '')


def convert_delex(diag_fn, delex_fn, output_fn):
    """
    系统回复槽位生成，将结果保存在一个文件中
    """
    with open(diag_fn) as file:
        dialogues = json.load(file)

    with open(delex_fn) as file:
        delexed = file.readlines()

    delex_iter = iter(delexed)
    for diag_idx, diag in enumerate(dialogues):
        for turn_idx, turn in enumerate(diag['diaglogue']):
            dialogues[diag_idx]['diaglogue'][turn_idx]['system_transcript'] = next(delex_iter).replace("\t", "").strip()

    with open(output_fn, 'w', encoding='utf-8') as file:
        file.write(json.dumps(dialogues, indent=4, ensure_ascii=False))


def preprocess_raw_task_data(raw_data, tokenized_data, semi_dict, database, ontology):
    """
    专门针对task标注数据的client和agent对话的token数据处理
    :param raw_data:  原始对话数据路径
    :param tokenized_data: 生成token数据保存路径
    :return:
    """
    # 首先判断原数据集是否存在，不存在则退出
    if not os.path.exists(raw_data):
        print('数据集不存在，请添加数据集!')
        exit()

    pairs = []
    delex = create_delexicaliser(semi_dict, database, ontology)

    with open(raw_data, encoding='utf-8') as file:
        pair_count = 0
        dialogues = json.load(file)

        for diag in dialogues:
            for turn in diag['dialogue']:
                user = turn['transcript']
                system = delex.delex(turn['system_transcript'])
                pairs.append([user, system])
                pair_count += 1
                if pair_count % 1000 == 0:
                    print('已处理：', pair_count, '个问答对')

    print('读取完毕，处理中...')
    results = []
    for pair in pairs:
        results.append(pair[0] + '\t' + pair[1])

    train_tokenized = open(tokenized_data, 'w', encoding='utf-8')
    for i in range(len(results)):
        train_tokenized.write(results[i] + '\n')
        if i % 1000 == 0:
            print(len(range(len(results))), '处理进度：', i)

    train_tokenized.close()
