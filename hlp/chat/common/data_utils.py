import io
import os
import json
import random
import tensorflow as tf
from pathlib import Path
from model.task.kb import load_kb
from collections import defaultdict
import config.get_config as _config
from nltk import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer


def preprocess_sentence(w):
    """
    用于给句子首尾添加start和end
    :param w:
    :return: 合成之后的句子
    """
    w = 'start ' + w + ' end'
    return w


def create_dataset(path, num_examples):
    """
    用于将分词文本读入内存，并整理成问答对
    :param path:
    :param num_examples:
    :return: 整理好的问答对
    """
    is_exist = Path(path)
    if not is_exist.exists():
        file = open(path, 'w', encoding='utf-8')
        file.write('吃饭 了 吗' + '\t' + '吃 了')
        file.close()
    size = os.path.getsize(path)
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    """
    :param tensor:
    :return: 列表中最大的长度
    """
    return max(len(t) for t in tensor)


def read_data(path, num_examples):
    """
    读取数据，将input和target进行分词后返回
    :param path:
    :param num_examples:
    :return: input_tensor, input_token, target_tensor, target_token
    """
    input_lang, target_lang = create_dataset(path, num_examples)
    input_tensor, input_token = tokenize(input_lang)
    target_tensor, target_token = tokenize(target_lang)
    return input_tensor, input_token, target_tensor, target_token


def pad_sequence(seqs):
    """
    填充序列，0
    :param seqs: 序列
    :return: 返回填充好的序列
    """
    max_len = max([len(seq) for seq in seqs])
    padded = [seq + [0] * (max_len - len(seq)) for seq in seqs]
    return padded


def tokenize(lang):
    """
    分词方法，使用Keras API中的Tokenizer进行分词操作
    :param lang:
    :return: tensor, lang_tokenizer
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=_config.max_length_inp, padding='post')

    return tensor, lang_tokenizer


def tokenize_en(sent, tokenizer):
    """
    用来针对英文句子的分词
    :param sent: 句子
    :param tokenizer: 正则表达式分词器
    :return: 分好的句子
    """
    tokens = tokenizer.tokenize(sent)
    ret = []
    for t in tokens:
        # 这里要注意，如果是槽位，要直接作为一个token放进去，例如<v.pricerange>
        if '<' not in t:
            ret.extend(wordpunct_tokenize(t))
        else:
            ret.append(t)
    return ret


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


def load_dataset():
    """
    数据加载方法，含四个元素的元组，包括如下：
    :return:input_tensor, input_token, target_tensor, target_token
    """
    return read_data(_config.data, _config.max_train_data_size)


def load_dialogs(diag_fn, kb, groups_fn=None):
    """
    加载数据集中的对话，按照格式整理好并返回
    :param diag_fn: 数据集文件路径
    :param kb: knowledge base的词表
    :param groups_fn: 语句槽位集合文件路径
    :return: 整理好的数据
    """
    with open(diag_fn) as file:
        dialogues = json.load(file)

    group_iter = None
    if groups_fn is not None:
        with open(groups_fn) as file:
            groups = json.load(file)["labels"]
        group_iter = iter(groups)

    data = []
    for dialogue in dialogues:
        usr_utterances = []
        sys_utterances = []
        states = []
        kb_found = []
        sys_utterance_groups = []

        for turn in dialogue['dialogue']:
            usr_utterances.append(turn['transcript'])
            sys_utterances.append('<sos> ' + turn['system_transcript'] + '<eos>')
            slots = []
            search_keys = []

            for state in turn['belief_state']:
                if state['act'] == 'inform':
                    slots.append(state['slots'][0])
                    state['slots'][0][0] = state['slots'][0][0].replace(' ', '').replace('center', 'centre')
                    search_keys.append(state['slots'][0])
                elif state['act'] == 'request':
                    slots.append((state['slots'][0][1].replace(' ', '') + '_req', 'care'))
                else:
                    raise RuntimeError('illegal state : %s' % (state,))

            states.append(slots)
            ret = kb.search_multi(search_keys)
            kb_found.append(len(ret))

        # 这里就跳过第一个，因为一般系统第一个是空
        sys_utterances = sys_utterances[1:]
        if group_iter is not None:
            for _ in sys_utterances:
                group = next(group_iter)
                sys_utterance_groups.append(group)

        data.append({
            'usr_utterances': usr_utterances,
            'sys_utterances': sys_utterances,
            'sys_utterance_groups': sys_utterance_groups,
            'states': states,
            'kb_found': kb_found,
        })
    return data


def load_ontology(fn):
    """
    加载对话数据集中的本体
    :param fn:本体数据集的文件路径
    :return:返回整理好的本体和本体索引
    """
    with open(fn) as file:
        data = json.load(file)

    onto = {}
    onto_idx = defaultdict(dict)
    # 这里获取用户告知系统的信息
    inform_data = data['informable']

    for key, values in inform_data.iter():
        onto[key] = values + ['dontcare']
        onto_idx[key]['dontcare'] = 0
        for value in values:
            onto_idx[key][value] = len(onto_idx[key])

        key = key + '_req'
        onto[key] = values + ['dontcare']
        onto_idx[key] = {
            'dontcare': 0,
            'care': 1,
        }

    req_data = data['requestable']
    for key, values in req_data.iter():
        key = key + '_req'
        onto[key] = values + ['dontcare']
        onto_idx[key] = {
            'dontcare': 0,
            'care': 1,
        }

    return onto, onto_idx


class DataLoader:
    """
    对话数据加载工具类
    """
    def __init__(self, dialogues, word2idx, sys_word2idx, onto, onto_idx, kb_fonud_len=5, mode='train'):
        self.dialogues = dialogues
        self.word2idx = word2idx
        self.sys_word2idx = sys_word2idx
        self.cur = 0
        self.onto = onto
        self.onto_idx = onto_idx
        self.kb_found_len = kb_fonud_len
        self.kb_indicator = tf.cast(tf.eye(kb_fonud_len + 2), dtype=tf.int64)
        self.mode = mode

    def get_vocabs(self):
        """
        获取对话数据集中的token集合，分为user和system两个token集合
        :return: user和system两个token集合
        """
        vocabs = []
        sys_vocabs = []
        for dialogue in self.dialogues:
            for s in dialogue['usr_utterances']:
                vocabs.extend(self._sent_normalize(s))
            for s in dialogue['sys_utterances']:
                sys_vocabs.extend(self._sent_normalize(s))
        return set(vocabs), set(sys_vocabs)

    def _sent_normalize(self, sent):
        """
        分词器
        :param sent: 语句
        :return: 语句序列
        """
        tokenizer = RegexpTokenizer(r'<[a-z][.\w]+>|[^<]+')
        return tokenize_en(sent=sent.lower(), tokenizer=tokenizer)

    def _get(self, i):
        """
        获取整理对话数据集中的第i个对话的相关数据，整理
        至对应格式，并统一将数据类型转成tf.int64
        :param i: 第i个对话数据
        :return: 整理好的对话数据
        """
        dialogue = self.dialogues[i]
        usr_utterances = [self._gen_utterance_seq(self.word2idx, s) for s in dialogue['usr_utterances']]
        usr_utterances = tf.convert_to_tensor(pad_sequence(usr_utterances), dtype=tf.int64)
        states = self._gen_state_vectors(dialogue['states'])
        kb_found = tf.concat([tf.reshape(self.kb_indicator[x], [1, -1])
                              if x <= self.kb_found_len else tf.reshape(self.kb_indicator[self.kb_found_len + 1],
                                                                        [1, -1])
                              for x in dialogue['kb_found']])
        sys_utterances = [self._gen_utterance_seq(self.sys_word2idx, s) for s in dialogue['sys_utterances']]
        sys_utterances = [tf.reshape(tf.convert_to_tensor(utt, dtype=tf.int64), [1, -1]) for utt in sys_utterances]
        sys_utterance_groups = tf.convert_to_tensor(dialogue['sys_utterance_groups'], dtype=tf.int64)

        return dialogue['usr_utterances'], dialogue[
            'sys_utterances'], usr_utterances, sys_utterances, states, kb_found, sys_utterance_groups

    def _gen_utterance_seq(self, word2idx, utterance):
        """
        将语句转成token索引向量
        :param word2idx: 索引字典
        :param utterance: 语句
        :return: 返回转换好的向量
        """
        utterance = self._sent_normalize(utterance)
        utterance = [word2idx.get(x, 0) for x in utterance]
        return utterance

    def _gen_state_vectors(self, states):
        """
        将状态序列中槽位值转成Tensor序列
        :param states: 状态列表
        :return: 整理好的状态张量
        """
        state_vectors = {slot: tf.cast(tf.zeros(len(states)), dtype=tf.int64) for slot in self.onto}
        for t, states_at_time_t in enumerate(states):
            for s, v in states_at_time_t:
                state_vectors[s][t] = self.onto_idx[s][v]
        return state_vectors

    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0

    def next(self):
        """
        移动到下一个对话，如果运行到test数据集，直接停止
        :return: 返回对应对话的数据
        """
        ret = self._get(self.cur)
        self.cur += 1
        # 没运行完一个epoch，就直接乱进行下一个epoch
        if self.cur == len(self.dialogues):
            if self.mode == 'test':
                raise StopIteration()
            random.shuffle(self.dialogues)
            self.cur = 0

        return ret

def load_data(kb, ):
    kb = load_kb(kb, 'name')