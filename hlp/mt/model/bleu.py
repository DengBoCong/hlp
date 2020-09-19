import numpy as np
import re
import math
from common import preprocess
from model import translator
import config.get_config as _config

"""
Description:
    BLEU
"""


def calculate_average(precisions, weights):
    """Calculate the geometric weighted mean."""
    tmp_res = 0
    for id, item in enumerate(precisions):
        if item == 0:
            continue
        else:
            tmp_res += weights[id] * math.log(item)
    if tmp_res == 0:
        return tmp_res
    else:
        return math.exp(tmp_res)



def calculate_candidate(gram_list, candidate):
    """Calculate the count of gram_list in candidate."""
    gram_sub_str = ' '.join(gram_list)
    return len(re.findall(gram_sub_str, candidate))


def calculate_reference(gram_list, references):
    """Calculate the count of gram_list in references"""
    gram_sub_str = ' '.join(gram_list)
    gram_count = []
    for item in references:
        # calculate the count of the sub string
        gram_count.append(len(re.findall(gram_sub_str, item)))
    return gram_count  # 返回列表，为每个 n-gram 在参考句子中数量


def sentence_bleu(candidate_sentence, reference_sentences, max_gram = 4, weights=(0.25, 0.25, 0.25, 0.25), ch = False):
    """
    :param candidate_sentence:机翻句子
    :param reference_sentence:参考句子列表
    :param max_gram:计算至max_gram的N-gram，默认为 4
    :param weights:各N-gram的权重，默认为 (0.25, 0.25, 0.25, 0.25)
    :param ch:输入是否为中文句子。若为TRUE，则函数会进行预处理
    :description: 此BLUE为改良的BLEU,采用了截断、加权平均及短句惩罚因子
    :return:精度

    注意：输入的句子的单位之间需要用' '分割
    """
    if ch:
        candidate_sentence = ' '.join(candidate_sentence)
        reference_sentences = [' '.join(s) for s in reference_sentences]

    candidate_corpus = list(candidate_sentence.split(' '))
    # number of the reference sentences
    refer_len = len(reference_sentences)
    candidate_tokens_len = len(candidate_corpus)
    # 首先需要计算各种长度的gram 的precision值

    # 计算当前gram 在candiate_sentence中出现的次数 同时计算这个gram 在所有的reference sentence中的出现的次数
    # 每一次计算时将当前candidate_sentence中当前gram出现次数与在当前reference sentence中出现的gram次数选择最小值
    # 作为这个gram相对于 参考文献j的截断次数
    # 然后将所有的参考文献对应的截断次数做最大值 作为这个gram在整个参考文献上的综合截断值 这个值就是当前gram对应的分子
    # 分母依然是这个gram 在candidate sentence中出现的次数
    # 在计算当前长度(n)的其他的gram的综合截断次数 然后加起来作为长度为n的gram的综合截断次数值 分母是所有长度为n的gram的相加的值
    # 两个值相除即可得到这个长度为n的gram 的precision值
    gram_precisions= []
    for i in range(max_gram):
        # calculate each gram precision
        # set current gram length
        curr_gram_len = i+1
        # calculate current gram length mole(分子)
        curr_gram_mole = 0
        # calculate current gram length deno(分母)
        curr_gram_deno = 0
        for j in range(0, candidate_tokens_len):
            if j + curr_gram_len > candidate_tokens_len:  # 判断是否是最后一个 n-gram
                continue
            else:  # curr_gram_list 为机翻的第j个 n-gram 列表
                curr_gram_list = candidate_corpus[j:j+curr_gram_len]
                # print('curr_gram_list:' + ' '.join(curr_gram_list))
                gram_candidate_count = calculate_candidate(curr_gram_list, candidate_sentence)  #
                # print(' current gram candidate count')
                # print(gram_candidate_count)
                gram_reference_count_list = calculate_reference(curr_gram_list, reference_sentences)  # gram_reference_count_list 为计算 n-gram 在参考句子中数量的列表
                # print(' current gram reference count list')
                # print(gram_reference_count_list)
                truncation_list = []
                for item in gram_reference_count_list:
                    truncation_list.append(np.min([gram_candidate_count, item]))  # 在截断列表中添加该n-gram在机翻与各个参考句子中最小次数
                curr_gram_mole += np.max(truncation_list)  # 将该n-gram的截断count加入分子
                curr_gram_deno += gram_candidate_count  # 将该n-gram在机翻句子中数量加入分母
        print(' current length %d and gram mole %d and deno %d' % (i+1, curr_gram_mole, curr_gram_deno))
        gram_precisions.append(curr_gram_mole/curr_gram_deno * 100)  # 将该阶n-gram的precisions加入列表gram_precisions
    # 此处得到的gram_precisions为 1 ~ N 的gram的 precision 的列表
    print('all the precisions about the grams')
    print(gram_precisions)


    # 其次对多元组合(n-gram)的precision 进行加权取平均作为最终的bleu评估指标
    # 一般选择的做法是计算几何加权平均 exp(sum(w*logP))
    average_res = calculate_average(gram_precisions, weights)
    # print(' current average result：')
    # print(average_res)

    # 最后引入短句惩罚项 避免短句翻译结果取得较高的bleu值, 影响到整体评估
    # 涉及到最佳的匹配长度 当翻译的句子的词数量与任意的参考翻译句子词数量一样的时候 此时无需惩罚项
    # 如果不相等 那么需要设置一个参考长度r 当翻译的句子长度(c) 大于 r 的时候不需要进行惩罚 而 当c小于r
    # 需要在加权平均值前乘以一个惩罚项exp(1-r/c) 作为最后的bleu 指标输出
    # r 的选择可以这样确定 当翻译句子长度等于任何一个参考句子长度时不进行惩罚 但是当都不等于参考句子长度时
    # 可以选择参考句子中最长的句子作为r 当翻译句子比r 长时不进行惩罚 小于r时进行惩罚
    bp = 1
    reference_len_list = [len(item.split(' ')) for item in reference_sentences]
    if candidate_tokens_len in reference_len_list:
        bp = 1
    else:
        if candidate_tokens_len < np.max(reference_len_list):
            bp = np.exp(1-(np.max(reference_len_list)/candidate_tokens_len))
    print('bp:%f' % bp)
    return bp * average_res

def calc_bleu():

    # 导入评估文本计算BLEU
    path_to_eval_file = _config.path_to_eval_file  # 评估文本路径
    num_eval = _config.num_eval  # 用来评估的句子数量
    en, ch = preprocess.create_eval_dataset(path_to_eval_file, num_eval)
    print('开始计算BLEU指标...')
    bleu_sum = 0
    for i in range(num_eval):
        candidate_sentence = translator.translate(en[i])
        print('-' * 20)
        print('第%d个句子：' % (i + 1))
        print('Input:' + en[i])
        print('Translate:' + candidate_sentence)
        print('Reference:' + ch[i])
        bleu_i = sentence_bleu(candidate_sentence, [ch[i]], ch=True)
        print('此句子BLEU:%.2f' % bleu_i)
        bleu_sum += bleu_i
    bleu = bleu_sum / num_eval
    print('-' * 20)
    print('平均BLEU指标为：%.2f' % bleu)
