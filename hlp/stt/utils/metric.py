
def wers(truths, preds):
    """多个文本WER计算

    :param truths: 以空格分隔的真实文本串list
    :param preds: 以空格分隔的预测文本串list
    :return: WER列表，WER平均值
    """
    count = len(truths)
    assert count > 0
    rates = []
    mean = 0.0
    assert count == len(preds)
    for i in range(count):
        rate = wer(truths[i], preds[i])
        mean = mean + rate
        rates.append(rate)

    return rates, mean / float(count)


def wer(truth, pred):
    """单个WER计算

    :param truth: 以空格分隔的真实文本串
    :param pred: 以空格分隔的预测文本串
    :return: WER
    """
    truth = truth.split()
    pred = pred.split()

    return _levenshtein(truth, pred) / float(len(truth))


def lers(truths, preds):
    """多个文本LER计算

    :param truths: 以空格分隔的真实文本串list
    :param preds: 以空格分隔的预测文本串list
    :return: 多个ler指标组成的list; ler均值; 规范化ler指标组成的list; 规范化ler均值
    """
    count = len(truths)
    assert count > 0
    assert count == len(preds)

    rates = []
    norm_rates = []
    mean = 0.0
    norm_mean = 0.0

    for i in range(count):
        rate = _levenshtein(truths[i], preds[i])
        mean = mean + rate

        normrate = (float(rate) / len(truths[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean / float(count))


def ler(truth, pred):
    """LER

    :param truth: 以空格分隔的真实文本串
    :param pred: 以空格分隔的预测文本串
    :return: LER
    """
    return _levenshtein(truth, pred) / float(len(truth))


def _levenshtein(a, b):
    """计算a和b之间的Levenshtein距离

    :param a: 原始文本
    :param b: 预测文本
    :return: a和b之间的Levenshtein距离
    """
    n, m = len(a), len(b)
    if n > m:
        # 确保n <= m, 使用O(min(n, m))
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


if __name__ == "__main__":
    pred1 = "i like you"
    truth1 = "i like u"
    print(wer(truth1, pred1))
    print(ler(truth1, pred1))
