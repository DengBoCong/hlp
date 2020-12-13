
def wers(originals, results):
    """多个文本WER计算

    :param originals: 以空格分隔的真实文本串list
    :param results: 以空格分隔的预测文本串list
    :return: WER列表，WER平均值
    """
    count = len(originals)
    assert count > 0
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)

    return rates, mean / float(count)


def wer(original, result):
    """单个WER计算

    :param original: 以空格分隔的真实文本串
    :param result: 以空格分隔的预测文本串
    :return: WER
    """
    original = original.split()
    result = result.split()

    return _levenshtein(original, result) / float(len(original))


def lers(originals, results):
    """多个文本LER计算

    :param originals: 以空格分隔的真实文本串list
    :param results: 以空格分隔的预测文本串list
    :return: 多个ler指标组成的list; ler均值; 规范化ler指标组成的list; 规范化ler均值
    """
    count = len(originals)
    assert count > 0
    assert count == len(results)

    rates = []
    norm_rates = []
    mean = 0.0
    norm_mean = 0.0

    for i in range(count):
        rate = _levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean / float(count))


def ler(original, result):
    """LER

    :param original: 以空格分隔的真实文本串
    :param result: 以空格分隔的预测文本串
    :return: LER
    """
    return  _levenshtein(original, result) / float(len(original))


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
    ref1 = "i like u"
    print(wer(ref1, pred1))
    print(ler(ref1,pred1))
