

# originals是原始语料中的文本串list，results是模型解码后的文本串list
# 进行wer指标的计算
def wers(originals, results):
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise ("error, assert count>0 - 可能数据丢失")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = _wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    
    return rates, mean / float(count)

# 单个原始语料文本和模型解码文本的wer指标计算
def _wer(original, result):
    """
    WER的定义是在单词水平上编辑/Levenshtein距离除以原文中的单词量。
    如果原稿的字数（N）比结果多，而且两者完全不同（N个字都会导致一次编辑操作），则WER始终为1（N/N=1）
    
    WER是在单词（而不是字符）级别上计算的。
    因此，我们首先将字符串分成单词：
    """
    original = original.split()
    result = result.split()

    return _levenshtein(original, result) / float(len(original))

# originals是原始语料中的文本串list，results是模型解码后的文本串list
# 进行ler指标的计算
def lers(originals, results):
    count = len(originals)
    assert count > 0
    rates = []
    norm_rates = []

    mean = 0.0
    norm_mean = 0.0

    assert count == len(results)
    for i in range(count):
        rate = _levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean / float(count))

# 计算a和b之间的Levenshtein距离
def _levenshtein(a, b):
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
