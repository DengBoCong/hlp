import tensorflow as tf

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
    :return: 规范化ler指标组成的list; 规范化ler均值
    """
    count = len(truths)
    assert count > 0
    assert count == len(preds)

    norm_rates = []
    norm_mean = 0.0

    for i in range(count):
        rate = _levenshtein(truths[i], preds[i])

        normrate = (float(rate) / len(truths[i]))
        norm_mean = norm_mean + normrate
        norm_rates.append(round(normrate, 4))

    return norm_rates, (norm_mean / float(count))


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

# def compute_metric(model, val_data_generator, val_batchs, val_batch_size):
#     dataset_information = config.get_dataset_info()
#     units = config.units
#     dec_units = config.dec_units
#     # 确定使用的model类型
#     model_type = config.model_type
#     word_index = dataset_information["word_index"]
#     index_word = dataset_information["index_word"]
#     max_label_length = dataset_information["max_label_length"]
#     results = []
#     labels_list = []
#     for batch, (input_tensor, _, text_list) in zip(range(1, val_batchs + 1), val_data_generator):
#         if model_type == "las_d_w":
#             hidden = tf.zeros((val_batch_size, dec_units))
#         elif model_type == "las":
#             hidden = tf.zeros((val_batch_size, units))
#         dec_input = tf.expand_dims([word_index['<start>']] * val_batch_size, 1)
#         result = ''  # 识别结果字符串
#
#         for t in range(max_label_length):
#             predictions, _ = model(input_tensor, hidden, dec_input)
#             predicted_ids = tf.argmax(predictions, 1).numpy()  # 贪婪解码，取最大
#             idx = str(predicted_ids[0])
#             if index_word[idx] == '<end>':
#                 break
#             else:
#                 result += index_word[idx]
#             dec_input = tf.expand_dims(predicted_ids, 1)
#
#         results.append(result)
#         labels_list.append(text_list[0])
#     norm_rates_lers, norm_aver_lers = lers(labels_list, results)
#
#     return norm_rates_lers, norm_aver_lers

if __name__ == "__main__":
    pred1 = "i like you"
    truth1 = "i like u"
    print(wer(truth1, pred1))
    print(ler(truth1, pred1))
