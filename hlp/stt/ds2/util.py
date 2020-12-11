import json

import tensorflow as tf

from hlp.stt.utils.metric import wers, lers
from hlp.stt.utils.text_process import int_to_text_sequence


# 获取配置文件
def get_config():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs


# 获取预处理得到的语料集信息
def get_dataset_info(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset_information = json.load(f)
    return dataset_information


# 计算ctc api中的参数input_length，基于https://github.com/tensorflow/models/blob/master/research/deep_speech
# 将ctc相关的api函数(ctc_loss,ctc_decode)所需要的参数input_length进行一个按比例缩减
# 这个比例是input_tensor的max_timestep:模型输出outputs的time_step
def compute_ctc_input_length(max_time_steps, ctc_time_steps, input_length):
    ctc_input_length = tf.cast(tf.multiply(input_length, ctc_time_steps),
                               dtype=tf.float32)
    return tf.cast(
        tf.math.floordiv(ctc_input_length, tf.cast(max_time_steps, dtype=tf.float32)),
        dtype=tf.int32)


# 在valid或test计算指标
def compute_metric(model, test_data_generator, batches, text_process_mode, index_word):
    aver_wers = 0
    aver_lers = 0
    aver_norm_lers = 0

    for batch, (input_tensor, input_length, text_list) in zip(range(1, batches + 1), test_data_generator):
        originals = text_list
        results = []
        y_pred = model(input_tensor)
        ctc_input_length = compute_ctc_input_length(input_tensor.shape[1], y_pred.shape[1], input_length)
        output = tf.keras.backend.ctc_decode(y_pred=y_pred,
                                             input_length=tf.reshape(ctc_input_length, [ctc_input_length.shape[0]]),
                                             # input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
                                             greedy=True)
        results_int_list = output[0][0].numpy().tolist()

        # 解码
        for i in range(len(results_int_list)):
            tokens = int_to_text_sequence(results_int_list[i], index_word, text_process_mode).strip()
            results.append(tokens)

        # 通过wer、ler指标评价模型
        _, aver_wer = wers(originals, results)
        _, aver_ler, _, norm_aver_ler = lers(originals, results)

        aver_wers += aver_wer
        aver_lers += aver_ler
        aver_norm_lers += norm_aver_ler

    return aver_wers / batches, aver_lers / batches, aver_norm_lers / batches


def can_stop(numbers):
    last = numbers[-1]
    rest = numbers[:-1]
    # 最后一个错误率比所有的大则返回True
    if all(i <= last for i in rest):
        return True
    else:
        return False


if __name__ == "__main__":
    pass
