import os
import wave
import time
import pyaudio
import tensorflow as tf
import matplotlib.pyplot as plt


def record(record_path, record_duration):
    """从麦克风录音声音道文件

    :param record_path: 声音文件保存路径
    :param record_duration: 录制时间，秒
    :return: 无
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = record_duration
    WAVE_OUTPUT_FILENAME = record_path
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    print("开始录音：请在%d秒内输入语音:" % RECORD_SECONDS)
    for i in range(1, int(RATE / CHUNK * RECORD_SECONDS) + 1):
        data = stream.read(CHUNK)
        frames.append(data)
        if (i % (RATE / CHUNK)) == 0:
            print('\r%s%d%s' % ("剩余", int(RECORD_SECONDS - (i // (RATE / CHUNK))), "秒"), end="")
    print("\r录音结束\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def plot_history(history, valid_epoch_freq, history_img_path):
    """
    绘制各种指标数据
    :param history: 历史指标数据
    :param valid_epoch_freq: 验证频率
    :param history_img_path: 历史指标显示图片保存位置
    :return: 无返回值
    """
    plt.subplot(2, 1, 1)
    epoch1 = [i for i in range(1, 1 + len(history["loss"]))]
    epoch2 = [i * valid_epoch_freq for i in range(1, 1 + len(history["wers"]))]

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch1, history["loss"], "--*b")
    plt.xticks(epoch1)

    # 绘制metric(valid_loss、wers、norm_lers)
    plt.subplot(2, 1, 2)
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.plot(epoch2, history["wers"], "--*r", label="wers")
    plt.plot(epoch2, history["norm_lers"], "--*y", label="norm_lers")
    plt.xticks(epoch2)

    plt.legend()
    if not os.path.exists(history_img_path):
        os.makedirs(history_img_path, exist_ok=True)
    plt.savefig(history_img_path + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time())))
    plt.show()


def wers(truths, preds):
    """
    多个文本WER计算
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
    """
    单个WER计算
    :param truth: 以空格分隔的真实文本串
    :param pred: 以空格分隔的预测文本串
    :return: WER
    """
    truth = truth.split()
    pred = pred.split()

    return _levenshtein(truth, pred) / float(len(truth))


def lers(truths, preds):
    """
    多个文本LER计算
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
    """
    LER
    :param truth: 以空格分隔的真实文本串
    :param pred: 以空格分隔的预测文本串
    :return: LER
    """
    return _levenshtein(truth, pred) / float(len(truth))


def _levenshtein(a, b):
    """
    计算a和b之间的Levenshtein距离
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


def load_checkpoint(model: tf.keras.Model, checkpoint_dir: str,
                    execute_type: str, checkpoint_save_size: int):
    """
    恢复检查点
    :param model: 传入的模型
    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if execute_type == "recognize":
            print("没有检查点，请先执行train模式")
            exit(0)

    return ckpt_manager


def compute_ctc_input_length(max_time_steps, ctc_time_steps, input_length):
    """
    计算ctc api中的参数input_length，基于https://github.com/tensorflow/models/blob/master/research/deep_speech
    将ctc相关的api函数(ctc_loss,ctc_decode)所需要的参数input_length进行一个按比例缩减
    这个比例是input_tensor的max_timestep:模型输出outputs的time_step
    :param max_time_steps: 最大时间维度大小
    :param ctc_time_steps: ctc计算的时间维度大小
    :param input_length: 音频特征实际时间维度大小
    :return: 计算的ctc长度
    """
    ctc_input_length = tf.cast(tf.multiply(input_length, ctc_time_steps), dtype=tf.float32)
    return tf.cast(tf.math.floordiv(ctc_input_length, tf.cast(max_time_steps,
                                                              dtype=tf.float32)), dtype=tf.int32)


def can_stop(numbers):
    last = numbers[-1]
    rest = numbers[:-1]
    # 最后一个错误率比所有的大则返回True
    if all(i <= last for i in rest):
        return True
    else:
        return False


if __name__ == "__main__":
    pred1 = "i like you"
    truth1 = "i like u"
    print(wer(truth1, pred1))
    print(ler(truth1, pred1))
