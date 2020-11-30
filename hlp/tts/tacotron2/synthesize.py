import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
from playsound import playsound

from .audio_process import melspectrogram2wav
from .config2 import Tacotron2Config
from .prepocesses import get_tokenizer_keras, dataset_seq
from .tacotron2 import Tacotron2, load_checkpoint


# 下面两个方法没使用，暂时保留
def _plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = _save_figure_to_numpy(fig)
    plt.close()
    return data


def _save_figure_to_numpy(fig):
    # 保存成numpy
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


if __name__ == "__main__":
    config = Tacotron2Config()
    # 检查点路径
    path = config.checkpoingt_dir

    # 字典路径
    save_path_dictionary = config.save_path_dictionary

    # 恢复字典
    tokenizer, vocab_size = get_tokenizer_keras(save_path_dictionary)

    # 模型初始化
    tacotron2 = Tacotron2(vocab_size, config)

    # 加载检查点
    checkpoint = load_checkpoint(tacotron2, path, config)
    print('已恢复至最新的检查点！')

    # 用于命名
    i = 0
    # 抓取文本数据
    while True:
        i = i + 1
        b = str(i)
        print("请输入您要合成的话，输入ESC结束：")
        seq = input()
        if seq == 'ESC':
            break
        sequences_list = []
        sequences_list.append(seq)
        input_ids = dataset_seq(sequences_list, tokenizer, config)
        input_ids = tf.convert_to_tensor(input_ids)
        print(input_ids)
        # 预测
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(input_ids)

        # 生成预测声音
        wav = melspectrogram2wav(mel_outputs_postnet[0].numpy(), config.max_db, config.ref_db, config.sr, config.n_fft,
                                 config.n_mels, config.preemphasis, config.n_iter, config.hop_length, config.win_length)
        name = 'generated' + b + '.wav'
        wave.write(name, rate=config.sr, data=wav)
        playsound(name)
        print("已合成")
    print("合成结束")
