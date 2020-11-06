import tensorflow as tf
from prepocesses import dataset_txt,dataset_sentence
from config2 import Tacotron2Config
from tacotron2.tacotron2 import Tacotron2
from tacotron2.griffinlim import melspectrogram2wav,plot_spectrogram_to_numpy
import scipy.io.wavfile as wave
import os
import matplotlib.pyplot as plt
from playsound import playsound
#恢复检查点
def load_checkpoint(tacotron2, path):
    # 加载检查点
    checkpoint_path = path
    ckpt = tf.train.Checkpoint(tacotron2=tacotron2)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        #ckpt.restore('./checkpoints/train/ckpt-2')

if __name__=="__main__":
    config = Tacotron2Config()
    path = config.checkpoingt_dir
    #max_len = config.max_len
    vocab_inp_size = 55
    #模型初始化
    tacotron2 = Tacotron2(vocab_inp_size, config)
    #加载检查点
    load_checkpoint(tacotron2, path)
    print('已恢复至最新的检查点！')
    # 设置文件路径
    text_path1 = config.text_path
    # 抓取文本数据
    print("请输入需要合成的话：")
    seq = input()
    dataset_sentence(seq, text_path1)
    input_ids, en_tokenizer = dataset_txt(text_path1)
    print("input_ids:", input_ids.shape)
    input_ids = tf.convert_to_tensor(input_ids)
    os.remove(text_path1)
    #预测
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(input_ids)
    print("mel_outputs1:", mel_outputs_postnet.shape)
    #生成预测声音
    wav = melspectrogram2wav(mel_outputs_postnet[0].numpy())
    sr = 22050
    wave.write('predict.wav', rate=sr, data=wav)
    playsound('.\predict.wav')
    #画图
    plt.figure()
    mel_gts = tf.transpose(mel_outputs_postnet, [0, 2, 1])
    plt.imshow(plot_spectrogram_to_numpy(mel_gts[0].numpy()))
    plt.show()
