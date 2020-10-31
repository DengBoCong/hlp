import tensorflow as tf
from prepocesses import dataset_txt,dataset_sentence,play
from config2 import Tacotron2Config
from model.tacotron2 import Tacotron2
from model.griffinlim import melspectrogram2wav,plot_spectrogram_to_numpy
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
from playsound import playsound
#恢复检查点
def load_checkpoint(tacotron2,path):
    # 加载检查点
    checkpoint_path = path
    ckpt = tf.train.Checkpoint(tacotron2=tacotron2)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        #ckpt.restore('./checkpoints/train/ckpt-2')
        print('已恢复至最新的检查点！')

if __name__=="__main__":
    path = './training_checkpoints2'
    config = Tacotron2Config()
    max_len = config.max_len
    vocab_inp_size = config.vocab_size
    #模型初始化
    tacotron2 = Tacotron2(vocab_inp_size, config)
    #加载检查点
    load_checkpoint(tacotron2,path)
    # 设置文件路径
    text_test_path = config.text_test_path

    # 抓取文本数据
    print("请输入需要合成的话：")
    seq = input()
    dataset_sentence(seq,text_test_path)
    #读入文本数据
    input_ids, en_tokenizer = dataset_txt(text_test_path)
    input_ids = tf.convert_to_tensor(input_ids)
    print("文本：",input_ids.shape)
    print("文字：",en_tokenizer.index_word)

    #预测
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(input_ids)
    #mel_outputs_postnet = tf.cast(mel_outputs_postnet,dtype = tf.float64)
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
