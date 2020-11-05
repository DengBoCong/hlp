import tensorflow as tf
from prepocesses import dataset_txt,dataset_wave
import numpy as np
from config2 import Tacotron2Config
from model.tacotron2 import Tacotron2
from predict import load_checkpoint
#计算mel谱之间的欧式距离
def compute_distence(mel1,mel2):
    mel1 = tf.transpose(mel1, [0, 2, 1])
    score = np.sqrt(np.sum((mel1 - mel2)**2))
    print(score)
    return score

def evluate(path1,path2):
    #读取测试集的文本数据
    input_ids,vocab_inp_size = dataset_txt(path1)
    input_ids = tf.convert_to_tensor(input_ids)
    #读取测试机的音频数据
    mel_gts, mel_len_wav = dataset_wave(path2, config)
    vocab_inp_size = 55
    # 模型初始化
    tacotron2 = Tacotron2(vocab_inp_size, config)
    path = './training_checkpoints2'
    load_checkpoint(tacotron2,path)
    print('已恢复至最新的检查点！')
    for i in range(input_ids.shape[0]):
        new_input_ids = input_ids[i]
        new_input_ids = tf.expand_dims(new_input_ids, axis=0)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(new_input_ids)
        mel2 = mel_gts[i]
        mel2 = tf.expand_dims(mel2, axis=0)
        print("欧式距离为：")
        compute_distence(mel_outputs_postnet, mel2)

if __name__=="__main__":
    config = Tacotron2Config()
    path1 = config.text_test_path
    path2 = config.wave_test_path
    evluate(path1, path2)




