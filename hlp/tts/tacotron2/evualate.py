import os
import time
import tensorflow as tf
from prepocesses import Dataset_txt,Dataset_wave,create_dataset
from config2 import Tacotron2Config
from model.tacotron2 import Tacotron2
from griffinlim.griffinlim import melspectrogram2wav,plot_spectrogram_to_numpy
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
#恢复检查点
def load_checkpoint(tacotron2,path):
    # 加载检查点
    checkpoint_path = path
    ckpt = tf.train.Checkpoint(tacotron2=tacotron2)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        #ckpt.restore('./checkpoints/train/ckpt-2')
        print('已恢复至最新的检查点！')




#画频谱图
if __name__=="__main__":
    path='./training_checkpoints2'
    config=Tacotron2Config()
    vocab_inp_size = config.vocab_size
    tacotron2 = Tacotron2(vocab_inp_size, config)
    load_checkpoint(tacotron2,path)
    # 设置文件路径
    text_test_path = config.text_test_path
    # 取数据
    input_ids, en_tokenizer = Dataset_txt(text_test_path)
    input_ids = tf.convert_to_tensor(input_ids)
    #预测
    mel_gts=tacotron2.inference(input_ids)
    #声码器
    wav = melspectrogram2wav(mel_gts.numpy())
    sr = 22050
    wave.write('预测的.wav', rate=sr, data=wav)
    #画图
    plt.figure()
    mel_gts = tf.transpose(mel_gts, [0, 2, 1])
    plt.imshow(plot_spectrogram_to_numpy(mel_gts.numpy()))
    plt.show()






    exit(0)





    config = Tacotron2Config()
    batch_size=config.batch_size
    #设置文件路径
    text_train_path=config.text_train_path
    wave_train_path = config.wave_train_path
    #取数据
    input_ids,en_tokenizer=Dataset_txt(text_train_path)
    input_ids=tf.convert_to_tensor(input_ids)
    print("input_ids:",input_ids.shape)
    mel_gts=Dataset_wave(wave_train_path)
    mel_gts=tf.transpose(mel_gts,[0,2,1])
    print("mel_gts:", mel_gts.shape)
    print(mel_gts[1])




    print("mel_gts:", mel_gts.shape)
    #建立输入输出流
    dataset,steps_per_epoch=create_dataset(batch_size, input_ids, mel_gts)
    # 初始化模型
    vocab_inp_size=config.vocab_size
    tacotron2 = Tacotron2(vocab_inp_size, config)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    epochs=1000
    #检查点
    checkpoint_dir = './training_checkpoints2'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(tacotron2=tacotron2)

    #训练
    mel_outputs=train(tacotron2, optimizer, dataset, epochs, steps_per_epoch)
    #mel_outputs=tf.transpose(mel_outputs,[0,2,1])
    print("mel_outputs:",mel_outputs.shape)
    wav=melspectrogram2wav(mel_outputs[1].numpy())
    sr=22050
    wave.write('4.wav', rate=sr, data=wav)
    #print(mel_outputs[1])
    plt.figure()
    mel_gts=tf.transpose(mel_gts,[0,2,1])
    plt.imshow(plot_spectrogram_to_numpy(mel_gts[1].numpy()))
    print("mel_gts[1]",mel_gts[1])
    plt.show()


    plt.imshow(plot_spectrogram_to_numpy(mel_outputs[1].numpy()))
    print("mel_outputs[1]", mel_gts[1])
    plt.show()
    print(4)

    # # 检查点
    #
    # checkpoint_dir = './training_checkpoints2'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(tacotron2=tacotron2)
    # manager = tf.train.CheckpointManager(
    #     checkpoint,
    #     directory='./training_checkpoints2',
    #     max_to_keep=5
    #     )
    # manager.save()


