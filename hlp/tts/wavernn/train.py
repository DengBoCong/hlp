import time
import numpy as np
import torch
import sys
import os
import argparse
import tensorflow as tf
from generator import generator
from hlp.tts.wavernn.module import load_checkpoint
from hlp.tts.wavernn.wavernn import WaveRNN
from hlp.tts.wavernn.module import Discretized_Mix_Logistic_Loss

from preprocess import process_wav_name


import json
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    #parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    #parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    #parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    #parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    #parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')

    parser.add_argument('--sample_rate', default=22050, type=int, required=False, help='采样比率')

    parser.add_argument('--n_fft', default=2048, type=int, required=False, help='FFT窗口大小')

    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')

    parser.add_argument('--num_mels', default=80, type=int, required=False, help='梅尔滤波器的步数')

    parser.add_argument('--hop_length', default=275, type=int, required=False, help='列之间的音频样本数')
    parser.add_argument('--win_length', default=1100, type=int, required=False, help='窗长')
    parser.add_argument('--fmin', default=40, type=int, required=False, help='最低频率')
    parser.add_argument('--bits', default=9, type=int, required=False, help='字节')
    parser.add_argument('--mu_law', default=True, type=bool, required=False, help='')
    parser.add_argument('--peak_norm', default=False, type=bool, required=False, help='')
    parser.add_argument('--min_level_db', default=-100, type=int, required=False, help='')
    parser.add_argument('--ref_level_db', default=20, type=int, required=False, help='')
    parser.add_argument('--voc_mode', default='MOL', type=str, required=False, help='模型方式')
    parser.add_argument('--voc_upsample_factors', default=(5, 5, 11), type=list, required=False, help='')
    parser.add_argument('--voc_rnn_dims', default=512, type=int, required=False, help='模型中rnn的输出单元')
    parser.add_argument('--voc_fc_dims', default=512, type=int, required=False, help='模型中dense层的输出单元')
    parser.add_argument('--voc_compute_dims', default=128, type=int, required=False, help='')
    parser.add_argument('--voc_res_out_dims', default=128, type=int, required=False, help='')
    parser.add_argument('--voc_res_blocks', default=10, type=int, required=False, help='')

    parser.add_argument('--voc_batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--voc_lr', default=1e-4, type=int, required=False, help='学习率')
    parser.add_argument('--voc_checkpoint_every', default=25000, type=int, required=False, help='需要保存检查点的轮数')
    parser.add_argument('--voc_gen_at_checkpoint', default=5, type=int, required=False, help='在每个检查点生成的样本数量')

    parser.add_argument('--voc_total_steps', default=1000000, type=int, required=False, help='训练步数')
    parser.add_argument('--voc_test_samples', default=50, type=int, required=False, help='测试样品数目')
    parser.add_argument('--voc_pad', default=2, type=int, required=False, help='填充')
    parser.add_argument('--voc_clip_grad_norm', default=4, type=int, required=False, help='')
    parser.add_argument('--voc_target', default=11000, type=int, required=False, help='在每个批次条目中产生的目标样品数量')
    parser.add_argument('--voc_overlap', default=550, type=int, required=False, help='批次间交叉衰落的样本数量')



    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--metadata_file', default='\\data\\LJSpeech-1.1\\metadata.csv', type=str, required=False,
                        help='原始语音数据的metadata文件路径')
    parser.add_argument('--wave_train_path', default='\\data\\LJSpeech-1.1\\wavs\\', type=str, required=False,
                        help='训练语音数据的保存目录')
    parser.add_argument('--train_file', default='\\data\\LJSpeech-1.1\\processed_train_file.txt', type=str,
                        required=False,
                        help='整理后的音频句子对保存路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\LJSpeech-1.1\\feature\\', type=str, required=False,
                        help='目标数据存放路径')

    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\tacotron2', type=str, required=False,
                        help='检查点保存路径')
    parser.add_argument('--wave_save_dir', default='\\data\\LJSpeech-1.1\\generate\\wavernn', type=str,
                        required=False,
                        help='合成的音频保存路径')
    parser.add_argument('--voc_seq_len', default='1375', type=int, required=False, help='音频句子长度=5*hop_length')
    parser.add_argument('--preemphasis', default='0.97', type=float, required=False, help='预加重')
    parser.add_argument('--max_db', default='100', type=int, required=False, help='峰值分贝值')
    parser.add_argument('--ref_db', default='20', type=int, required=False, help='参考分贝值')
    parser.add_argument('--top_db', default='15', type=int, required=False, help='峰值以下的阈值分贝值')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    print('\nInitialising Model...\n')
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\wavernn")]
    voc_model = WaveRNN(rnn_dims=options['voc_rnn_dims'],
                    fc_dims=options['voc_fc_dims'],
                    bits=options['bits'],
                    pad=options['voc_pad'],
                    upsample_factors=options['voc_upsample_factors'],
                    feat_dims=options['num_mels'],
                    compute_dims=options['voc_compute_dims'],
                    res_out_dims=options['voc_res_out_dims'],
                    res_blocks=options['voc_res_blocks'],
                    hop_length=options['hop_length'],
                    sample_rate=options['sample_rate'],
                    mode=options['voc_mode'])

    #检查以确保hop_length正确分解
    assert np.cumprod(options['voc_upsample_factors'])[-1] == options['hop_length']

    optimizer = tf.keras.optimizers.Adam(lr=options['voc_lr'])
    ckpt_manager = load_checkpoint(model=voc_model, checkpoint_dir=work_path + options['checkpoint_dir'],
                                   checkpoint_save_size=options['voc_gen_at_checkpoint'])

    wav_name_list = process_wav_name(work_path+options['wave_train_path'])
    print(wav_name_list)
    train_data_generator = generator(wav_name_list, options['voc_batch_size'], options['sample_rate'], options['peak_norm'],
                                     options['voc_mode'], options['bits'], options['mu_law'], work_path + options['wave_train_path'],
                                     options['voc_pad'], options['hop_length'], options['voc_seq_len'], options['preemphasis'],
                                     options['n_fft'], options['num_mels'], options['win_length'], options['max_db']
                                     , options['ref_db'], options['top_db'])

    if os.path.exists(work_path + options['checkpoint_dir']):
        ckpt_manager = load_checkpoint(voc_model, work_path+options['checkpoint_dir'], options['voc_gen_at_checkpoint'])
        print('已恢复至最新的检查点！')
    else:
        checkpoint_prefix = os.path.join(work_path+options['checkpoint_dir'], "ckpt")
        checkpoint = tf.train.Checkpoint(wavernn=voc_model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, work_path+options['checkpoint_dir'], options['voc_gen_at_checkpoint'])
        print('新的检查点已准备创建！')

    voc_train_loop(work_path+options['checkpoint_dir'], voc_model, optimizer, train_data_generator,
                   ckpt_manager, options['voc_total_steps'], options['voc_batch_size']
                   , options['voc_mode'])


def voc_train_loop(paths, model: WaveRNN, optimizer, train_set, ckpt_manager, epochs, batchs, voc_mode):
    # Use same device as model parameters
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        total_loss = 0
        for batch, (x, y, m) in zip(range(batchs), train_set):
            batch_start = time.time()
            batch_loss, y_hat = train_step(x, y, m, model, optimizer, voc_mode)  # 训练一个批次，返回批损失
            total_loss += batch_loss
            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                                  batchs,
                                                                  batch + 1,
                                                                  batch_loss.numpy(),
                                                                  (time.time() - batch_start)), end='')

        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            ckpt_manager.save()

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start) / batchs, total_loss / batchs))

    return y_hat


# 单次训练
def train_step(x, y, m, model, optimizer, voc_mode):
    loss = 0
    with tf.GradientTape() as tape:
        y_hat = model(x, m)
        y = tf.expand_dims(y, axis=-1)
        if voc_mode == 'RAW':
            y_hat = tf.expand_dims(tf.transpose(y_hat, (0, 2, 1)), axis=-1)
            loss_func = tf.nn.cross_entropy(y_hat, y)
        elif voc_mode == 'MOL':
            y = y.float()
            loss_func = Discretized_Mix_Logistic_Loss(y_hat, y)
        loss += loss_func
    batch_loss = loss
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss, y_hat


if __name__ == "__main__":
    main()