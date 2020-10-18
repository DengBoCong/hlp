import logging
import os
import time
import tensorflow as tf
from utils import process_text,process_wave,tf_data
from config2 import Tacotron2Config
from model.tacotron2 import Tacotron2


#数据处理部分
#对文字的处理
# path_to_file = r".\text\wenzi.txt"


#损失函数和优化器
def loss_function(mel_out, mel_out_postnet, mel_gts):
    mel_gts=tf.transpose(mel_gts,[0,2,1])
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_out, mel_gts) + tf.keras.losses.MeanSquaredError()(mel_out_postnet, mel_gts)
    return mel_loss


#单次训练
def train_step(input_ids, mel_gts,model,optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments=model(input_ids,mel_gts)
        loss+=loss_function(mel_outputs,mel_outputs_postnet,mel_gts)
    #batch_loss = (loss / int(mel_gts.shape[1]))
    batch_loss = loss*10000
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss,mel_outputs_postnet

#启动训练
def train(model, optimizer,dataset, epochs,steps_per_epoch):
    for epoch in range(epochs):
        start = time.time()

        total_loss = 0
        print("次数：+1")
        for (batch, (input_ids, mel_gts)) in enumerate(dataset.take(steps_per_epoch)):
            batch_start = time.time()
            print(input_ids.shape)
            batch_loss, mel_outputs = train_step(input_ids, mel_gts,model, optimizer)  # 训练一个批次，返回批损失
            total_loss += batch_loss

            if batch % 2 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
                print('Time taken for 2 batches {} sec\n'.format(time.time() - batch_start))

        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#画频谱图

if __name__=="__main__":
    config = Tacotron2Config(n_speakers=1, reduction_factor=1)
    batch_size=config.batch_size
    #设置文件路径
    text_train_path=config.text_train_path
    wave_train_path = config.wave_train_path
    #取数据
    input_ids,vocab_inp_size=process_text(text_train_path)
    print("input_ids:",input_ids.shape)
    print("vocab_inp_size::::::::::",vocab_inp_size)
    mel_gts=process_wave(wave_train_path)
    #建立输入输出流
    dataset,steps_per_epoch=tf_data(batch_size, input_ids, mel_gts)
    # 初始化模型
    tacotron2 = Tacotron2(vocab_inp_size, config)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    epochs=2
    #训练
    train(tacotron2, optimizer, dataset, epochs, steps_per_epoch)
    # 检查点
    checkpoint_dir = './training_checkpoints2'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(tacotron2=tacotron2)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory='./training_checkpoints2',
        max_to_keep=5
        )
    manager.save()