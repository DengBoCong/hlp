import logging
import os
import time
import tensorflow as tf
from plot.plot import plot_mel
from dataset.dataset_wav import Dataset_wave
from dataset.dataset_txt import Dataset_txt
from config.config import Tacotron2Config
from model.tacotron2 import Tacotron2

logging.basicConfig(
     level=logging.WARNING,
     format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
 )
#数据处理部分

#对文字的处理
path_to_file = r".\text\wenzi.txt"
input_ids,en_tokenizer = Dataset_txt(path_to_file)
input_ids = tf.convert_to_tensor(input_ids)
#对音频处理
path = r"./wavs/"
mel_gts = Dataset_wave(path)
mel_gts = tf.cast(mel_gts, tf.float32)
#tf.data
config = Tacotron2Config(n_speakers=1, reduction_factor=1)
batch_size=2
BUFFER_SIZE = len(input_ids)
steps_per_epoch = BUFFER_SIZE // batch_size
dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)
input_ids, mel_gts = next(iter(dataset))
vocab_inp_size = len(en_tokenizer.word_index) + 1
#初始化模型
tacotron2 = Tacotron2(vocab_inp_size, config)

#损失函数和优化器
def loss_function(mel_out, mel_out_postnet, mel_gts):
    mel_gts=tf.transpose(mel_gts,[0,2,1])
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_out, mel_gts) + tf.keras.losses.MeanSquaredError()(mel_out_postnet, mel_gts)
    return mel_loss

optimizer = tf.keras.optimizers.Adam(lr=0.001)

#检查点
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 tacotron2=tacotron2)

#训练
def train_step(input_ids, mel_gts):
    loss = 0
    with tf.GradientTape() as tape:
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments=tacotron2(input_ids,mel_gts)
        loss+=loss_function(mel_outputs,mel_outputs_postnet,mel_gts)
    batch_loss = (loss / int(mel_gts.shape[1]))
    variables = tacotron2.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss,mel_outputs_postnet

#启动训练
EPOCHS = 2
for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0
    print("次数：+1")
    for (batch, (input_ids, mel_gts)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()

        batch_loss,mel_outputs = train_step(input_ids, mel_gts)  # 训练一个批次，返回批损失
        total_loss += batch_loss

        if batch % 2 == 0:
             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                          batch,
                                                          batch_loss.numpy()))
             print('Time taken for 2 batches {} sec\n'.format(time.time() - batch_start))

    # 每 2 个周期（epoch），保存（检查点）一次模型
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
print("结束")
mel_outputs=tf.transpose(mel_outputs,[0,2,1])
#画频谱图
plot_mel(mel_outputs)
plot_mel(mel_gts)
exit(0)