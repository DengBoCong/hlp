import os
import time
import tensorflow as tf
from prepocesses import dataset_txt, dataset_wave, create_dataset, tar_stop_token, process_wav_name, map_to_text
from config2 import Tacotron2Config
from tacotron2 import Tacotron2

#损失函数
def loss_function(mel_out, mel_out_postnet, mel_gts, tar_token, stop_token):
    mel_gts = tf.transpose(mel_gts, [0, 2, 1])
    mel_out = tf.transpose(mel_out, [0, 2, 1])
    mel_out_postnet = tf.transpose(mel_out_postnet, [0, 2, 1])
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    stop_loss = binary_crossentropy(tar_token, stop_token)
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_out, mel_gts) + tf.keras.losses.MeanSquaredError()(mel_out_postnet, mel_gts)+stop_loss
    return mel_loss

#单次训练
def train_step(input_ids, mel_gts, model, optimizer, tar_token):
    loss = 0
    with tf.GradientTape() as tape:
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(input_ids, mel_gts)
        loss += loss_function(mel_outputs, mel_outputs_postnet, mel_gts, tar_token, gate_outputs)
    batch_loss = loss
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss, mel_outputs_postnet

#启动训练
def train(model, optimizer, dataset, epochs, steps_per_epoch, batch_size, checkpoint):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        total_loss = 0

        for (batch, (input_ids, mel_gts, tar_token)) in enumerate(dataset.take(steps_per_epoch)):
            batch_start = time.time()
            batch_loss, mel_outputs = train_step(input_ids, mel_gts, model, optimizer, tar_token)  # 训练一个批次，返回批损失
            total_loss += batch_loss
            print('\r{}/{} [Batch {} Loss {:.4f}]'.format((batch+1), steps_per_epoch, batch + 1, batch_loss.numpy()), end='')

         #每 50 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(' - {:.0f}s - {:.0f}s/step - loss: {:.4f}'.format(epoch, (time.time() - start), total_loss / steps_per_epoch))

    return mel_outputs

if __name__=="__main__":
    config = Tacotron2Config()
    batch = 2
    batch_size = config.batch_size

    #设置文件路径，训练文本与音频路径
    text_train_path = config.text_train_path
    wave_train_path = config.wave_train_path

    #测试文本路径
    test_to_file = config.text_test_path

    #检查点
    checkpoint_dir = config.checkpoingt_dir
    csv_dir = config.csv_dir

    #此路径包含了所有ljspeech的所有文本数据
    path_all_data_dir = config.path_all_data_dir

    #训练集包含样本数量，测试集包含样本数量
    train_number = config.train_number
    test_number = config.test_number

    #统计wav名称
    wav_name_list = process_wav_name(wave_train_path)
    #根据wav名称生成需要的列表
    sentence_list = map_to_text(csv_dir, wav_name_list)

    #取数据
    input_ids, vocab_inp_size = dataset_txt(sentence_list)
    input_ids = tf.convert_to_tensor(input_ids)
    mel_gts, mel_len_wav = dataset_wave(wave_train_path, config)

    # 生成stop_token的参照值
    tar_token = tar_stop_token(mel_len_wav, mel_gts, config.max_len)
    tar_token = tf.convert_to_tensor(tar_token)
    mel_gts = tf.transpose(mel_gts, [0, 2, 1])

    #建立输入输出流
    dataset, steps_per_epoch = create_dataset(batch, input_ids, mel_gts, tar_token)

    # 初始化模型和优化器
    tacotron2 = Tacotron2(vocab_inp_size, config)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    #检查点
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(tacotron2=tacotron2)

    #训练1000轮的声音会
    epochs = 2
    mel_outputs = train(tacotron2, optimizer, dataset, epochs, steps_per_epoch, batch_size, checkpoint)
