import time
import tensorflow as tf


def train(model, optimizer, train_data_generator, epochs, checkpoint, batchs):
    """
    训练模块
    """
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        total_loss = 0
        for batch, (input_ids, mel_gts, mel_len_wav) in zip(range(batchs), train_data_generator):
            batch_start = time.time()
            print("mel_gts:", mel_gts.shape)
            tar_token = tar_stop_token(mel_len_wav, mel_gts, config.max_input_length)
            batch_loss, mel_outputs = _train_step(input_ids, mel_gts, model, optimizer, tar_token)  # 训练一个批次，返回批损失
            total_loss += batch_loss
            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                                  batchs,
                                                                  batch + 1,
                                                                  batch_loss.numpy(),
                                                                  (time.time() - batch_start)), end='')

        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save()

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start) / batchs, total_loss / batchs))

    return mel_outputs


def _loss_function(mel_out, mel_out_postnet, mel_gts, tar_token, stop_token):
    """
    损失函数
    """
    mel_gts = tf.transpose(mel_gts, [0, 2, 1])
    mel_out = tf.transpose(mel_out, [0, 2, 1])
    mel_out_postnet = tf.transpose(mel_out_postnet, [0, 2, 1])
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    stop_loss = binary_crossentropy(tar_token, stop_token)
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_out, mel_gts) + tf.keras.losses.MeanSquaredError()(
        mel_out_postnet, mel_gts) + stop_loss
    return mel_loss


def train_step(input_ids, mel_gts, model, optimizer, tar_token):
    loss = 0
    with tf.GradientTape() as tape:
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(input_ids, mel_gts)
        loss += _loss_function(mel_outputs, mel_outputs_postnet, mel_gts, tar_token, gate_outputs)
    batch_loss = loss
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss, mel_outputs_postnet
