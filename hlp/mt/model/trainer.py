import tensorflow as tf
from model import network
from common import self_attention
from config import get_config as _config
import time
from common import preprocess


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = self_attention.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = network.transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = network.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, network.transformer.trainable_variables)
    network.optimizer.apply_gradients(zip(gradients, network.transformer.trainable_variables))

    network.train_loss(loss)
    network.train_accuracy(tar_real, predictions)


def train():
    print('开始处理数据集...')
    train_dataset, val_dataset = preprocess.split_batch(network.input_sequences, network.target_sequences)
    print('数据集处理完毕！')
    checkpoint_path = _config.checkpoint_path

    ckpt = tf.train.Checkpoint(transformer=network.transformer,
                               optimizer=network.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    print("开始训练...")
    for epoch in range(_config.EPOCHS):
        start = time.time()

        network.train_loss.reset_states()
        network.train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, network.train_loss.result(), network.train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            network.train_loss.result(),
                                                            network.train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('训练完毕！')


if __name__ == '__main__':
    train()