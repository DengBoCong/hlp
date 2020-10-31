import joblib
from tensorflow.keras.optimizers import Adam
from config import Tacotron2Config
from tacotron_model import tacotron1
import tensorflow as tf
import time


def tf_data(batch_size, input_ids, mel_gts):
    BUFFER_SIZE = len(input_ids)
    steps_per_epoch = BUFFER_SIZE // batch_size
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).cache().shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # input_ids, mel_gts = next(iter(dataset))
    return dataset, steps_per_epoch


def loss_function(mel_out, mel_out_pred, mag_gts, mag_gts_pred):
    mel_loss = tf.keras.losses.MeanAbsoluteError()(mel_out, mel_out_pred) + tf.keras.losses.MeanAbsoluteError()(
        mag_gts, mag_gts_pred)
    return mel_loss


optimizer = tf.keras.optimizers.Adam()

config = Tacotron2Config()
batch_size = config.BATCH_SIZE
decoder_input_training = joblib.load('./data/' + 'decoder_input_training.pkl')
mel_spectro_training = joblib.load('./data/' + 'mel_spectro_training.pkl')
spectro_training = joblib.load('./data/' + 'spectro_training.pkl')
text_input_training = joblib.load('./data/' + 'text_input_ml_training.pkl')
vocabulary = joblib.load('./data/' + 'vocabulary.pkl')

dataset, steps_per_epoch = tf_data(batch_size=2, input_ids=(text_input_training, decoder_input_training),
                                   mel_gts=(mel_spectro_training, spectro_training))


def train_step(model_dir, text_input_training, decoder_input_training, mel_spectro_training, spectro_training):
    loss = 0
    batch_size = 2
    config = Tacotron2Config()

    # dataset, steps_per_epoch=tf_data(batch_size, input_ids, mel_gts)

    vocab_size = len(vocabulary)
    model = tacotron1(vocab_size, config)
#go
    dec_input = tf.expand_dims(tf.zeros(shape=[decoder_input_training.shape[0], decoder_input_training.shape[2]]), 1)
    with tf.GradientTape() as tape:
        # 解码器输入符号
        # _go_frames

        # 教师强制 - 将目标词作为下一个输入
        for t in range(config.MAX_MEL_TIME_LENGTH):
            mag_output, mel_hat_last_frame, mel_output = model(dec_input, text_input_training)
            # 将编码器输出 （enc_output） 传送至解码器，解码

            loss += loss_function(tf.expand_dims(mel_spectro_training[:, t, :], axis=1), mel_output,
                                  tf.expand_dims(spectro_training[:, t, :], axis=1),
                                  mag_output)  # 根据预测计算损失

            # 使用教师强制，下一步输入符号是训练集中对应目标符号
            dec_input = tf.expand_dims(decoder_input_training[:, t, :], 1)  # mel_hat_last_frame

            #print("dec_input",dec_input.shape)
    batch_loss = (loss / int(decoder_input_training.shape[1]))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    print("batch_loss:", batch_loss)
    return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()

        decoder_input_training = inp[1]
        spectro_training = targ[1]
        mel_spectro_training = targ[0]
        text_input_training = inp[0]

        batch_loss = train_step('./data/', text_input_training, decoder_input_training, mel_spectro_training,
                                spectro_training)  # 训练一个批次，返回批损失
        total_loss += batch_loss

        if batch % 2 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
            print('Time taken for 2 batches {} sec\n'.format(time.time() - batch_start))

    # 每 2 个周期（epoch），保存（检查点）一次模型
    # if (epoch + 1) % 2 == 0:
    # checkpoint.save(file_prefix=checkpoint_prefix)

# print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
# print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#
#
# def train(model_dir, data_dir='./data/'):
#     config=Tacotron2Config()
#     decoder_input_training = joblib.load(data_dir + 'decoder_input_training.pkl')
#     mel_spectro_training = joblib.load(data_dir + 'mel_spectro_training.pkl')
#     spectro_training = joblib.load(data_dir + 'spectro_training.pkl')
#     text_input_training = joblib.load(data_dir + 'text_input_ml_training.pkl')
#     vocabulary = joblib.load(data_dir + 'vocabulary.pkl')
#     vocab_size=len(vocabulary)
#     model = tacotron1(vocab_size, config)
#


# opt = Adam()
# model.compile(optimizer=opt,
#               loss=['mean_absolute_error', 'mean_absolute_error'])
#


# TODO: 采用teacher-forcing方式进行训练
# model.fit([text_input_training, decoder_input_training],
#                           [mel_spectro_training, spectro_training],
#                           epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
#                           verbose=1, validation_split=0.15)

# model.save(model_dir + 'tts-model.h5')

# train('./data/')
