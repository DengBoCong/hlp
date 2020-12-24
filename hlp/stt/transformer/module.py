import os
import time
import tensorflow as tf
from hlp.utils.beamsearch import BeamSearch
from hlp.stt.utils.load_dataset import load_data
from hlp.utils.optimizers import loss_func_mask
from hlp.stt.utils.audio_process import wav_to_feature


def train(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
          epochs: int, checkpoint: tf.train.CheckpointManager, train_data_path: str, max_len: int,
          vocab_size: int, batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    训练模块
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param optimizer: 优化器
    :param checkpoint: 检查点管理器
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    """
    _, train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                  batch_size=batch_size, buffer_size=buffer_size, dict_path=dict_path,
                  valid_data_split=valid_data_split, valid_data_path=valid_data_path,
                  max_train_data_size=max_train_data_size, max_valid_data_size=max_valid_data_size)

    if steps_per_epoch == 0:
        print("训练数据量过小，小于batch_size，请添加数据后重试")
        exit(0)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        total_loss = 0

        for (batch, (audio_feature, sentence)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_start = time.time()
            sentence_input = sentence[:, :-1]
            sentence_real = sentence[:, 1:]

            batch_loss, sentence_predictions = _train_step(encoder, decoder, optimizer,
                                                           sentence_input, sentence_real, audio_feature)
            total_loss += batch_loss

            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format(
                (batch + 1), steps_per_epoch, batch + 1, batch_loss.numpy(), (time.time() - batch_start)), end="")

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                      total_loss / steps_per_epoch))

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()

            if valid_steps_per_epoch == 0:
                print("验证数据量过小，小于batch_size，请添加数据后重试")
                exit(0)

            _valid_step(encoder=encoder, decoder=decoder, dataset=valid_dataset, steps_per_epoch=valid_steps_per_epoch)


def recognize(encoder: tf.keras.Model, decoder: tf.keras.Model, beam_size: int,
              audio_feature_type: str, max_length: int, max_sentence_length: int, dict_path: str):
    """
    语音识别模块
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param beam_size: beam_size
    :param audio_feature_type: 特征类型
    :param max_length: 最大音频补齐长度
    :param max_sentence_length: 最大音频补齐长度
    :param dict_path: 字典保存路径
    """
    beam_search_container = BeamSearch(beam_size=beam_size, max_length=max_sentence_length, worst_score=0)

    print("Agent: 你好！结束识别请输入ESC。")
    while True:
        path = input("Path: ")
        if path == "ESC":
            print("Agent: 再见！")
            exit(0)

        if not os.path.exists(path):
            print("音频文件不存在，请重新输入")
            continue

        audio_feature = wav_to_feature(path, audio_feature_type)
        audio_feature = tf.expand_dims(audio_feature, axis=0)
        audio_feature = tf.keras.preprocessing.sequence.pad_sequences(audio_feature, maxlen=max_length,
                                                                      dtype="float32", padding="post")

        with open(dict_path, 'r', encoding='utf-8') as dict_file:
            json_string = dict_file.read().strip().strip("\n")
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        dec_input = tf.expand_dims([tokenizer.word_index.get("<start>", "<unk>")], 0)

        beam_search_container.reset(inputs=audio_feature, dec_input=dec_input)
        for i in range(max_sentence_length):
            enc_outputs, padding_mask = encoder(audio_feature)
            sentence_predictions = decoder(inputs=[dec_input, enc_outputs, padding_mask])
            sentence_predictions = tf.nn.softmax(sentence_predictions)
            sentence_predictions = sentence_predictions[:, -1, :]

            beam_search_container.expand(predictions=sentence_predictions, end_sign=tokenizer.word_index.get("<end>"))
            if beam_search_container.beam_size == 0:
                break

            audio_feature, dec_input = beam_search_container.get_search_inputs()

        beam_search_result = beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = tokenizer.sequences_to_texts(temp)[0]
            text = text.replace("<start>", '').replace("<end>", '').replace(' ', '')
            result = '<' + text + '>' + result

        print("识别句子为：{}".format(result))

    print("识别结束")


def _train_step(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer,
                sentence_input: tf.Tensor, sentence_real: tf.Tensor, audio_feature: tf.Tensor):
    """
    训练步
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param sentence_input: sentence序列
    :param audio_feature: 音频特征序列
    :param sentence_real: ground-true句子序列
    :param optimizer 优化器
    :return: batch损失和post_net输出
    """
    with tf.GradientTape() as tape:
        enc_outputs, padding_mask = encoder(audio_feature)
        sentence_predictions = decoder(inputs=[sentence_input, enc_outputs, padding_mask])
        loss = loss_func_mask(sentence_real, sentence_predictions)

    batch_loss = loss
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss, sentence_predictions


def _valid_step(encoder: tf.keras.Model, decoder: tf.keras.Model,
                dataset: tf.data.Dataset, steps_per_epoch: int):
    """
    验证模块
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param dataset: 验证数据dataset
    :param steps_per_epoch: 验证训练步
    :return: 无返回值
    """
    print("验证轮次")
    start_time = time.time()
    total_loss = 0

    for (batch, (audio_feature, sentence)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()
        sentence_input = sentence[:, :-1]
        sentence_real = sentence[:, 1:]

        enc_outputs, padding_mask = encoder(audio_feature)
        sentence_predictions = decoder(inputs=[sentence_input, enc_outputs, padding_mask])
        batch_loss = loss_func_mask(sentence_real, sentence_predictions)
        total_loss += batch_loss

        print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                              steps_per_epoch, batch + 1, batch_loss.numpy(),
                                                              (time.time() - batch_start)), end='')
    print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                  total_loss / steps_per_epoch))


def load_checkpoint(encoder: tf.keras.Model, decoder: tf.keras.Model,
                    checkpoint_dir: str, execute_type: str, checkpoint_save_size: int):
    """
    恢复检查点
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if execute_type == "generate":
            print("没有检查点，请先执行train模式")
            exit(0)

    return ckpt_manager
