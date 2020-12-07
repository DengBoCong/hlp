import tensorflow as tf
from config import get_config as _config
from common import preprocess
from common import tokenize
import numpy
from sklearn.model_selection import train_test_split
import time
from model import nmt_model


class LanguageModel(tf.keras.Model):
    """
    语言模型，将input输入
    """

    def __init__(self, vocab_size, d_embedding, batch_size, d_rnn):
        super(LanguageModel, self).__init__()
        # 初始参数
        self.d_embedding = d_embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, d_embedding)
        self._init_state = [tf.zeros([batch_size, d_rnn]), tf.zeros([batch_size, d_rnn])]

        self.cell0 = tf.keras.layers.LSTMCell(d_rnn)
        self.cell1 = tf.keras.layers.LSTMCell(d_rnn)

        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, sequences):
        """
        传入已编码的句子,shape ---> (batch_size, seq_len)
        返回预测序列
        """
        output = []
        sequences = self.embedding(sequences)  # shape ---> (batch_size, seq_len, vocab_size)
        sequences *= tf.math.sqrt(tf.cast(self.d_embedding, tf.float32))
        # output 为输出的字符列表，每个列表元素shape --> (batch_size, vocab_size)
        state0 = self._init_state
        state1 = self._init_state
        for sequences_t in tf.unstack(sequences, axis=1):  # sequences_t.shape --> (batch_size, vocab_size)
            out0, state0 = self.cell0(sequences_t, state0)  # out0.shape --> (batch_size, vocab_size)
            out1, state1 = self.cell1(out0, state1)
            out1 = self.output_layer(out1)
            output.append(out1)

        predictions = tf.stack(output, axis=1)  # prediction.shape --> (batch_size, seq_len, vocab_size)
        return predictions


def _loss_function(real, pred):
    """
    损失计算
    使用mask将填充部分loss去掉
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def _train_step(sequences, language_model, optimizer, train_loss, train_accuracy):
    """
    @param sequences: 已编码的一个batch的数据集  shape --> (batch_size, seq_length)
    @param language_model: 语言模型实例
    @param optimizer: 优化器
    """
    seq_input = sequences[:, :-1]
    seq_real = sequences[:, 1:]

    with tf.GradientTape() as tape:
        predictions = language_model(seq_input)
        loss = _loss_function(seq_real, predictions)

    gradients = tape.gradient(loss, language_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, language_model.trainable_variables))

    train_loss(loss)
    train_accuracy(seq_real, predictions)


def _lm_preprocess():
    """数据的预处理及编码"""
    print('正在加载、预处理数据...')
    sentences = preprocess.load_single_sentences(_config.lm_path_to_train_file, _config.lm_num_sentences, column=2)
    sentences = preprocess.preprocess_sentences_lm(sentences, language=_config.lm_language)
    print('已加载句子数量:%d' % _config.lm_num_sentences)
    print('数据加载、预处理完毕！\n')

    # 生成及保存字典
    tokenizer, vocab_size = tokenize.create_tokenizer(sentences
                                                      , _config.lm_language
                                                      , model_type="lm")
    print('生成字典大小:%d' % vocab_size)
    print('源语言字典生成、保存完毕！\n')

    print("正在编码训练集句子...")
    max_sequence_length = tokenize.create_encoded_sentences(sentences=sentences
                                                            , tokenizer=tokenizer
                                                            , language=_config.lm_language
                                                            , postfix='_lm'
                                                            , model_type="lm")
    print('最大句子长度:%d' % max_sequence_length)
    print("句子编码完毕！\n")

    return tokenizer, vocab_size, max_sequence_length


def _get_dataset_lm():
    """数据集加载及划分"""
    # 加载
    _, sentences_path = tokenize.get_mode_and_path_sentences(_config.lm_language, model_type="lm", postfix='_lm')
    tensor = numpy.loadtxt(sentences_path, dtype='int32')

    # 划分
    train_dataset, val_dataset = train_test_split(tensor, train_size=_config.lm_train_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(_config.lm_BATCH_SIZE).batch(_config.lm_BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)
    val_dataset = val_dataset.shuffle(_config.lm_BATCH_SIZE).batch(_config.lm_BATCH_SIZE, drop_remainder=True)

    return train_dataset, val_dataset


def train(epochs=_config.lm_EPOCHS):

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    tokenizer, vocab_size, max_sequence_length = _lm_preprocess()
    train_dataset, val_dataset = _get_dataset_lm()
    language_model = LanguageModel(vocab_size
                                   , _config.lm_d_embedding
                                   , _config.lm_BATCH_SIZE
                                   , _config.lm_d_rnn)
    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    ckpt = tf.train.Checkpoint(language_model=language_model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, _config.lm_checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if nmt_model.check_point(model_type='lm'):
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    batch_sum = int((_config.lm_num_sentences*_config.lm_train_size)//_config.lm_BATCH_SIZE)
    train_seq_sum = int(batch_sum * _config.lm_BATCH_SIZE)

    print("开始训练...")
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        trained_seq_sum = 0
        for batch, sequences in enumerate(train_dataset):
            _train_step(sequences, language_model, optimizer, train_loss, train_accuracy)
            trained_seq_sum += _config.lm_BATCH_SIZE
            print('\r{}/{} [batch {} loss {:.4f} accuracy {:.4f}]'.format(trained_seq_sum, train_seq_sum, batch + 1
                                                                          , train_loss.result()
                                                                          , train_accuracy.result()), end='')
        print('\r{}/{} [==============================]'.format(train_seq_sum, train_seq_sum), end='')
        history['accuracy'].append(train_accuracy.result().numpy())
        history['loss'].append(train_loss.result().numpy())

        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / (_config.lm_num_sentences*_config.lm_train_size)
        print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - accuracy {:.4f}'
              .format(epoch_time, step_time * 1000, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % _config.checkpoints_save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('检查点已保存至：{}'.format(ckpt_save_path))

    if (epoch + 1) % _config.checkpoints_save_freq != 0:
        ckpt_save_path = ckpt_manager.save()
        print('检查点已保存至：{}'.format(ckpt_save_path))

    return history


def main():
    train()


if __name__ == '__main__':
    main()
