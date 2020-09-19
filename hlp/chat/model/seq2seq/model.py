import tensorflow as tf
import config.get_config as _config
import common.layers as layers
from common.data_utils import load_dataset


class Encoder(tf.keras.Model):
    """
    seq2seq的encoder，主要就是使用Embedding和GRU对输入进行编码，
    这里需要注意传入一个初始化的隐藏层，随机也可以，但是我这里就
    直接写了一个隐藏层方法。
    """

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    """
    seq2seq的decoder，将初始化的x、隐藏层和encoder的输出作为
    输入，encoder的输入用来和隐藏层进行attention，得到的上下文
    向量和x进行整合然后丢到gru里去，最后Dense输出一下
    """

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = layers.BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


_, input_token, _, target_token = load_dataset()

encoder = Encoder(len(input_token.word_index) + 1, _config.embedding_dim, _config.units, _config.BATCH_SIZE)
decoder = Decoder(len(target_token.word_index) + 1, _config.embedding_dim, _config.units, _config.BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    """
    :param real:
    :param pred:
    :return: loss
    """
    # 这里进来的real和pred的shape为（128,）
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    # 这里要注意了，因为前面我们对于短的句子进行了填充，所
    # 以对于填充的部分，我们不能用于计算损失，所以要mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

def question_to_answer(inputs={}):
    # 初始化隐藏层，并使用encoder得到隐藏层和decoder的输入给decoder使用
    hidden = [tf.zeros((1, _config.units))]
    enc_out, enc_hidden = encoder(inputs[0], hidden)
    dec_hidden = enc_hidden

    predictions, dec_hidden, attention_weights = decoder(inputs[1], dec_hidden, enc_out)

# @tf.function
def train_step(inp, targ, targ_lang, enc_hidden):
    """
    seq2seq的自定义训练步
    :param inp:
    :param targ:
    :param targ_lang:
    :param enc_hidden:
    :return: 返回的是传入的数据的损失(batch)
    """
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        # 这里初始化decoder的输入，首个token为start，shape为（128, 1）
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * _config.BATCH_SIZE, 1)
        # 这里针对每个训练出来的结果进行损失计算
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # 这一步使用teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
