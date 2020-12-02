import tensorflow as tf
import utils.layers as layers


class Encoder(tf.keras.Model):
    """
    seq2seq的encoder，主要就是使用Embedding和GRU对输入进行编码，
    这里需要注意传入一个初始化的隐藏层，随机也可以，但是我这里就
    直接写了一个隐藏层方法。
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, batch_sz: int):
        """
        Args:
            vocab_size: 词汇量大小
            embedding_dim: 词嵌入维度
            enc_units: 单元大小
            batch_sz: batch大小
        Returns:
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, inputs: tf.Tensor, hidden: tf.Tensor):
        """
        Args:
            inputs: 输入序列
            hidden: gru初始化隐藏层
        Returns:output, state
        """
        inputs = self.embedding(inputs)
        output, state = self.gru(inputs, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    """
    seq2seq的decoder，将初始化的x、隐藏层和encoder的输出作为
    输入，encoder的输入用来和隐藏层进行attention，得到的上下文
    向量和x进行整合然后丢到gru里去，最后Dense输出一下
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int, batch_sz: int):
        """
        Args:
            vocab_size: 词汇量大小
            embedding_dim: 词嵌入维度
            dec_units: 单元大小
            batch_sz: batch大小
        Returns:
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = layers.BahdanauAttention(self.dec_units)

    def call(self, inputs: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor):
        """
        Args:
            inputs: 输入序列
            hidden: gru初始化隐藏层
            enc_output: encoder的输出
        Returns: inputs, state, attention_weights
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)
        inputs = self.embedding(inputs)
        inputs = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)
        output, state = self.gru(inputs)

        output = tf.reshape(output, (-1, output.shape[2]))
        inputs = self.fc(output)

        return inputs, state, attention_weights
