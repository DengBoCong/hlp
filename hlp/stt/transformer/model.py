import tensorflow as tf
from hlp.utils.layers import positional_encoding
from hlp.utils.layers import create_padding_mask
from hlp.utils.layers import create_look_ahead_mask
from hlp.utils.layers import transformer_encoder_layer
from hlp.utils.layers import transformer_decoder_layer


def encoder(vocab_size: int, embedding_dim: int, num_layers: int, feature_dim: int,
            encoder_units: int, num_heads: int, dropout: float = 0.1) -> tf.keras.Model:
    """
    transformer tts的encoder层
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param num_layers: encoder层数量
    :param feature_dim: 特征维度
    :param encoder_units: 单元大小
    :param dropout: encoder的dropout采样率
    :param num_heads: 头注意力数量
    """
    inputs = tf.keras.Input(shape=(None, feature_dim))
    padding_mask = tf.keras.layers.Lambda(_create_padding_mask,
                                          output_shape=(1, 1, None))(inputs)
    outputs = tf.keras.layers.Dense(embedding_dim)(inputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    outputs = outputs * tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
    pos_encoding = positional_encoding(vocab_size, embedding_dim)
    outputs = outputs + pos_encoding[:, :tf.shape(outputs)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    for i in range(num_layers):
        outputs = transformer_encoder_layer(
            units=encoder_units,
            d_model=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            name="transformer_encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=inputs, outputs=[outputs, padding_mask])


def decoder(vocab_size: int, embedding_dim: int, num_layers: int,
            decoder_units: int, num_heads: int, dropout: float = 0.1) -> tf.keras.Model:
    """
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param num_layers: encoder层数量
    :param decoder_units: 单元大小
    :param num_heads: 头注意力数量
    :param dropout: decoder的dropout采样率
    """
    enc_outputs = tf.keras.Input(shape=(None, None))
    dec_inputs = tf.keras.Input(shape=(None,))
    padding_mask = tf.keras.Input(shape=(1, 1, None))
    pos_encoding = positional_encoding(vocab_size, embedding_dim)
    look_ahead_mask = tf.keras.layers.Lambda(_combine_mask,
                                             output_shape=(1, None, None))(dec_inputs)

    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim)(dec_inputs)
    embeddings *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = transformer_decoder_layer(
            units=decoder_units, d_model=embedding_dim, num_heads=num_heads,
            dropout=dropout, name="transformer_decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(outputs)

    return tf.keras.Model(inputs=[dec_inputs, enc_outputs, padding_mask], outputs=outputs)


def _combine_mask(seq: tf.Tensor):
    """
    对input中的不能见单位进行mask
    :param seq: 输入序列
    :return: mask
    """
    look_ahead_mask = create_look_ahead_mask(seq)
    padding_mask = create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)


def _create_padding_mask(seq: tf.Tensor):
    """
    用于创建输入序列的扩充部分的mask，专用于mel序列
    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = seq[:, :, 0]
    return seq[:, tf.newaxis, tf.newaxis, :]
