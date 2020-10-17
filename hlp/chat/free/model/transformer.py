import tensorflow as tf
import common.layers as layers
import common.data_utils as _data


def encoder(vocab_size, num_layers, units, d_model,
            num_heads, dropout, name="encoder"):
    """
    transformer的encoder，使用函数式API进行编写，实现了
    模型层内部的一系列操作，num_layers决定了使用多少个
    encoder_layer层，更具Transformer架构里面的描述，可以根据
    效果进行调整，在encoder中还进行了位置编码，具体原理自行翻阅
    资料，就是实现公式的问题，这里就不多做注释了
    :param vocab_size:token大小
    :param num_layers:编码解码的数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return: Model(inputs=[inputs, padding_mask], outputs=outputs)
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = layers.PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 这里layer使用的name是为了调试的时候答应信息方便查看，也可以不写
    for i in range(num_layers):
        outputs = layers.transformer_encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="transformer_encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
    """
    transformer的decoder，使用函数式API进行编写，实现了
    模型层内部的一系列操作，相关的一些变量的时候基本和上面
    的encoder差不多，这里不多说
    :param vocab_size:token大小
    :param num_layers:编码解码的层数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = layers.PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = layers.transformer_decoder_layer(
            units=units, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="transformer_decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                          outputs=outputs, name=name)


def transformer(vocab_size, num_layers, units, d_model,
                num_heads, dropout, name="transformer"):
    """
    transformer的粗粒度的结构实现，在忽略细节的情况下，看作是
    encoder和decoder的实现，这里需要注意的是，因为是使用self_attention，
    所以在输入的时候，这里需要进行mask，防止暴露句子中带预测的信息，影响
    模型的效果
    :param vocab_size:token大小
    :param num_layers:编码解码层的数量
    :param units:单元大小
    :param d_model:深度
    :param num_heads:多头注意力的头部层数量
    :param dropout:dropout的权重
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 使用了Lambda将方法包装成层，为的是满足函数式API的需要
    enc_padding_mask = tf.keras.layers.Lambda(
        _data.create_padding_mask, output_shape=(1, 1, None),
        name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        _data.create_look_ahead_mask, output_shape=(1, None, None),
        name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        _data.create_padding_mask, output_shape=(1, 1, None),
        name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    优化器将 Adam 优化器与自定义的学习速率调度程序配合使用，这里直接参考了官网的实现
    因为是公式的原因，其实大同小异
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def accuracy(real, pred):
    real = tf.reshape(real, shape=(-1, 40 - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(real, pred)
