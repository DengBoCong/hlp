"""
transformer中网络层的部分
包括：
- 多头注意力（Multi-head attention）
- 点式前馈网络（Point wise feed forward network）
- 编码器层（Encoder layer）
- 解码器层（Decoder layer）
- 编码器（Encoder）
- 解码器（Decoder）
- Transformer
- 优化器（Optimizer）
- 损失函数与指标（Loss and metrics）

在此模块空间中完成数据集及字典的加载

"""

import sys
sys.path.append('..')
import copy
import tensorflow as tf
from common import self_attention
from config import get_config as _config


# Beam search类
class BeamSearch(object):
    """
    BeamSearch使用说明：
    1.首先需要将问句编码成token向量并对齐，然后调用init_input方法进行初始化
    2.对模型要求能够进行批量输入
    3.BeamSearch使用实例已经集成到Chatter中，如果不进行自定义调用，
    可以将聊天器继承Chatter，在满足上述两点的基础之上设计_create_predictions方法，并调用BeamSearch
    """

    def __init__(self, beam_size, max_length, worst_score):
        """
        初始化BeamSearch的序列容器
        """
        self.remain_beam_size = beam_size
        self.max_length = max_length - 1
        self.remain_worst_score = worst_score
        self._reset_variables()

    def __len__(self):
        """
        已存在BeamSearch的序列容器的大小
        """
        return len(self.container)

    def init_variables(self, inputs, dec_input):
        """
        用来初始化输入
        :param inputs: 已经序列化的输入句子
        :param dec_input: 编码器输入序列
        :return: 无返回值
        """
        self.container.append((1, dec_input))
        self.inputs = inputs
        self.dec_inputs = dec_input

    def get_variables(self):
        """
        用来动态的更新模型的inputs和dec_inputs，以适配随着Beam Search
        结果的得出而变化的beam_size
        :return: requests, dec_inputs
        """
        # 生成多beam输入
        inputs = self.inputs
        for i in range(len(self) - 1):
            inputs = tf.concat([inputs, self.inputs], 0)
        requests = inputs
        # 生成多beam的decoder的输入
        temp = self.container[0][1]
        for i in range(1, len(self)):
            temp = tf.concat([temp, self.container[i][1]], axis=0)
        self.dec_inputs = copy.deepcopy(temp)
        return requests, self.dec_inputs

    def _reduce_end(self, end_sign):
        """
        当序列遇到了结束token，需要将该序列从容器中移除
        :return: 无返回值
        """
        for idx, (s, dec) in enumerate(self.container):
            temp = dec.numpy()
            if temp[0][-1] == end_sign:
                self.result.append(self.container[idx][1])
                del self.container[idx]
                self.beam_size -= 1

    def _reset_variables(self):
        """
        重置相关变量
        :return: 无返回值
        """
        self.beam_size = self.remain_beam_size
        self.worst_score = self.remain_worst_score
        self.container = []  # 保存中间状态序列的容器，元素格式为(score, sequence)类型为(float, [])
        self.result = []  # 用来保存已经遇到结束符的序列
        self.inputs = tf.constant(0, shape=(1, 1))
        self.dec_inputs = tf.constant(0, shape=(1, 1))  # 处理后的的编码器输入

    def add(self, predictions, end_sign):
        """
        往容器中添加预测结果，在本方法中对预测结果进行整理、排序的操作
        :param predictions: 传入每个时间步的模型预测值
        :return: 无返回值
        """
        remain = copy.deepcopy(self.container)
        for i in range(self.dec_inputs.shape[0]):
            for k in range(predictions.shape[-1]):
                if predictions[i][k] <= 0:
                    continue
                # 计算分数
                score = remain[i][0] * predictions[i][k]
                # 判断容器容量以及分数比较

                if len(self) < self.beam_size or score > self.worst_score:
                    self.container.append((score, tf.concat([remain[i][1], tf.constant([[k]], shape=(1, 1))], axis=-1)))
                    if len(self) > self.beam_size:
                        sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.container)])
                        del self.container[sorted_scores[0][1]]
                        self.worst_score = sorted_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)
        self._reduce_end(end_sign=end_sign)

    def get_result(self):
        """
        获取最终beam个序列
        :return: beam个序列
        """
        result = self.result

        # 每轮回答之后，需要重置容器内部的相关变量值
        self._reset_variables()
        return result


# 多头注意力层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self_attention.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# 点式前馈网络（Point wise feed forward network）
def point_wise_feed_forward_network(d_model, dff):
    """
    简单的两个全连接层网络
    Args:
        d_model:第二层dense的维度
        dff: 第一层dense的维度

    Returns:包含两个dense层的Sequential

    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# 编码器层（Encoder layer）
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# 解码器层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# 编码器
class Encoder(tf.keras.layers.Layer):
    """
    包含
    - 输入嵌入（Input Embedding）
    - 位置编码（Positional Encoding）
    - N 个编码器层（encoder layers）
    输入经过嵌入（embedding）后，该嵌入与位置编码相加。该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入。

    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self_attention.positional_encoding(maximum_position_encoding,
                                                               self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# 解码器（Decoder）
class Decoder(tf.keras.layers.Layer):
    """
    解码器包括：
    - 输出嵌入（Output Embedding）
    - 位置编码（Positional Encoding）
    - N 个解码器层（decoder layers）
    目标（target）经过一个嵌入后，该嵌入和位置编码相加。该加法结果是解码器层的输入。解码器的输出是最后的线性层的输入。
    """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self_attention.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# Transformer模型
class Transformer(tf.keras.Model):
    """
    Transformer 包括编码器，解码器和最后的线性层。解码器的输出是线性层的输入，返回线性层的输出。
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# 自定义优化器（Optimizer）
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_model(input_vocab_size, target_vocab_size):
    """获取模型相关变量"""
    learning_rate = CustomSchedule(_config.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    transformer = Transformer(_config.num_layers, _config.d_model, _config.num_heads, _config.dff,
                              input_vocab_size + 1, target_vocab_size + 1,
                              pe_input=input_vocab_size + 1,
                              pe_target=target_vocab_size + 1,
                              rate=_config.dropout_rate)
    return optimizer, train_loss, train_accuracy, transformer


def load_checkpoint(transformer, optimizer):
    # 加载检查点
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新的检查点！')


def main():
    """
    忽略

    测试部分，用来测试实现schedule sampling
    需要完成：
    1.Embedding Mix 设置混合词向量算法
    2.增加decoder 使得模型训练经过两层decoder，两个decoder参数相同
    3.Weights update 只反向传播第二个decoder

    """
    # 模拟输入输出
    inp = tf.ones([64, 30])
    tar = tf.ones([64, 20])
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = self_attention.create_masks(inp, tar_inp)
    # 模型创建
    transformer = Transformer(_config.num_layers, _config.d_model, _config.num_heads, _config.dff,
                              666, 666,
                              pe_input=666,
                              pe_target=666,
                              rate=_config.dropout_rate)
    transformer(inp, tar_inp,
                True,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask)
    transformer.summary()
    pass


if __name__ == '__main__':
    main()
