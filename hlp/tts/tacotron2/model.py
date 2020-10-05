import tensorflow as tf
from dataset.dataset_wav import Dataset_wave
from dataset.dataset_txt import Dataset_txt
from config.config import Tacotron2Config
#数据处理部分
config = Tacotron2Config()
batch_size=2
path_to_file = r".\text\wenzi.txt"
input_ids,en_tokenizer = Dataset_txt(path_to_file)
input_ids = tf.convert_to_tensor(input_ids)
speaker_ids = tf.convert_to_tensor([0] * batch_size, tf.int32)
path = r"./wavs/"
mel_gts = Dataset_wave(path)
mel_gts = tf.cast(mel_gts, tf.float32)
BUFFER_SIZE = len(input_ids)
steps_per_epoch = BUFFER_SIZE // batch_size
dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)
input_ids, mel_gts = next(iter(dataset))
# 模型
class Encoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size, config):
        super(Encoder, self).__init__()
        self.num_filters = config.encoder_conv_filters
        self.kernel_size = config.encoder_conv_kernel_sizes
        self.lstm_unit = config.encoder_lstm_units
        self.rate = config.encoder_conv_dropout_rate
        self.vocab_size = vocab_size
        self.embedding_dim = config.embedding_hidden_size
        # 定义嵌入层
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        # 定义三层卷积层
        self.conv1d1 = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size,padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.output1 = tf.keras.layers.BatchNormalization()

        self.conv1d2 = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size,padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.output2 = tf.keras.layers.BatchNormalization()

        self.conv1d3 = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size,padding='same', activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        self.output3 = tf.keras.layers.BatchNormalization()
        # 定义两次LSTM
        self.forward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True)
        self.backward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, activation='relu', return_sequences=True,
                                                   go_backwards=True)
        self.bidir = tf.keras.layers.Bidirectional(layer=self.forward_layer,
                                                   backward_layer=self.backward_layer)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1d1(x)
        x = self.dropout1(x)
        x = self.output1(x)
        x = self.conv1d2(x)
        x = self.dropout2(x)
        x = self.output2(x)
        x = self.conv1d3(x)
        x = self.dropout3(x)
        x = self.output3(x)
        output = self.bidir(x)
        return output

vocab_inp_size = len(en_tokenizer.word_index)+1
encoder = Encoder(vocab_inp_size,config)
encode_output= encoder(input_ids)


class LocationLayer(tf.keras.layers.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding="same",
            use_bias=False,
            name="location_conv",
        )
        self.location_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, activation="tanh", name="location_layer"
        )

    def __call__(self, attention_weights_cum):
        attention_weights_cum = tf.expand_dims(attention_weights_cum, 1)
        processed_attention_weights = self.location_convolution(attention_weights_cum)
        processed_attention_weights = self.location_layer(processed_attention_weights)
        return processed_attention_weights

class Attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attention_rnn_dim=config.attention_dim
        self.attention_dim=config.attention_dim
        self.attention_location_n_filters=config.attention_filters
        self.attention_location_kernel_size=config.attention_kernel
        self.query_layer = tf.keras.layers.Dense(self.attention_rnn_dim, activation="tanh")
        self.memory_layer = tf.keras.layers.Dense(self.attention_rnn_dim, activation="tanh")
        self.V = tf.keras.layers.Dense(1)
        self.location_layer = LocationLayer(self.attention_location_n_filters,self.attention_location_kernel_size,self.attention_rnn_dim)
        self.score_mask_value = -float("inf")
    def get_alignment_energies(self, query, memory, attention_weights_cat):
        processed_memory=tf.keras.layers.Dense(self.attention_rnn_dim, activation="tanh")(memory)
        processed_query = self.query_layer(query)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = tf.squeeze(self.V(tf.nn.tanh(processed_query+processed_memory+processed_attention_weights)), -1)
        return energies

    def __call__(self, attention_hidden_state, memory, attention_weights_cat):
        alignment = self.get_alignment_energies(
            attention_hidden_state, memory, attention_weights_cat)
        attention_weights = tf.nn.softmax(alignment, axis=1)
        attention_context = tf.expand_dims(attention_weights, 2) * memory
        attention_context = tf.squeeze(attention_context)
        return attention_context, attention_weights

class Prenet(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()
        self.prenet_units = config.prenet_units
        self.n_prenet_layers = config.n_prenet_layers
        self.prenet_dropout_rate = config.prenet_dropout_rate
        self.prenet_dense = [
            tf.keras.layers.Dense(
                units=self.prenet_units,
                activation='relu'
            )
            for i in range(self.n_prenet_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(
            rate=self.prenet_dropout_rate
        )
        def call(self, inputs):
            outputs = inputs
            for layer in self.prenet_dense:
                outputs = layer(outputs)
                outputs = self.dropout(outputs)
            return outputs


class Postnet(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()
        self.conv1d1 = tf.keras.layers.Conv1D(
            filters=config.postnet_conv_filters,
            kernel_size=config.n_conv_postnet,
            activation="tanh",
            padding="same"

        )

        self.norm1 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.conv1d2 = tf.keras.layers.Conv1D(
            filters=config.postnet_conv_filters,
            kernel_size=config.n_conv_postnet,
            activation="tanh",
            padding="same"
        )
        self.norm2 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.dropout2 = tf.keras.layers.Dropout(rate=config.postnet_dropout_rate)
        self.conv1d3 = tf.keras.layers.Conv1D(
            filters=config.postnet_conv_filters,
            kernel_size=config.n_conv_postnet,
            activation="tanh",
            padding="same"

        )
        self.norm3 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.dropout3 = tf.keras.layers.Dropout(rate=config.postnet_dropout_rate)
        self.conv1d4 = tf.keras.layers.Conv1D(
            filters=config.postnet_conv_filters,
            kernel_size=config.n_conv_postnet,
            activation="tanh",
            padding="same"
        )
        self.norm4 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.dropout4 = tf.keras.layers.Dropout(rate=config.postnet_dropout_rate)
        self.conv1d5 = tf.keras.layers.Conv1D(
            filters=config.postnet_conv_filters,
            kernel_size=config.n_conv_postnet,
            activation=None,
            padding="same"
        )
        self.norm5 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.dropout5 = tf.keras.layers.Dropout(rate=config.postnet_dropout_rate)
        self.fc= tf.keras.layers.Dense( units=config.n_mels, name="frame_projection1")

    def call(self, inputs):
        x = inputs
        x = self.conv1d1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.conv1d2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.conv1d3(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.conv1d4(x)
        x = self.norm4(x)
        x = self.dropout4(x)
        x = self.conv1d5(x)
        x = self.norm5(x)
        x = self.dropout5(x)
        x = self.fc(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self,config):

        super(Decoder, self).__init__()
        self.prenet2=Prenet(config)
        self.postnet = Postnet(config)
        # 两个单层LSTM
        self.decoder_lstms1 = tf.keras.layers.LSTM(config.decoder_lstm_dim,dropout=config.decoder_lstm_rate,return_sequences=True)
        self.decoder_lstms2 = tf.keras.layers.LSTM(config.decoder_lstm_dim,dropout=config.decoder_lstm_rate,return_sequences=True)
        # 线性变换投影成目标帧
        self.frame_projection = tf.keras.layers.Dense(
            units=config.n_mels, name="frame_projection"
        )
        # 停止记号
        self.stop_projection = tf.keras.layers.Dense(
            units=1, name="stop_projection"
        )
        # 用于注意力
        self.attention_layer = Attention(config)

    def initialize_decoder_states(self, memory):
        B =tf.shape(memory)[0]
        MAX_TIME = tf.shape(memory)[1]
        C=int(tf.shape(memory)[2]/2)
        self.attention_weights = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_weights_cum = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.context_vector = tf.zeros(shape=[B, MAX_TIME ,C], dtype=tf.float32)
        self.memory = memory
        return self.context_vector

    def call(self, dec_input,enc_output,context_vector,time):
        self.context_vector=context_vector
        x1 = tf.transpose(dec_input,[0,2,1])
        x1 = tf.keras.layers.Dense(enc_output.shape[1],activation="relu")(x1)
        x = tf.transpose(x1, [0, 2, 1])
        x = tf.keras.layers.Dense(256,activation="relu")(x)
        x = self.prenet2(x)
        x = tf.concat([self.context_vector, x], axis=-1)
        # 通过双层LSTM
        attention_hidden_state =  tf.keras.layers.LSTM(1024,dropout=0.1,return_sequences=True)(x)
        self.context_vector, self.attention_weights = self.attention_layer(attention_hidden_state, enc_output, self.attention_weights_cum)
        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights
        x = tf.concat([self.context_vector, attention_hidden_state], axis=-1)
        x = self.decoder_lstms2(x)
        x = tf.concat((self.context_vector, x), axis=-1)
        # 投影出目标频谱
        x1 = self.frame_projection(x)
        # 投影出stop-token
        stop_token = self.stop_projection(x)
        # 预测残差
        x3 = self.postnet(x1)
        decoder_output=x1
        mel_output = x3 + x1
        x1 = tf.transpose(mel_output, [0, 2, 1])
        x1 = tf.keras.layers.Dense(500, activation="relu")(x1)
        mel_output = tf.transpose(x1, [0, 2, 1])
        time = time+1
        return decoder_output,mel_output, stop_token, self.context_vector,time

dec_input = tf.expand_dims([en_tokenizer.word_index['<start>']] * batch_size, 1)
dec_input = tf.expand_dims(dec_input, 2)
time=1
decoder = Decoder(config)
if time == 1:
    context_vector = decoder.initialize_decoder_states(encode_output)
decoder_output,mel_outputs, gate_outputs, context_vector,time = decoder(dec_input,encode_output,context_vector,time)
print("以上是第一次跑 确定可以")
dec_input=decoder_output
context_vector=context_vector
decoder_output,mel_outputs, gate_outputs, context_vector,time = decoder(dec_input,encode_output,context_vector,time)
print("以上是第二次跑 确定可以循环的")
exit(0)