import logging
import os
import time
import numpy as np
import tensorflow as tf
from plot.plot import plot_mel
from dataset.dataset_wav import Dataset_wave
from dataset.dataset_txt import Dataset_txt
from config.config import Tacotron2Config
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 自己的模型
class Encoder(tf.keras.Model):
    def __init__(self,vocab_size, config):
        # self.num_filters=512,kernel_size=5,rate=0.5,self.lstm_unit=256
        super(Encoder, self).__init__()
        # self.batch_sz = batch_sz
        # self.enc_units = enc_units
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
        self.backward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True,
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

class LocationLayer(tf.keras.Model):
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



    def call(self, attention_weights_cat):
        processed_attention = self.location_convolution(attention_weights_cat)
        processed_attention = tf.transpose(processed_attention, [0, 2, 1])
        processed_attention = self.location_layer(processed_attention)
        return processed_attention

class Attention(tf.keras.Model):
        def __init__(self, config):
            super(Attention, self).__init__()
            self.attention_rnn_dim = config.attention_dim
            self.attention_dim = config.attention_dim
            self.attention_location_n_filters = config.attention_filters
            self.attention_location_kernel_size = config.attention_kernel

            self.query_layer = tf.keras.layers.Dense(self.attention_rnn_dim,use_bias=False, activation="tanh")
            self.memory_layer = tf.keras.layers.Dense(self.attention_rnn_dim, use_bias=False,activation="tanh")
            self.V = tf.keras.layers.Dense(1,use_bias=False)
            self.location_layer = LocationLayer(self.attention_location_n_filters, self.attention_location_kernel_size,
                                                self.attention_rnn_dim)

            self.score_mask_value = -float("inf")

        def get_alignment_energies(self, query, memory, attention_weights_cat):
            """
            PARAMS
            ------
            query: decoder output (batch, n_mel_channels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, attention_dim)
            attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
            RETURNS
            -------
            alignment (batch, max_time)
            """
            processed_query = self.query_layer(tf.expand_dims(query,axis=1))
            processed_memory = self.memory_layer(memory)
            processed_attention_weights = self.location_layer(attention_weights_cat)
            energies = tf.squeeze(self.V(tf.nn.tanh(processed_query + processed_memory + processed_attention_weights)),-1)
            return energies


        def __call__(self, attention_hidden_state, memory, attention_weights_cat):
             """
             PARAMS
         ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
             alignment = self.get_alignment_energies(
                attention_hidden_state, memory, attention_weights_cat)
             attention_weights = tf.nn.softmax(alignment, axis=1)
             attention_context = tf.expand_dims(attention_weights, 1)
             attention_context =tf.matmul(attention_context,memory)
             attention_context = tf.squeeze(attention_context,axis=1)
             return attention_context, attention_weights
#attention结束
class Prenet(tf.keras.Model):

    def __init__(self,config,):
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

    def __call__(self, inputs):
        """Call logic."""
        outputs = inputs
        for layer in self.prenet_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs)
        return outputs


class Postnet(tf.keras.Model):
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
        self.fc= tf.keras.layers.Dense( units=500, activation=None,name="frame_projection1")

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


class Decoder(tf.keras.Model):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.attention_dim=config.attention_dim
        self.decoder_lstm_dim=config.decoder_lstm_dim
        self.embedding_hidden_size=config.embedding_hidden_size
        self.gate_threshold = config.gate_threshold
        self.n_mels=config.n_mels
        self.prenet2=Prenet(config)
        self.postnet = Postnet(config)
        # 两个单层LSTM
        self.decoder_lstms1 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim,dropout=config.decoder_lstm_rate)
        self.decoder_lstms2 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim,dropout=config.decoder_lstm_rate)
        # 线性变换投影成目标帧
        self.frame_projection = tf.keras.layers.Dense(
            units=self.n_mels,activation=None, name="frame_projection"
        )
        # 停止记号
        self.stop_projection = tf.keras.layers.Dense(
            units=1,activation='sigmoid', name="stop_projection"
        )
        # 用于注意力
        self.attention_layer = Attention(config)

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = tf.shape(memory)[0]
        decoder_input =tf.zeros(shape = [B, self.n_mels], dtype = tf.float32)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        """
        B = tf.shape(memory)[0]
        MAX_TIME = tf.shape(memory)[1]

        self.attention_hidden =  tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.attention_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.decoder_hidden = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.decoder_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.attention_weights = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_weights_cum = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_context = tf.zeros(shape=[B, self.embedding_hidden_size], dtype=tf.float32)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = tf.transpose(decoder_inputs,(0,2,1))
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = tf.transpose(decoder_inputs, (1, 0, 2))
        decoder_inputs = tf.transpose(decoder_inputs, (2, 1, 0))

        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = tf.stack(alignments)
        alignments=tf.transpose(alignments,(1,0,2))
        # (T_out, B) -> (B, T_out)
        gate_outputs=tf.stack(gate_outputs)
        gate_outputs=tf.transpose(gate_outputs,(1,0))
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = tf.stack( mel_outputs)
        mel_outputs=tf.transpose(mel_outputs,(1,0,2))
        # decouple frames per step
        mel_outputs = tf.reshape(mel_outputs,(mel_outputs.shape[0], -1, self.n_mels))
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = tf.transpose(mel_outputs,(0,2,1))

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
               PARAMS
               ------
               decoder_input: previous mel output
               RETURNS
               -------
               mel_output:
               gate_output: gate output energies
               attention_weights:
               """
        cell_input =  tf.concat((decoder_input, self.attention_context), -1)
        B = tf.shape(decoder_input)[0]
        #第一次lstmcell
        self.attention_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)
        self.attention_hidden, self.attention_cell = self.decoder_lstms1(cell_input, (self.attention_hidden, self.attention_cell))
        #拼接
        attention_weights_cat =tf.concat(((tf.expand_dims(self.attention_weights,axis=1)),(tf.expand_dims(self.attention_weights_cum,axis=1))),axis=1)
        #注意力
        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory,attention_weights_cat)
        self.attention_weights_cum += self.attention_weights
        #拼接
        decoder_input = tf.concat((self.attention_hidden, self.attention_context), -1)
        # 第2次lstmcell
        self.decoder_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)
        self.decoder_hidden, self.decoder_cell =  self.decoder_lstms2(decoder_input, (self.decoder_hidden, self.decoder_cell))
        #拼接
        decoder_hidden_attention_context = tf.concat((self.decoder_hidden, self.attention_context), axis=1)
        #投影梅尔频谱
        decoder_output = self.frame_projection(decoder_hidden_attention_context)
        #投影stop_token
        gate_prediction = self.stop_projection(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def call(self, memory, decoder_inputs,memory_lengths):
        """ Decoder forward pass for training
               PARAMS
               ------
               memory: Encoder outputs
               decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
               memory_lengths: Encoder output lengths for attention masking.
               RETURNS
               -------
               mel_outputs: mel outputs from the decoder
               gate_outputs: gate outputs from the decoder
               alignments: sequence of attention weights from the decoder
               """
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = self.prenet2(decoder_inputs)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.shape[0]:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [tf.squeeze(mel_output)]
            gate_outputs += [tf.squeeze(gate_output)]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
             mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [tf.squeeze(mel_output,axis=1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            if tf.nn.sigmoid(gate_output.data) > self.gate_threshold:
                break
            decoder_input = mel_output
            mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
                mel_outputs, gate_outputs, alignments)
            return mel_outputs, gate_outputs, alignments

class Tacotron2(tf.keras.Model):
    def __init__(self, vocab_inp_size,config):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(vocab_inp_size,config)
        self.decoder = Decoder(config)
        self.postnet = Postnet(config)

    def call(self, inputs,mel_gts):
        encoder_outputs = self.encoder(inputs)
        memory_lengths = encoder_outputs.shape[1]
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_gts, memory_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
