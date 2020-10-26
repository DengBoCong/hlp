import sys
import tensorflow as tf
sys.path.append(sys.path[0][:-10])
from model.chatter import Chatter
import model.seq2seq as seq2seq
from common.utils import CmdParser
import config.get_config as _config
from common.pre_treat import preprocess_raw_lccc_data


class Seq2SeqChatter(Chatter):
    """
    Seq2Seq模型的聊天类
    """

    def __init__(self, model, checkpoint_dir, beam_size, vocab_size):
        """
        Seq2Seq聊天器初始化，用于加载模型
        """
        super().__init__(model, checkpoint_dir, beam_size)
        self.encoder = seq2seq.Encoder(vocab_size, _config.embedding_dim, _config.units, _config.BATCH_SIZE)
        self.decoder = seq2seq.Decoder(vocab_size, _config.embedding_dim, _config.units, _config.BATCH_SIZE)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

        print('正在检查是否存在检查点...')
        if self.ckpt:
            print('存在检查点，正在从{}中加载检查点'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            print('不存在检查点，请先执行train模式，再进入chat模式')
            if model == 'chat':
                exit(0)

    def _train_step(self, inp, tar, step_loss):
        loss = 0
        enc_hidden = self.encoder.initialize_hidden_state()

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            # 这里初始化decoder的输入，首个token为start，shape为（128, 1）
            dec_input = tf.expand_dims([2] * _config.BATCH_SIZE, 1)
            # 这里针对每个训练出来的结果进行损失计算
            for t in range(1, tar.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self._loss_function(tar[:, t], predictions)
                # 这一步使用teacher forcing
                dec_input = tf.expand_dims(tar[:, t], 1)

        batch_loss = (loss / int(tar.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        step_loss[0] += batch_loss

    def _create_predictions(self, inputs, dec_input, t):
        hidden = tf.zeros((inputs.shape[0], _config.units))
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(dec_input[:, t], 1)
        predictions, _, _ = self.decoder(dec_input, dec_hidden, enc_out)
        return predictions

    def _loss_function(self, real, pred):
        """
        :param real:
        :param pred:
        :return: loss
        """
        # 这里进来的real和pred的shape为（128,）
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        # 这里要注意了，因为前面我们对于短的句子进行了填充，所
        # 以对于填充的部分，我们不能用于计算损失，所以要mask
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


def get_chatter(model):
    # 初始化要使用的聊天器
    chatter = Seq2SeqChatter(model=model,
                             checkpoint_dir=_config.seq2seq_train_data,
                             beam_size=_config.beam_size,
                             vocab_size=_config.vocab_size,
                             dict_fn=_config.seq2seq_dict_fn)
    return chatter


def main():
    parser = CmdParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(options.type)
        chatter.train(chatter.checkpoint,
                      dict_fn=_config.seq2seq_dict_fn,
                      data_fn=_config.data,
                      max_train_data_size=_config.max_train_data_size)
    elif options.type == 'chat':
        chatter = get_chatter(options.type)
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    elif options.type == 'pre_treat':
        preprocess_raw_lccc_data(raw_data=_config.transformer_lccc_data,
                                 tokenized_data=_config.transformer_lccc_tokenized_data)
        # preprocess_raw_data(raw_data=_config.resource_data, tokenized_data=_config.tokenized_data)
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    Seq2Seq入口：指令需要附带运行参数
    cmd：python seq2seq2_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入ESC即退出对话
    """
    main()
