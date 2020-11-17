import sys
import tensorflow as tf
import common.data_utils as data_utils
sys.path.append(sys.path[0][:-10])
from model.chatter import Chatter
import model.seq2seq as seq2seq
from common.utils import CmdParser
import config.get_config as get_config
from common.pre_treat import dispatch_tokenized_func_dict_single


class Seq2SeqChatter(Chatter):
    """
    Seq2Seq模型的聊天类
    """

    def __init__(self, execute_type: str, checkpoint_dir: str, units: int, embedding_dim: int, batch_size: int,
                 start_sign: str, end_sign: str, beam_size: int, vocab_size: int, dict_fn: str, max_length: int):
        """
        Seq2Seq聊天器初始化，用于加载模型
        Args:
            execute_type: 对话执行模式
            checkpoint_dir: 检查点保存目录路径
            units: 单元数
            embedding_dim: 嵌入层维度
            batch_size: batch大小
            start_sign: 开始标记
            end_sign: 结束标记
            beam_size: batch大小
            vocab_size: 词汇量大小
            dict_fn: 保存字典路径
            max_length: 单个句子最大长度
        Returns:
        """
        super().__init__(checkpoint_dir, beam_size, max_length)
        self.units = units
        self.start_sign = start_sign
        self.end_sign = end_sign
        self.batch_size = batch_size

        self.encoder = seq2seq.Encoder(vocab_size, embedding_dim, units,
                                       batch_size)
        self.decoder = seq2seq.Decoder(vocab_size, embedding_dim, units,
                                       batch_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

        if execute_type == "chat":
            print('正在从“{}”处加载字典...'.format(dict_fn))
            self.token = data_utils.load_token_dict(dict_fn=dict_fn)
        print('正在检查是否存在检查点...')
        if self.ckpt:
            print('存在检查点，正在从“{}”中加载检查点...'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式...')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

    def _train_step(self, inp: tf.Tensor, tar: tf.Tensor, weight: int, step_loss: float):
        """
        Args:
            inp: 输入序列
            tar: 目标序列
            weight: 样本权重序列
            step_loss: 每步损失
        Returns:
            step_loss: 每步损失
        """
        loss = 0
        enc_hidden = self.encoder.initialize_hidden_state()

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            # 这里初始化decoder的输入，首个token为start，shape为（128, 1）
            dec_input = tf.expand_dims([2] * self.batch_size, 1)
            # 这里针对每个训练出来的结果进行损失计算
            for t in range(1, tar.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self._loss_function(tar[:, t], predictions, weight)
                # 这一步使用teacher forcing
                dec_input = tf.expand_dims(tar[:, t], 1)

        batch_loss = (loss / int(tar.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        step_loss += batch_loss
        return step_loss

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        获取目前已经保存在容器中的序列
        Args:
            inputs: 对话中的问句
            dec_input: 对话中的答句
            t: 记录时间步
        Returns:
            predictions: 预测
        """
        hidden = tf.zeros((inputs.shape[0], self.units))
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(dec_input[:, t], 1)
        predictions, _, _ = self.decoder(dec_input, dec_hidden, enc_out)
        return predictions

    def _loss_function(self, real: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor):
        """
        用于计算预测损失，注意要将填充的0进行mask，不纳入损失计算
        Args:
            real: 真实序列
            pred: 预测序列
            weights: 样本数据的权重
        Returns:
            loss: 该batch的平均损失
        """
        # 这里进来的real和pred的shape为（128,）
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred, sample_weight=weights)
        # 这里要注意了，因为前面我们对于短的句子进行了填充，所
        # 以对于填充的部分，我们不能用于计算损失，所以要mask
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


def get_chatter(execute_type: str):
    """
    初始化要使用的聊天器
    Args:
        execute_type: 对话执行模型
    Returns:
        chatter: 返回对应的聊天器
    """
    chatter = Seq2SeqChatter(execute_type=execute_type, checkpoint_dir=get_config.seq2seq_checkpoint,
                             beam_size=get_config.beam_size, units=get_config.seq2seq_units,
                             embedding_dim=get_config.seq2seq_embedding_dim, batch_size=get_config.BATCH_SIZE,
                             start_sign=get_config.start_sign, end_sign=get_config.end_sign,
                             vocab_size=get_config.seq2seq_vocab_size, dict_fn=get_config.seq2seq_dict_fn,
                             max_length=get_config.seq2seq_max_length)
    return chatter


def main():
    parser = CmdParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type)
        chatter.train(chatter.checkpoint,
                      dict_fn=get_config.seq2seq_dict_fn,
                      data_fn=get_config.lccc_tokenized_data,
                      max_train_data_size=get_config.seq2seq_max_train_data_size,
                      epochs=get_config.epochs)
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
        dispatch_tokenized_func_dict_single(operator="lccc", raw_data=get_config.lccc_data,
                                            tokenized_data=get_config.lccc_tokenized_data, if_remove=True)
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
