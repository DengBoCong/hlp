import os
import sys
import time
import tensorflow as tf
from model.chatter import Chatter
from optparse import OptionParser
import config.get_config as _config
import model.seq2seq.model as seq2seq
from common.pre_treat import preprocess_raw_data


class Seq2SeqChatter(Chatter):
    """
    Seq2Seq模型的聊天类
    """

    def __init__(self, checkpoint_dir, beam_size):
        """
        Seq2Seq聊天器初始化，用于加载模型
        """
        super().__init__(checkpoint_dir, beam_size)
        if self.ckpt:
            seq2seq.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    def train_step(self, inp, tar, step_loss):
        enc_hidden = seq2seq.encoder.initialize_hidden_state()
        step_loss[0] += seq2seq.train_step(inp, tar, self.target_token, enc_hidden)

    def create_predictions(self, inputs, dec_input, t):
        hidden = tf.zeros((inputs.shape[0], _config.units))
        enc_out, enc_hidden = seq2seq.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(dec_input[:, t], 1)
        predictions, _, _ = seq2seq.decoder(dec_input, dec_hidden, enc_out)
        self.beam_search_container.add(predictions)


def main():
    parser = OptionParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    # 初始化要使用的聊天器
    chatter = Seq2SeqChatter(checkpoint_dir=_config.seq2seq_train_data, beam_size=_config.beam_size)

    if options.type == 'train':
        chatter.train(seq2seq.checkpoint)
    elif options.type == 'chat':
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                chatter.stop()
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req)
            print("Agent: ", response)
    elif options.type == 'pre_treat':
        preprocess_raw_data()
    else:
        print('Error:不存在', sys.argv[2], '模式!')


if __name__ == "__main__":
    """
    Seq2Seq入口：指令需要附带运行参数
    cmd：python seq2seq2_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入ESC即退出对话
    """
    main()
