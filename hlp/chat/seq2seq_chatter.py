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

    def __init__(self, checkpoint_dir):
        """
        Seq2Seq聊天器初始化，用于加载模型
        """
        super().__init__(checkpoint_dir)
        if self.ckpt:
            seq2seq.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    def respond(self, req):
        # 对req进行初步处理
        inputs, dec_input = self.pre_treat_inputs(req)
        result = ''
        # 初始化隐藏层，并使用encoder得到隐藏层和decoder的输入给decoder使用
        hidden = [tf.zeros((1, _config.units))]
        enc_out, enc_hidden = seq2seq.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        for t in range(_config.max_length_tar):
            predictions, dec_hidden, attention_weights = seq2seq.decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            # 这里就是做一下判断，当预测结果解码是end，说明预测结束
            if self.target_token.index_word.get(predicted_id) == 'end':
                break
            # 把相关结果拼接起来
            result += self.target_token.index_word.get(predicted_id, '')
            # 这里就是更新一下decoder的输入，因为要将前面预测的结果反过来
            # 作为输入丢到decoder里去
            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def train(self):
        """
        Seq2Seq的训练方法，直接通过
        """
        print('训练开始，正在准备数据中...')
        steps_per_epoch = len(self.input_tensor) // _config.BATCH_SIZE
        if self.ckpt:
            seq2seq.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()
        dataset, checkpoint_prefix = self.treat_dataset()
        start_time = time.time()

        for epoch in range(_config.epochs):
            print('当前训练epoch为：{}'.format(epoch + 1))
            enc_hidden = seq2seq.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = seq2seq.train_step(inp, targ, self.target_token, enc_hidden)
                total_loss += batch_loss
                print('当前batch的损失：{:.4f}'.format(batch_loss.numpy()))
            step_loss = total_loss / steps_per_epoch
            step_time = (time.time() - start_time) / (epoch + 1)
            print('epochs平均耗时：{:.4f}s'.format(step_time))
            print('当前epoch的损失为：{:.4f}'.format(step_loss.numpy()))
            seq2seq.checkpoint.save(file_prefix=checkpoint_prefix)
            sys.stdout.flush()

        print('训练结束')


def main():
    parser = OptionParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    # 初始化要使用的聊天器
    chatter = Seq2SeqChatter(checkpoint_dir=_config.seq2seq_train_data)

    if options.type == 'train':
        chatter.train()
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
