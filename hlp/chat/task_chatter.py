import sys
from model.task.kb import load_kb
from model.chatter import Chatter
from optparse import OptionParser
import config.get_config as _config


class TaskChatter(Chatter):
    """
    Task模型的聊天器
    """

    def __init__(self, checkpoint_dir, beam_size):
        super().__init__(checkpoint_dir, beam_size)
        if self.ckpt:
            print('s')

    def init_loss_accuracy(self):
        print('d')

    def train_step(self, inp, tar, step_loss):
        print('d')

    def create_predictions(self, inputs, dec_input, t):
        print('s')


def main():
    parser = OptionParser(version='%task chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    chatter = TaskChatter(checkpoint_dir=_config.task_train_data, beam_size=_config.beam_size)

    if options.type == 'train':
        # chatter.train()
        kb = load_kb(_config.cam_rest, "name")
        print(kb.index['food']['italian'])
        print(kb.search_multi([[u'food', u'swedish'], [u'pricerange', u'cheap']]))
    elif options.type == 'chat':
        print('Agent: 你好！结束聊天请输入ESC。')
        while True:
            req = input('User: ')
            if req == 'ESC':
                chatter.stop()
                print('Agent: 再见！')
                exit(0)
            # response = chatter.respond(req)
            response = '待完善'
            print('Agent: ', response)
    elif options.type == 'pre_treat':
        print('待完善')
    else:
        print('Error:不存在', sys.argv[2], '模式!')


if __name__ == "__main__":
    """
    TaskModel入口：指令需要附带运行参数
    cmd：python task_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()
