import sys
from model.chatter import Chatter
from common.utils import CmdParser

class SMNChatter(Chatter):
    """
    SMN的聊天器
    """
    def __init__(self, checkpoint_dir, beam_size, max_length):
        super().__init__(checkpoint_dir, beam_size, max_length)

def get_chatter(excute_type):
    chatter = SMNChatter()

    return

def main():
    parser = CmdParser(version='%smn chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        print("待完善")
        # chatter = get_chatter(execute_type=options.type)
        # chatter.train(chatter.checkpoint,
        #               dict_fn=_config.transformer_dict_fn,
        #               data_fn=_config.lccc_tokenized_data,
        #               max_train_data_size=_config.transformer_max_train_data_size)
    elif options.type == 'chat':
        print("待完善")
        # chatter = get_chatter(options.type)
        # print("Agent: 你好！结束聊天请输入ESC。")
        # while True:
        #     req = input("User: ")
        #     if req == "ESC":
        #         print("Agent: 再见！")
        #         exit(0)
        #     response = chatter.respond(req=req)
        #     print("Agent: ", response)
    else:
        parser.error(msg='')

if __name__ == '__main__':
    """
    SMN入口：指令需要附带运行参数
    cmd：python smn_chatter.py -t/--type [执行模式]
    执行类别：train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()