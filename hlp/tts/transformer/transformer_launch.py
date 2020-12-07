from optparse import OptionParser


class CmdParser(OptionParser):
    def error(self, msg):
        print('Error!提示信息如下：')
        self.print_help()
        self.exit(0)

    def exit(self, status=0, msg=None):
        exit(status)


if __name__ == '__main__':
    """
    Transformer TTS入口：指令需要附带运行参数
    cmd：python transformer_launch.py -t/--type [执行模式]
    执行类别：pre_treat/train/generate

    generate模式下运行时，输入ESC即退出语音合成
    """
    parser = CmdParser(version='transformer tts V1.0.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/generate")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        print("待完善")
    elif options.type == 'chat':
        print("待完善")
        # print("Agent: 你好！结束合成请输入ESC。")
        # while True:
        #     req = input("Sentence: ")
        #     if req == "ESC":
        #         print("Agent: 再见！")
        #         exit(0)
    elif options.type == 'pre_treat':
        print("待完善")
    else:
        parser.error(msg='')
