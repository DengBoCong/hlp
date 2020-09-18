"""聊天器使用例子
"""

from Chatter import Chatter

class CopyChatter(Chatter):
    """ 简单重复外部聊天请求的聊天器

    本类只为了演示使用。
    """
    def __init__(self):
        """ 这里可以加载模型
        """
        pass

    def respond(self, req):
        return req


def main():
    chatter = CopyChatter()  # 初始化要使用的聊天器
    print("Agent: 你好！结束聊天请输入ESC。")
    while True:
        req = input("User: ")
        response = chatter.respond(req)
        if response=="ESC":
            chatter.stop()
            print("Agent: 再见！")
            exit(0)
        print("Agent: ", response)


if __name__ == "__main__":
    main()