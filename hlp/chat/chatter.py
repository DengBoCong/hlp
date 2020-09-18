""""面向使用者的聊天器基类

该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
不同模型或方法实现的聊天子类化该类。
"""


class Chatter(object):
    def respond(self, req):
        """ 对外部聊天请求进行回复

        子类可能需要利用模型进行推断和搜索以产生回复。

        :param req: 外部聊天请求字符串
        :return: 系统回复字符串
        """
        pass

    def stop(self):
        """ 结束聊天

        可以做一些清理工作
        :return:
        """
        pass
