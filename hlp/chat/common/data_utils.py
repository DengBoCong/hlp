

def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


class MyDataset():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        print('')

    def __len__(self):
        return len(self.data)
    # 抽象类


def loadDataset(filename):
    print('')
    # local
    # url
    # 数据加载模块，下一个issue进行完善
