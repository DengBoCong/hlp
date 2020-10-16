import config
import os
from utils import get_index_and_char_map, text_process


if __name__ == "__main__":
    #当前预处理的是训练数据集和测试数据集里的两个文本文件，并构建出字符集
    data_path = config.configs_other()["text_preprocess_path"]

    char_set_path = config.configs_other()["char_set_path"]
    if not os.path.exists(char_set_path):
        f = open(char_set_path,"w")
        f.close()
    #获取index_map和char_map
    index_map,char_map = get_index_and_char_map()
    #基于训练集和测试集的文本数据构建字符集
    for path in data_path:
        files = os.listdir(path) #得到文件夹下的所有文件名称
        text_path = files[len(files)-1]
        with open(path+"/"+text_path,"r") as f:
            text_list = f.readlines()
        with open(char_set_path,"a") as f:
            for i in range(len(text_list)):
                for ch in text_process(text_list[i]):
                    if ch not in index_map.values():
                        index = len(index_map) + 1
                        index_map[index] = ch
                        if ch == " ":
                            char_map["<space>"] = index
                            f.write("<space>" + " " + str(index) + "\n")
                        else:
                            char_map[ch] = index
                            f.write(ch + " " + str(index) + "\n")
