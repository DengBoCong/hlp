#字符集类
class Char_set():
    def __init__(self,data_path):
        self.data_path = data_path
        self.char_map = self.get_map()[0]
        self.index_map = self.get_map()[1]
    def get_map(self):
        char_map={}
        index_map={}
        with open(self.data_path,"r") as f:
            str_list = f.readlines()
        for i in range(len(str_list)):
            index_and_char = str_list[i].strip().split()
            char_map[index_and_char[1]] = int(index_and_char[0])
            if index_and_char[1] == "<space>":
                index_map[int(index_and_char[0])] = " "
            else:
                index_map[int(index_and_char[0])] = index_and_char[1]
        return (char_map,index_map)
    def add_char(self,ch):
        index = len(self.index_map) + 1
        self.index_map[index] = ch
        if ch == " ":
            self.char_map["<space>"] = index
        else:
            self.char_map[ch] = index
        with open(self.data_path,"a") as f:
            if ch == " ":
                f.write(str(index) + " " + "<space>" +"\n")
            else:
                f.write(str(index) + " " + ch +"\n")

if __name__ == "__main__":
    pass