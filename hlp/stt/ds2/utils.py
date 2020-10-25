import json


#获取配置文件
def get_config():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs

#写入配置文件
def set_config(config_class, key, value):
    configs = get_config()
    configs[config_class][key] = value
    with open("config.json", 'w', encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

#获取字典集合
def get_index_word():
    configs = get_config()
    index_word_path = configs["other"]["index_word_path"]
    with open(index_word_path, "r", encoding="utf-8") as f:
        index_word = json.load(f)
    return index_word


if __name__ == "__main__":
    pass