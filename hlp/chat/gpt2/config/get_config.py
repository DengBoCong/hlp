import json


def get_config(filename="config/config.json"):
    with open(filename) as file:
        model_config = json.load(file)
    return model_config


if __name__ == '__main__':
    config = get_config()
    print(config["epochs"])
