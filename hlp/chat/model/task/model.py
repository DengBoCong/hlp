import json
import config.get_config as _config
from common.layers import InformSlotTracker
from common.layers import RequestSlotTracker

sent_groups = {}

def main():
    global sent_groups

    # 获取本体信息
    with open(_config.sent_groups) as file:
        sent_groups = json.load(file)
