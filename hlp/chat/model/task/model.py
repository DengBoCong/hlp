import config.get_config as _config
from common.layers import InformSlotTracker
from common.layers import RequestSlotTracker


def gen_tracker_model_and_loss(onto, embedding, conf, kn):
    state_tracker_hidden_size = _config.state_tracker_hidden_size
    slot_trackers = {}
    slot_len_sum = 0

    for slot in onto:
        if len(onto[slot]) > 2:
            slot_trackers[slot] = InformSlotTracker(n_choices=len(onto[slot]))
            slot_len_sum += len(onto[slot]) + 1
        else:
            slot_trackers[slot] = RequestSlotTracker()
            slot_len_sum += 2
