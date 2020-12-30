import os
from pathlib import Path

from hlp.mt.config import get_config as _config


def check_and_create(checkpoint_dir=_config.checkpoint_path):
    chkpt_path = Path(checkpoint_dir)
    if not chkpt_path.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
        return False
    else:
        return True