import os
from pathlib import Path


def check_and_create(checkpoint_dir):
    chkpt_path = Path(checkpoint_dir)
    if not chkpt_path.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
        return False
    else:
        return True