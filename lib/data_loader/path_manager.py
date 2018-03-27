import os

class PathManager:
    def __init__(self, base):
        self.base = base

    @property
    def raw_info(self):
        return os.path.join(self.base, "data1.csv")

    @property
    def npz_save(self):
        return os.path.join(self.base, "npz")


DATASET_BASE = "/media/dn/doc/zhanshiyuan/data"
PATH = PathManager(DATASET_BASE)  # singleton
