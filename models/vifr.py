## VIFR model contains Age Estimation TD module
## Moreover, an MTL estimation model for TID tasks
## Date: 14 Jan 2023

from . import TDBlock, TIDBlock

class VIFR(object):
    def __init__(self, opt):
        self.opt = opt
        self.td_block = TDBlock(self.opt)
        self.mtl_tid_block = TIDBlock(self.opt)

    def train():
        pass
    def evaluate():
        pass
