import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention =