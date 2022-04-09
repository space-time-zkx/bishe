import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import GenericMLP
from models.position_encoding import PositionEmbeddingSine,MLP,PositionEmbeddingCoordsSine,NerfPositionalEncoding
from models.transformer import (TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer,Transformer)