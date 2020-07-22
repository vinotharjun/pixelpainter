from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import types
import numpy as np
import collections
import pandas as pd
from random import shuffle