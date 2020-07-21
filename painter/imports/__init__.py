from .torchpacks import *
from .core import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")