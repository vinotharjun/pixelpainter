from .torchpacks import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")