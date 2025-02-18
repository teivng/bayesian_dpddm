from .models import DPDDMConvModel
from .monitors import DPDDMBayesianMonitor
from ._metadata import *
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayesian_dpddm")  
except PackageNotFoundError:
    __version__ = "unknown"
