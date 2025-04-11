from .models import ConvModel, MLPModel, ResNetModel
from .monitors import DPDDMBayesianMonitor, DPDDMFullInformationMonitor
from ._metadata import __author__, __description__, __email__, __license__, __url__
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayesian_dpddm")  
except PackageNotFoundError:
    __version__ = "unknown"
