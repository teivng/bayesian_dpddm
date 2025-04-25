from .bayesian_monitor import DPDDMBayesianMonitor
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset


class WILDSMonitor(DPDDMBayesianMonitor):
    """Bayesian D-PDDM Monitor for the WILDS benchmark
    Uses evaluation metrics provided by WILDS.

    Attributes:
        model (DPDDMAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        train_cfg (TrainConfig): TrainConfig object configuring all aspects of training
        device (torch.device): torch.device, cuda or cpu
    """
    
    def __init__(self, *args, **kwargs):
        super(WILDSMonitor, self).__init__(*args, **kwargs)
        