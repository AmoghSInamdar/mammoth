from models.meta_cl_utils.meta_cl import MetaCL
from models.sgd import Sgd


class MetaSgd(MetaCL, Sgd):
    """MetaCL wrapper that uses SGD for the inner loop adaptation."""
    NAME = 'meta_sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaSgd, self).__init__(backbone, loss, args, transform, dataset=dataset)