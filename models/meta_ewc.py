from argparse import ArgumentParser

from models.ewc_on import EwcOn
from models.meta_cl_utils.meta_cl import MetaCL


class MetaEwc(MetaCL, EwcOn):
    """MetaCL wrapper that uses online EWC for the inner loop adaptation."""
    NAME = 'meta_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        merged_parser = MetaCL.get_parser(EwcOn.get_parser(parser))
        return merged_parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaEwc, self).__init__(backbone, loss, args, transform, dataset=dataset)