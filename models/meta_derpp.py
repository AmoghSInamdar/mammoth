from argparse import ArgumentParser

from models.meta_cl_utils.meta_cl import MetaCL
from models.derpp import Derpp


class MetaDerpp(MetaCL, Derpp):
    """MetaCL wrapper that uses DER++ for the inner loop adaptation."""
    NAME = 'meta_derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        merged_parser = MetaCL.get_parser(Derpp.get_parser(parser))
        return merged_parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaDerpp, self).__init__(backbone, loss, args, transform, dataset=dataset)