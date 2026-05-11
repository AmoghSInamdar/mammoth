from argparse import ArgumentParser

from models.agem import AGem
from models.meta_cl_utils.meta_cl import MetaCL


class MetaAGem(MetaCL, AGem):
    """MetaCL wrapper that uses ER for the inner loop adaptation."""
    NAME = 'meta_agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        merged_parser = MetaCL.get_parser(AGem.get_parser(parser))
        return merged_parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaAGem, self).__init__(backbone, loss, args, transform, dataset=dataset)