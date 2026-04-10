from argparse import ArgumentParser

from models.er import Er
from models.meta_cl_utils.meta_cl import MetaCL


class MetaER(MetaCL, Er):
    """MetaCL wrapper that uses ER for the inner loop adaptation."""
    NAME = 'meta_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Didn't work: parser = ArgumentParser(parents=[Er.get_parser(parser), MetaCL.get_parser(parser)], conflict_handler='resolve')
        merged_parser = MetaCL.get_parser(Er.get_parser(parser))
        return merged_parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaER, self).__init__(backbone, loss, args, transform, dataset=dataset)