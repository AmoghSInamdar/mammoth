# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import copy
import logging

import torch


from models.meta_cl_utils.maml import train_maml_parallel, train_maml_sequential
from models.utils.continual_model import ContinualModel
from utils.few_shot import adapt_model, create_k_shot_loader


class MetaCL(ContinualModel):
    """Foresight Meta-Continual Learning (MCL) model wrapper for k-shot evaluation."""
    NAME = 'meta_cl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--meta_method', choices=['reptile', 'maml'], default='reptile',
                            help='Meta-learning method.')
        parser.add_argument('--meta_strategy', choices=['parallel', 'sequential'], default='parallel',
                            help='Meta-learning strategy.')
        parser.add_argument('--num_lookahead_tasks', type=int, default=3,
                            help='Number of lookahead tasks for meta learning.')
        parser.add_argument('--num_adapt_steps', type=int, default=5,
                            help='Number of adaptation steps during evaluation.')
        parser.add_argument('--adapt_lr', type=float, required=False,
                            help='Learning rate for adaptation steps.')
        parser.add_argument('--meta_lr', type=float, default=0.1,
                            help='Learning rate for meta-updates.')
        parser.add_argument('--use_fast_weights', action='store_true',
                            help='Whether to use a forward model with fast weights.')
        parser.add_argument('--num_foresight_examples_per_task', type=int, default=32,
                            help='Number of foresight examples per task.')
        parser.add_argument('--sampling_seed', type=int, default=42,
                            help='Seed for sampling foresight examples.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MetaCL, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.global_task_counter = 0
        if self.args.adapt_lr is None:
            self.args.adapt_lr = self.args.lr

    def get_meta_learning_dataloaders(self):
        forward_task_ids = list(
            range(self._task_iteration, min(self._task_iteration + self.args.num_lookahead_tasks, self.n_tasks)))
        train_k = self.args.num_foresight_examples_per_task // self.classes_per_task

        meta_train_dataloaders = [create_k_shot_loader(
            self.dataset, id, train_k, num_samples_per_class=train_k, sampling_seed=self.args.sampling_seed
        ) for id in forward_task_ids]
        logging.info(f"Created meta-learning train dataloaders for tasks {forward_task_ids} with {train_k} examples per class.")

        meta_val_dataloaders = None
        if self.args.meta_method == 'maml':
            meta_val_dataloaders = [create_k_shot_loader(
                self.dataset, id, train_k, num_samples_per_class=train_k, sampling_seed=self.args.sampling_seed*2
            ) for id in forward_task_ids]
            logging.info(f"Created meta-learning validation dataloaders for tasks {forward_task_ids} with {train_k} examples per class.")

        return meta_train_dataloaders, meta_val_dataloaders

    def meta_initialize_reptile(self, meta_learning_dataloaders):
        logging.info("Initializing meta-learning with Reptile strategy.")
        if self.args.meta_strategy == 'parallel':
            adapted_net_list = [adapt_model(
                self, dataloader, self.args.num_adapt_steps, self.args.lr).net for dataloader in meta_learning_dataloaders]
        elif self.args.meta_strategy == 'sequential':
            adapted_model = copy.deepcopy(self)
            for dataloader in meta_learning_dataloaders:
                adapted_model = adapt_model(adapted_model, dataloader, self.args.num_adapt_steps, self.args.lr)
            adapted_net_list = [adapted_model.net]
        else:
            raise ValueError(f"Unknown meta strategy: {self.args.meta_strategy}")
        logging.info("Trained lookahead models for meta-learning.")
        
        old_params = self.net.get_params().data.clone()

        # Reptile update from each adapted model
        for adapted_net in adapted_net_list:
            meta_update = adapted_net.get_params() - self.net.get_params()
            new_params = self.net.get_params() + self.args.meta_lr * meta_update
            self.net.set_params(new_params)
        logging.info(f"Completed Reptile meta-update. Difference in parameters: {torch.linalg.norm(self.net.get_params() - old_params).item():.4f}")

    def meta_initialize_maml(self, meta_train_dataloaders, meta_val_dataloaders): 
        logging.info("Initializing meta-learning with MAML strategy.")
        if self.args.meta_strategy == 'parallel':
            adapted_model = train_maml_parallel(self, meta_train_dataloaders, meta_val_dataloaders)
        elif self.args.meta_strategy == 'sequential':
            adapted_model = train_maml_sequential(self, meta_train_dataloaders, meta_val_dataloaders)
        else:
            raise ValueError(f"Unknown meta strategy: {self.args.meta_strategy}")
        
        self.net.set_params(adapted_model.net.get_params())
        logging.info("Completed MAML meta-update.")

    def begin_task(self, dataset):
        meta_train_dataloaders, meta_val_dataloaders = self.get_meta_learning_dataloaders()
        if self.args.meta_method == 'reptile':
            self.meta_initialize_reptile(meta_train_dataloaders)
        elif self.args.meta_method == 'maml':
            self.meta_initialize_maml(meta_train_dataloaders, meta_val_dataloaders)

        return super().begin_task(dataset)

