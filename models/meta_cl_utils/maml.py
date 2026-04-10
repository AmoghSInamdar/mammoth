import copy
import logging

import higher
import torch
from tqdm import tqdm


from models.utils.continual_model import ContinualModel


def train_maml_sequential(model: ContinualModel, meta_train_dataloaders, meta_val_dataloaders):
    adapted_model = copy.deepcopy(model)
    optimizer_class = type(adapted_model.opt)
    meta_opt = optimizer_class(
        adapted_model.net.parameters(),
        lr=model.args.meta_lr
    )
    inner_opt = optimizer_class(
        adapted_model.net.parameters(),
        lr=model.args.adapt_lr
    )
    meta_opt.zero_grad()
    adapted_model.net.train()

    with higher.innerloop_ctx(adapted_model.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
        for i, train_loader in enumerate(meta_train_dataloaders):
            logging.info(f"Inner loop step {i+1}/{len(meta_train_dataloaders)}: Adapting to task with {len(train_loader.dataset)} examples.")
            pbar = tqdm(range(model.args.num_adapt_steps), desc="Adapting model")
            for step in pbar:
                total_loss = 0.0
                num_batches = 0
                for train_batch in train_loader:
                    inputs, labels, not_aug_inputs = train_batch[0], train_batch[1], train_batch[2]
                    inputs, labels = inputs.to(fnet.device), labels.to(fnet.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(fnet.device)
                    outputs = fnet(inputs)
                    loss = adapted_model.loss(outputs, labels)
                    diffopt.step(loss)
                    total_loss += loss.item()
                    num_batches += 1
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
        
        # Outer loop: Use the last validation dataloader for meta-update
        val_loss = 0.0
        for val_batch in meta_val_dataloaders[-1]:
            val_inputs, val_labels, val_not_aug_inputs = val_batch[0], val_batch[1], val_batch[2]
            val_inputs, val_labels = val_inputs.to(fnet.device), val_labels.to(fnet.device, dtype=torch.long)
            val_not_aug_inputs = val_not_aug_inputs.to(fnet.device)
            val_loss += adapted_model.loss(fnet(val_inputs), val_labels)
        logging.info(f"Validation loss after adaptation: {val_loss.item():.4f}")
        val_loss.backward()
    
    meta_opt.step()
    return adapted_model


def train_maml_parallel(model: ContinualModel, meta_train_dataloaders, meta_val_dataloaders):
    adapted_model = copy.deepcopy(model)
    optimizer_class = type(adapted_model.opt)
    meta_opt = optimizer_class(
        adapted_model.net.parameters(),
        lr=model.args.meta_lr
    )
    inner_opt = optimizer_class(
        adapted_model.net.parameters(),
        lr=model.args.adapt_lr
    )
    meta_opt.zero_grad()
    adapted_model.net.train()

    for i, (train_loader, val_loader) in enumerate(zip(meta_train_dataloaders, meta_val_dataloaders)):
        logging.info(f"Outer loop step {i+1}/{len(meta_train_dataloaders)}: Adapting to task with {len(train_loader.dataset)} examples.")

        with higher.innerloop_ctx(adapted_model.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            pbar = tqdm(range(model.args.num_adapt_steps), desc="Adapting model")
            for step in pbar:
                total_loss = 0.0
                num_batches = 0
                for train_batch in train_loader:
                    inputs, labels, not_aug_inputs = train_batch[0], train_batch[1], train_batch[2]
                    inputs, labels = inputs.to(fnet.device), labels.to(fnet.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(fnet.device)
                    outputs = fnet(inputs)
                    loss = adapted_model.loss(outputs, labels)
                    diffopt.step(loss)
                    total_loss += loss.item()
                    num_batches += 1
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
        
            # Outer loop: Use the last validation dataloader for meta-update
            val_loss = 0.0
            for val_batch in val_loader:
                val_inputs, val_labels, val_not_aug_inputs = val_batch[0], val_batch[1], val_batch[2]
                val_inputs, val_labels = val_inputs.to(fnet.device), val_labels.to(fnet.device, dtype=torch.long)
                val_not_aug_inputs = val_not_aug_inputs.to(fnet.device)
                val_loss += adapted_model.loss(fnet(val_inputs), val_labels)
            logging.info(f"Validation loss after adaptation: {val_loss.item():.4f}")
            val_loss.backward()
        
    meta_opt.step()
    return adapted_model
