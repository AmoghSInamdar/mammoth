# Re-evaluating Continual Learning with Few-Shot Adaptation

Code for [Re-evaluating Continual Learning with Few-Shot Adaptation](https://arxiv.org/abs/2606.03843) (Inamdar, So, Milenia and Zemel, 2026).

This repository is a fork of [Mammoth](https://github.com/aimagelab/mammoth). Models are trained with Mammoth as usual, and the fork adds tooling to evaluate the saved checkpoints after few-shot adaptation. It also adds the SAUCE metric for the rate of adaptation, foresight meta-learning variants of five continual learning methods, and the task sequences used in the paper.

## K-shot evaluation

Training with `--savecheck task` saves a checkpoint at the end of every task. The evaluation takes each checkpoint and tests it on every task in the sequence, including the tasks it has yet to train on. For each task, it fine-tunes a copy of the checkpoint on k examples per class and measures accuracy and loss on a test set for that task. This is repeated for k = 0, 1, 2, 5 and 10, where k = 0 is the standard zero-shot evaluation.

The k-shot examples are held out from training, and the test set is the same for every value of k. Fine-tuning runs for `--num_adapt_steps` steps at learning rate `--adapt_lr`, where a step is one pass over the k-shot examples in batches of 32. Adding `--kshot_full_batch` uses all of the k-shot examples as a single batch.

With `--layer_min N`, only layer N and the layers after it are fine-tuned, which is how we evaluate the pretrained models. Layers are the convolutional and linear layers of the backbone, counted from 1. For ViT-B/16, `--layer_min 46` fine-tunes the last transformer block and the classification head.

The results of a training run are written to one CSV in `results/k_shot_evaluation/`, with a row for each checkpoint, task and value of k. Runs with `--layer_min N` are written to `results/k_shot_evaluation_N/`.

## Per-shot plasticity

The evaluation also summarises how each checkpoint's error on a task falls as k grows. AUAC is the area under the curve of error against k. SAUCE is the same area after the curve is rescaled to run from 1 at its highest error to 0 at its lowest, so it measures how quickly the model reaches its best performance. A lower SAUCE means faster adaptation.

Both are computed from accuracy and from loss, and stored in the results CSV in columns such as `SAUCE_accuracy` and `SAUCE_loss`.

## Foresight meta-learning

The models `meta_sgd`, `meta_er`, `meta_derpp`, `meta_ewc` and `meta_agem` add foresight meta-learning to the corresponding Mammoth method. Before training on each task, the model takes a meta-update on a few examples from each of the next few tasks.

`--meta_method` selects MAML or Reptile, and `--meta_strategy` selects whether the future tasks are adapted to in parallel or in sequence. `--num_lookahead_tasks` sets how many future tasks are used, and `--num_meta_examples` sets how many examples are drawn from each. MAML requires the `higher` package.

## Task sequences

The fork adds five task sequences to those in Mammoth.

| `--dataset` | Name in the paper | Description |
| --- | --- | --- |
| `seq-cifar100-20task` | SEQ-CIFAR100-20 | CIFAR-100 in 20 tasks of 5 classes |
| `struct-cifar100` | STRUCT-CIFAR100-20 | CIFAR-100 in 20 tasks, one superclass per task |
| `smooth-rot-mnist` | ROT-MNIST-20 | MNIST in 20 tasks, with a rotation that advances from task to task |
| `seq-aircraft` | SEQ-AIRCRAFT-10 | FGVC-Aircraft in 10 tasks of 10 classes |
| `seq-imagenet-a` | SEQ-IMAGENET-A-10 | ImageNet-A in 10 tasks of 20 classes |

The last two read their images from `data/FGVC_AIRCRAFT/fgvc-aircraft-2013b/` and `data/IMAGENET_A/imagenet-a/`, which need to be downloaded by hand.

## Usage

```bash
pip install -r requirements.txt
pip install higher        # for MAML
```

`run_pipeline_full.py` trains a model, evaluates its checkpoints and plots the results.

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_pipeline_full.py \
    --dataset seq-cifar100 --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.03 --num_adapt_steps 30
```

Here `--adapt_lr` and `--num_adapt_steps` apply to the evaluation, and the remaining flags are passed to Mammoth's `main.py`. Adding `--skip_train` evaluates checkpoints that already exist.

For a meta model, the fine-tuning inside the meta-update is set with `--meta_adapt_lr` and `--meta_adapt_steps`.

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_pipeline_full.py \
    --dataset seq-cifar100 --model meta_er --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 30
```

The three stages can also be run separately.

```bash
python main.py --dataset seq-cifar100 --model er --buffer_size 500 --lr 0.1 --savecheck task

CUDA_VISIBLE_DEVICES=0,1 python run_k_shot_evaluation.py --models er --datasets seq-cifar100 \
    --k_values 0,1,2,5,10 --adapt_lr 0.03 --num_adapt_steps 30

python results_for_paper.py --dataset seq-cifar100 --plot-type stability --no-meta --k-values 5
```

`run_k_shot_evaluation.py` runs one process per checkpoint across the visible GPUs. Model names are written with a dash, as in the checkpoint file names (`ewc-on`, `meta-sgd`). For meta models, `--meta_method` and `--meta_strategy` select which checkpoints are evaluated.

`results_for_paper.py` produces the figures in the paper and writes them to `plots/paper_plots/`. The commands for every experiment and figure are collected in [k_shot_pipeline_commands.sh](k_shot_pipeline_commands.sh).

## Citation

```bibtex
@article{inamdar2026reevaluating,
  title={Re-Evaluating Continual Learning with Few-Shot Adaptation},
  author={Inamdar, Amogh and So, Matthew and Milenia, Vici and Zemel, Richard},
  journal={arXiv preprint arXiv:2606.03843},
  year={2026}
}
```

Please also cite Mammoth, whose documentation is at [aimagelab.github.io/mammoth](https://aimagelab.github.io/mammoth/).

```bibtex
@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@inproceedings{buzzega2020dark,
  author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
  pages = {15920--15930},
  publisher = {Curran Associates, Inc.},
  title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
  volume = {33},
  year = {2020}
}
```

The code is released under the MIT License, with the exceptions listed in [NOTICE.md](NOTICE.md).
