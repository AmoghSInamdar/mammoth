# Plot Results

## Row figures for the paper (one figure per row of datasets, at text width; -> plots/paper_plots/rows/)
## Datasets are comma-separated; "-20task" selects the 20-task variant, "@N" a per-dataset layer_min.

python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@46 --row-name main_lm46 --plot-type stability --no-meta --k-values 5
python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@46 --row-name main_lm46 --plot-type forward_transfer --no-meta --k-values 10
python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@50 --row-name main_lm50 --plot-type stability --no-meta --k-values 5
python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@50 --row-name main_lm50 --plot-type forward_transfer --no-meta --k-values 10
python results_for_paper.py --dataset seq-cifar100-20task,struct-cifar100 --row-name 20task --plot-type improvement --no-meta --k-values 10
python results_for_paper.py --dataset seq-cifar100-20task,seq-aircraft@46 --row-name cifar20_aircraft_lm46 --plot-type improvement --no-meta --k-values 10
python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@46 --row-name main_lm46 --plot-type meta_improvement --num-lookahead 1,3,3,3 --k-values 5
python results_for_paper.py --dataset seq-mnist,seq-cifar100,struct-cifar100,seq-aircraft@46 --row-name main_lm46 --plot-type sauce_avg --metric loss
python results_for_paper.py --dataset seq-cifar100-20task,struct-cifar100,seq-aircraft@46 --row-name 20task_aircraft_lm46 --plot-type sauce --metric loss
## Appendix rows (ROT-MNIST-20, SEQ-TINYIMAGENET-10 and SEQ-IMAGENET-A-10)
python results_for_paper.py --dataset smooth-rot-mnist,seq-tinyimg,seq-imagenet-a@46 --row-name appendix_lm46 --plot-type stability --no-meta --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist,seq-tinyimg,seq-imagenet-a@46 --row-name appendix_lm46 --plot-type forward_transfer --no-meta --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist,seq-tinyimg,seq-imagenet-a@46 --row-name appendix_lm46 --plot-type improvement --no-meta --k-values 10
## Appendix rows: late-layer adaptation of the non-pretrained learners (MLP layer 3 = classifier; ResNet-18 layer 21 = classifier)
python results_for_paper.py --dataset seq-mnist@3,seq-cifar100@21,struct-cifar100@21,smooth-rot-mnist@21 --row-name latelayer --plot-type stability --no-meta --k-values 5
python results_for_paper.py --dataset seq-mnist@3,seq-cifar100@21,struct-cifar100@21,smooth-rot-mnist@21 --row-name latelayer --plot-type forward_transfer --no-meta --k-values 10
python results_for_paper.py --dataset seq-mnist@3,seq-cifar100@21,struct-cifar100@21,smooth-rot-mnist@21 --row-name latelayer --plot-type improvement --no-meta --k-values 10
python results_for_paper.py --dataset seq-mnist@3,seq-cifar100@21,struct-cifar100@21 --row-name latelayer --plot-type meta_improvement --num-lookahead 1,3,3 --k-values 5
## The camera-ready LaTeX (Meta_CL_Neurips_2026/sections_camera_ready) includes these from Meta_CL_Neurips_2026/figures_camera_ready/

## SAUCE metric: every CSV stores SAUCE_accuracy and SAUCE_loss (utils/per_shot_plasticity.py); the SAUCE plot types
## and all_results_table read the one selected by --metric (default accuracy). The June-2026 paper figures 8/9 are --metric loss.

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce --metric loss
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type sauce --metric loss
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce --metric loss
python results_for_paper.py --dataset seq-mnist --plot-type sauce --metric loss
python results_for_paper.py --dataset seq-aircraft --layer_min 46 --plot-type sauce --metric loss

## Results table (one row per method and metric; the 20-task variant goes to all_results_table_20task.csv)

python results_for_paper.py --dataset seq-cifar100 --plot-type all_results_table
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type all_results_table
python results_for_paper.py --dataset struct-cifar100 --plot-type all_results_table
python results_for_paper.py --dataset seq-mnist --plot-type all_results_table
python results_for_paper.py --dataset smooth-rot-mnist --plot-type all_results_table


## stability

python results_for_paper.py --dataset seq-cifar100 --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset seq-tinyimg --plot-type stability --no-meta  --k-values 5

python results_for_paper.py --dataset seq-cifar100 --plot-type stability --no-meta  --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type stability --no-meta  --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type stability --no-meta  --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type stability --no-meta  --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type stability --no-meta  --k-values 10
python results_for_paper.py --dataset seq-imagenet-r --plot-type stability --no-meta  --k-values 10

## forward transfer

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset seq-tinyimg --no-meta --plot-type forward_transfer --k-values 10

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type forward_transfer --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type forward_transfer --k-values 5
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type forward_transfer --k-values 5
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type forward_transfer --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type forward_transfer --k-values 5
python results_for_paper.py --dataset seq-imagenet-r --no-meta --plot-type forward_transfer --k-values 5
## k-shot improvement 

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type improvement --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type improvement --k-values 5
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type improvement --k-values 5
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type improvement --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type improvement --k-values 5
python results_for_paper.py --dataset seq-tinyimg --no-meta --plot-type improvement --k-values 5

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type improvement --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type improvement --k-values 10
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type improvement --k-values 10
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type improvement --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type improvement --k-values 10

## sauce

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce --no-meta
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type sauce --no-meta
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce --no-meta
python results_for_paper.py --dataset seq-mnist --plot-type sauce --no-meta
python results_for_paper.py --dataset smooth-rot-mnist --plot-type sauce --no-meta
python results_for_paper.py --dataset seq-tinyimg --plot-type sauce --no-meta

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type sauce
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce
python results_for_paper.py --dataset seq-mnist --plot-type sauce
python results_for_paper.py --dataset smooth-rot-mnist --plot-type sauce

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce --only-last-20
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type sauce --only-last-20
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce --only-last-20
python results_for_paper.py --dataset seq-mnist --plot-type sauce --only-last-20
python results_for_paper.py --dataset smooth-rot-mnist --plot-type sauce --only-last-20


## meta-cl improvement

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --k-values 1
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --k-values 1
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --k-values 1
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --k-values 1
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --k-values 1

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --k-values 2
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --k-values 2
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --k-values 2
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --k-values 2
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --k-values 2

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --k-values 5

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --k-values 10

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --meta-method no_meta --meta-strategy parallel --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type meta_improvement --meta-method no_meta --meta-strategy parallel --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --meta-method no_meta --meta-strategy parallel --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --meta-method no_meta --meta-strategy parallel --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --meta-method no_meta --meta-strategy parallel --k-values 10

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --meta-method reptile --meta-strategy sequential --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type meta_improvement --meta-method reptile --meta-strategy sequential --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --meta-method reptile --meta-strategy sequential --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --meta-method reptile --meta-strategy sequential --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --meta-method reptile --meta-strategy sequential --k-values 5

python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --k-values 1
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --k-values 2
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --k-values 10

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --k-values 1
python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --k-values 2
python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --k-values 5
python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --k-values 10

python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --k-values 1
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --k-values 2
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --k-values 10

python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --num-lookahead 3  --k-values 1
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --num-lookahead 3  --k-values 2
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --num-lookahead 3  --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement --num-lookahead 3  --k-values 10

python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --k-values 1
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --k-values 2
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --k-values 10

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement  --num-lookahead 3 --forward-only --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --forward-only --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 5

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement  --num-lookahead 3 --forward-only --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --forward-only --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --forward-only --k-values 10


python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement  --num-lookahead 3 --backward-only --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --backward-only --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 5

python results_for_paper.py --dataset seq-cifar100 --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type meta_improvement  --num-lookahead 3 --backward-only --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type meta_improvement --num-lookahead 1 --backward-only --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type meta_improvement --num-lookahead 3 --backward-only --k-values 10


## meta-cl Method Comparison

python results_for_paper.py --dataset seq-cifar100 --plot-type compare_meta_methods --k-values 1
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type compare_meta_methods --k-values 1
python results_for_paper.py --dataset struct-cifar100 --plot-type compare_meta_methods --k-values 1
python results_for_paper.py --dataset seq-mnist --plot-type compare_meta_methods --k-values 1
python results_for_paper.py --dataset smooth-rot-mnist --plot-type compare_meta_methods --k-values 1

python results_for_paper.py --dataset seq-cifar100 --plot-type compare_meta_methods --k-values 2
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type compare_meta_methods --k-values 2
python results_for_paper.py --dataset struct-cifar100 --plot-type compare_meta_methods --k-values 2
python results_for_paper.py --dataset seq-mnist --plot-type compare_meta_methods --k-values 2
python results_for_paper.py --dataset smooth-rot-mnist --plot-type compare_meta_methods --k-values 2

python results_for_paper.py --dataset seq-cifar100 --plot-type compare_meta_methods --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type compare_meta_methods --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type compare_meta_methods --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type compare_meta_methods --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type compare_meta_methods --k-values 5

python results_for_paper.py --dataset seq-cifar100 --plot-type compare_meta_methods --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type compare_meta_methods --k-values 10
python results_for_paper.py --dataset struct-cifar100 --plot-type compare_meta_methods --k-values 10
python results_for_paper.py --dataset seq-mnist --plot-type compare_meta_methods --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --plot-type compare_meta_methods --k-values 10


# FULL TRAIN -> EVAL -> PLOT PIPELINES

# Seq MNIST

## Non-meta baselines

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model sgd  --lr 0.1 \
    --backbone mnistmlp_small \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/sgd_seq_mnist.out 2>outputs/sgd_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model er --buffer_size 500 --lr 0.1 \
    --backbone mnistmlp_small \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/er_seq_mnist.out 2>outputs/er_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --backbone mnistmlp_small \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/derpp_seq_mnist.out 2>outputs/derpp_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --backbone mnistmlp_small \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/ewc_seq_mnist.out 2>outputs/ewc_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model agem --buffer_size 500 --lr 0.03 \
    --backbone mnistmlp_small \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/agem_seq_mnist.out 2>outputs/agem_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --backbone mnistmlp_small \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/mer_seq_mnist.out 2>outputs/mer_seq_mnist.err &

## Meta learning

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model meta_sgd  --lr 0.1 \
    --backbone mnistmlp_small \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 1 --meta_lr 0.01 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 10 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/meta_sgd_seq_mnist.out 2>outputs/meta_sgd_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model meta_sgd  --lr 0.1 \
    --backbone mnistmlp_small \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 1 --meta_lr 0.01 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 10 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/no_meta_sgd_seq_mnist.out 2>outputs/no_meta_sgd_seq_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-mnist --model meta_sgd  --lr 0.1 \
    --backbone mnistmlp_small \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 1 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 10 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_seq_mnist.out 2>outputs/maml_sgd_seq_mnist.err &


# Seq CIFAR 100

## Non-meta baselines

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model sgd  --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/sgd_seq_cifar100.out 2>outputs/sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/er_seq_cifar100.out 2>outputs/er_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/derpp_seq_cifar100.out 2>outputs/derpp_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model ewc_on --n_epochs 50 --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/ewc_seq_cifar100.out 2>outputs/ewc_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/agem_seq_cifar100.out 2>outputs/agem_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model lwf --lr 0.03 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/lwf_seq_cifar100.out 2>outputs/lwf_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/mer_seq_cifar100.out 2>outputs/mer_seq_cifar100.err &

## Meta learning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.2 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    >outputs/meta_sgd_seq_cifar100.out 2>outputs/meta_sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/no_meta_sgd_seq_cifar100.out 2>outputs/no_meta_sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_seq_cifar100.out 2>outputs/maml_sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_er  --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_seq_cifar100.out 2>outputs/maml_er_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_derpp_seq_cifar100.out 2>outputs/maml_derpp_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_ewc --lr 0.1 --e_lambda 10 --gamma 1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_ewc_seq_cifar100.out 2>outputs/maml_ewc_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_agem --buffer_size 500 --lr 0.03 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_agem_seq_cifar100.out 2>outputs/maml_agem_seq_cifar100.err &


# Seq CIFAR 100 20 task

## Non-meta baselines

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model sgd  --lr 0.1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/sgd_seq_cifar100_20task.out 2>outputs/sgd_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/er_seq_cifar100_20task.out 2>outputs/er_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/derpp_seq_cifar100_20task.out 2>outputs/derpp_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/ewc_seq_cifar100_20task.out 2>outputs/ewc_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/agem_seq_cifar100_20task.out 2>outputs/agem_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model lwf --lr 0.03 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/lwf_seq_cifar100_20task.out 2>outputs/lwf_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/mer_seq_cifar100_20task.out 2>outputs/mer_seq_cifar100_20task.err &

## Meta learning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_sgd  --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --skip_train \
    >outputs/meta_sgd_seq_cifar100_20task.out 2>outputs/meta_sgd_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_sgd  --lr 0.1 \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/no_meta_sgd_seq_cifar100_20task.out 2>outputs/no_meta_sgd_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/maml_sgd_seq_cifar100_20task.out 2>outputs/maml_sgd_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/seq_maml_sgd_seq_cifar100_20task.out 2>outputs/seq_maml_sgd_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_er  --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_seq_cifar100_20task.out 2>outputs/maml_er_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_derpp_seq_cifar100_20task.out 2>outputs/maml_derpp_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_ewc --lr 0.1 --e_lambda 10 --gamma 1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_ewc_seq_cifar100_20task.out 2>outputs/maml_ewc_seq_cifar100_20task.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100-20task --model meta_agem --buffer_size 500 --lr 0.03 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_agem_seq_cifar100_20task.out 2>outputs/maml_agem_seq_cifar100_20task.err &


# Struct CIFAR100

## Non-meta baselines

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model sgd  --lr 0.1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/sgd_struct_cifar100.out 2>outputs/sgd_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/er_struct_cifar100.out 2>outputs/er_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/derpp_struct_cifar100.out 2>outputs/derpp_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/ewc_struct_cifar100.out 2>outputs/ewc_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/agem_struct_cifar100.out 2>outputs/agem_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model lwf --lr 0.03 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/lwf_struct_cifar100.out 2>outputs/lwf_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/mer_struct_cifar100.out 2>outputs/mer_struct_cifar100.err &

## Meta learning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.01 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --skip_train \
    >outputs/meta_sgd_struct_cifar100.out 2>outputs/meta_sgd_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/no_meta_sgd_struct_cifar100.out 2>outputs/no_meta_sgd_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_struct_cifar100.out 2>outputs/maml_sgd_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/seq_maml_sgd_struct_cifar100.out 2>outputs/seq_maml_sgd_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_er --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_struct_cifar100.out 2>outputs/maml_er_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_derpp_struct_cifar100.out 2>outputs/maml_derpp_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_ewc --lr 0.1 --e_lambda 10 --gamma 1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_ewc_struct_cifar100.out 2>outputs/maml_ewc_struct_cifar100.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_agem --buffer_size 500 --lr 0.03 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_agem_struct_cifar100.out 2>outputs/maml_agem_struct_cifar100.err &

# Smooth Rotated MNIST

## Non-meta baselines

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model sgd  --lr 0.1 \
    --adapt_lr 0.05 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/sgd_smooth_mnist.out 2>outputs/sgd_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/er_smooth_mnist.out 2>outputs/er_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/derpp_smooth_mnist.out 2>outputs/derpp_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/ewc_smooth_mnist.out 2>outputs/ewc_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/agem_smooth_mnist.out 2>outputs/agem_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model lwf --lr 0.03 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/lwf_smooth_mnist.out 2>outputs/lwf_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/mer_smooth_mnist.out 2>outputs/mer_smooth_mnist.err &

## Meta learning

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_sgd  --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.01 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/meta_sgd_smooth_mnist.out 2>outputs/meta_sgd_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_sgd  --lr 0.1 \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/no_meta_sgd_smooth_mnist.out 2>outputs/no_meta_sgd_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_smooth_mnist.out 2>outputs/maml_sgd_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/seq_maml_sgd_smooth_mnist.out 2>outputs/seq_maml_sgd_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_er  --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_smooth_mnist.out 2>outputs/maml_er_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_er  --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/seq_maml_er_smooth_mnist.out 2>outputs/seq_maml_er_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_derpp --buffer_size 500 --lr 0.03 --alpha 0.3 --beta 0.5 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_derpp_smooth_mnist.out 2>outputs/maml_derpp_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_derpp --buffer_size 500 --lr 0.03 --alpha 0.3 --beta 0.5 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.003 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/seq_maml_derpp_smooth_mnist.out 2>outputs/seq_maml_derpp_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_ewc --lr 0.1 --e_lambda 10 --gamma 1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_ewc_smooth_mnist.out 2>outputs/maml_ewc_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model meta_agem --buffer_size 500 --lr 0.03 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.03 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_agem_smooth_mnist.out 2>outputs/maml_agem_smooth_mnist.err &


# Seq TinyImagenet

## Non-meta baselines

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-tinyimg --model sgd  --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/sgd_seq_tinyimg.out 2>outputs/sgd_seq_tinyimg.err &

CUDA_VISIBLE_DEVICES=5,6,7 \
    python run_pipeline_full.py --dataset seq-tinyimg --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/er_seq_tinyimg.out 2>outputs/er_seq_tinyimg.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-tinyimg --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.03 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/derpp_seq_tinyimg.out 2>outputs/derpp_seq_tinyimg.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset seq-tinyimg --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/ewc_seq_tinyimg.out 2>outputs/ewc_seq_tinyimg.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset seq-tinyimg --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/agem_seq_tinyimg.out 2>outputs/agem_seq_tinyimg.err &

## Meta learning

CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python run_pipeline_full.py --dataset seq-tinyimg --model meta_sgd  --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.01 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 100 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/reptile_sgd_seq_tinyimg.out 2>outputs/reptile_sgd_seq_tinyimg.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset seq-tinyimg --model meta_sgd  --lr 0.1 \
    --meta_method maml --meta_strategy sequential --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 100 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_seq_tinyimg.out 2>outputs/maml_sgd_seq_tinyimg.err &

## Split-ImageNet-R with pretrained ViT-B/16 (in21k-ft-in1k; default backbone for seq-imagenet-r)
# - 10 epochs, batch 128 (derpp peaks ~34GB on one A100 incl. replay passes)
# - IMAGENET_R_TRAIN_CAP=60 halves the train set (11,831 of 24,000 imgs; IN-R averages
#   120 imgs/class, so 60/class ~= half; classes below the cap keep all images)
# - Training lrs scaled ~10x down from the ResNet configs for ViT full finetuning
# - k-shot adaptation: head-only via --layer_min 50 (ViT-B has 50 weight layers;
#   use 46 for last-transformer-block + head), adapt_lr 0.001 (ViT-scale)
# - layer_min > 0 routes eval results to results/k_shot_evaluation_50/ and plots to
#   plots/layer_min_50/ automatically, keeping them separate from full-adapt runs

CUDA_VISIBLE_DEVICES=4 IMAGENET_R_TRAIN_CAP=60 \
    python run_pipeline_full.py --dataset seq-imagenet-r --model sgd --lr 0.01 \
    --n_epochs 10 --batch_size 128 \
    --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 \
    --savecheck task \
    >outputs/sgd_seq_imagenet_r.out 2>outputs/sgd_seq_imagenet_r.err &

CUDA_VISIBLE_DEVICES=5 IMAGENET_R_TRAIN_CAP=60 \
    python run_pipeline_full.py --dataset seq-imagenet-r --model er --buffer_size 500 --lr 0.01 \
    --n_epochs 10 --batch_size 128 \
    --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 \
    --savecheck task \
    >outputs/er_seq_imagenet_r.out 2>outputs/er_seq_imagenet_r.err &

CUDA_VISIBLE_DEVICES=6 IMAGENET_R_TRAIN_CAP=60 \
    python run_pipeline_full.py --dataset seq-imagenet-r --model derpp --lr 0.003 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --n_epochs 10 --batch_size 128 \
    --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 \
    --savecheck task \
    >outputs/derpp_seq_imagenet_r.out 2>outputs/derpp_seq_imagenet_r.err &

CUDA_VISIBLE_DEVICES=4 IMAGENET_R_TRAIN_CAP=60 \
    python run_pipeline_full.py --dataset seq-imagenet-r --model ewc_on --lr 0.01 --e_lambda 10 --gamma 1 \
    --n_epochs 10 --batch_size 128 \
    --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 \
    --savecheck task \
    >outputs/ewc_seq_imagenet_r.out 2>outputs/ewc_seq_imagenet_r.err &

CUDA_VISIBLE_DEVICES=7 IMAGENET_R_TRAIN_CAP=60 \
    python run_pipeline_full.py --dataset seq-imagenet-r --model agem --buffer_size 500 --lr 0.003 \
    --n_epochs 10 --batch_size 128 \
    --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 \
    --savecheck task \
    >outputs/agem_seq_imagenet_r.out 2>outputs/agem_seq_imagenet_r.err &

# Re-run just the k-shot eval + plots on existing checkpoints (add --skip_train), e.g.:
#   python run_pipeline_full.py --dataset seq-imagenet-r --model sgd --lr 0.01 \
#       --adapt_lr 0.001 --num_adapt_steps 10 --layer_min 50 --skip_train
