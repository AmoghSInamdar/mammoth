# Plot Results

python results_for_paper.py --dataset seq-cifar100 --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset seq-cifar100 --include-20task --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset struct-cifar100 --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset seq-mnist --plot-type stability --no-meta  --k-values 5
python results_for_paper.py --dataset smooth-rot-mnist --plot-type stability --no-meta  --k-values 5

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type forward_transfer --k-values 10
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type forward_transfer --k-values 10

python results_for_paper.py --dataset seq-cifar100 --no-meta --plot-type improvement --k-values avg
python results_for_paper.py --dataset seq-cifar100 --include-20task --no-meta --plot-type improvement --k-values avg
python results_for_paper.py --dataset struct-cifar100 --no-meta --plot-type improvement --k-values avg
python results_for_paper.py --dataset seq-mnist --no-meta --plot-type improvement --k-values avg
python results_for_paper.py --dataset smooth-rot-mnist --no-meta --plot-type improvement --k-values avg

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce --no-meta
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type sauce --no-meta
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce --no-meta
python results_for_paper.py --dataset seq-mnist --plot-type sauce --no-meta
python results_for_paper.py --dataset smooth-rot-mnist --plot-type sauce --no-meta

python results_for_paper.py --dataset seq-cifar100 --plot-type sauce
python results_for_paper.py --dataset seq-cifar100 --include-20task  --plot-type sauce
python results_for_paper.py --dataset struct-cifar100 --plot-type sauce
python results_for_paper.py --dataset seq-mnist --plot-type sauce
python results_for_paper.py --dataset smooth-rot-mnist --plot-type sauce


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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset seq-cifar100 --model sgd  --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/sgd_seq_cifar100.out 2>outputs/sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/er_seq_cifar100.out 2>outputs/er_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
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
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_sgd_seq_cifar100.out 2>outputs/maml_sgd_seq_cifar100.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset seq-cifar100 --model meta_er  --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_seq_cifar100.out 2>outputs/maml_er_seq_cifar100.err &


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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset struct-cifar100 --model meta_er --buffer_size 500 --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 25 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/maml_er_struct_cifar100.out 2>outputs/maml_er_struct_cifar100.err &


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
    >outputs/derpp_smooth_mnist.out 2>outputs/derpp_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model ewc_on --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/ewc_smooth_mnist.out 2>outputs/ewc_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    >outputs/agem_smooth_mnist.out 2>outputs/agem_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model lwf --lr 0.03 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train \
    >outputs/lwf_smooth_mnist.out 2>outputs/lwf_smooth_mnist.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python run_pipeline_full.py --dataset smooth-rot-mnist --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.003 --num_adapt_steps 10 \
    --savecheck task \
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