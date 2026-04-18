# INDIVIDUAL TRAINING, EVAL, PLOTTING COMMANDS FOR DEBUGGING

# Get checkpoints, copy first line from from scripts/reproduce.json

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=seq-cifar100  --model=derpp --buffer_size=500 --model_config=best  \
    --savecheck=task \
    >outputs/train_model.out 2>outputs/train_model.err &


CUDA_VISIBLE_DEVICES=7 python main.py --dataset=seq-cifar100-20task  --model=sgd --lr=0.1  \
    --savecheck=task \
    >outputs/train_sgd_20task.out 2>outputs/train_sgd_20task.err &


CUDA_VISIBLE_DEVICES=7 python main.py --dataset=rot-mnist  --model=sgd --lr=0.1  \
    --savecheck=task \
    >outputs/train_sgd_mnist.out 2>outputs/train_sgd_mnist.err &

CUDA_VISIBLE_DEVICES=5 python main.py --dataset=rot-mnist --model=er --buffer_size=500 --lr=0.1  \
    --savecheck=task \
    >outputs/train_er_mnist.out 2>outputs/train_er_mnist.err &


# Train meta-cl methods

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=seq-cifar100 --model=meta_sgd --lr=0.1 --meta_method=no_meta \
    --adapt_lr=0.2 --num_adapt_steps=15 \
    --savecheck=task \
    >outputs/meta_sgd.out 2>outputs/meta_sgd.err &

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=seq-cifar100-20task --model=meta_sgd --lr=0.1 --meta_strategy=parallel \
    --adapt_lr=0.1 --num_adapt_steps=10 --num_meta_examples=25 \
    --savecheck=task \
    >outputs/meta_sgd_20task.out 2>outputs/meta_sgd_20task.err &

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=rot-mnist --model=meta_sgd --lr=0.1 --meta_strategy=parallel \
    --adapt_lr=0.01 --num_adapt_steps=5 --num_meta_examples=30 \
    --savecheck=task \
    >outputs/meta_sgd_mnist.out 2>outputs/meta_sgd_mnist.err &


CUDA_VISIBLE_DEVICES=6 python main.py --dataset=seq-cifar100 --model=meta_sgd --lr=0.1 --meta_method=maml --meta_strategy=parallel --adapt_lr=0.01 --num_adapt_steps=2 \
    --savecheck=task \
    >outputs/meta_sgd_maml.out 2>outputs/meta_sgd_maml.err &

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=seq-cifar100 --model=meta_er --buffer_size=500 --lr=0.1 \
    --adapt_lr=0.3 --num_adapt_steps=10 \
    --savecheck=task \
    >outputs/meta_er.out 2>outputs/meta_er.err &

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=rot-mnist --model=meta_er --buffer_size=500 --lr=0.1 \
    --adapt_lr=0.3 --num_adapt_steps=10 \
    --savecheck=task \
    >outputs/meta_er_mnist.out 2>outputs/meta_er_mnist.err &


CUDA_VISIBLE_DEVICES=7 python main.py --dataset=seq-cifar100 --model=meta_ewc --n_epochs=50 --lr=0.1 --e_lambda=10 --gamma=1 \
    --savecheck=task \
    >outputs/meta_ewc.out 2>outputs/meta_ewc.err &

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=seq-cifar100 --model=meta_derpp --buffer_size=500 --lr=0.03 --alpha=0.3 --beta=0.5 \
    --adapt_lr=0.3 --num_adapt_steps=10 \
    --savecheck=task \
    >outputs/meta_derpp.out 2>outputs/meta_derpp.err &


# Smoke test

CUDA_VISIBLE_DEVICES=5 python eval_checkpoints.py \
    --checkpoint_paths=checkpoints/der_seq-cifar100_4.pt \
    --model=der \
    --eval_dataset=seq-cifar100 \
    --eval_tasks=0,2,4,6,8 \
    --k_values=0,1,10 \
    --lr=0.001 \
    --adapt_lr=0.0001 \
    --num_adapt_steps=5 \
    --output_dir results \
    > outputs/test_eval.out 2> outputs/test_eval.err &


# Full checkpoint evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python run_k_shot_evaluation.py \
    --adapt_lr=0.01 --num_adapt_steps=10 \
    > outputs/eval_all.out 2> outputs/eval_all.err &


# Plot results

python plot_k_shot_results.py \
    --plot-all \
    > outputs/plot_results.out 2> outputs/plot_results.err &

python plot_k_shot_results.py \
    --plot-plasticity-comparisons \
    --dataset=struct-cifar100 \
    > outputs/plot_results.out 2> outputs/plot_results.err &

python plot_k_shot_results.py \
    --plot-k-shot-comparisons \
    --dataset=struct-cifar100 \
    > outputs/plot_results.out 2> outputs/plot_results.err &

# Compute plasticity scores

python utils/per_shot_plasticity.py \
    --process-all \
    --metric loss \
    > outputs/plasticity_scores.out 2> outputs/plasticity_scores.err &
