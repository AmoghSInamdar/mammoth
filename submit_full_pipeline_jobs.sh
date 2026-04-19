#!/bin/bash

DATASET="${1:-smooth-mnist}"
MODELS_ARG="${2:-all}"

if [ "$MODELS_ARG" = "all" ]; then
    MODELS="sgd er derpp ewc_on agem lwf mer meta_sgd no_meta_sgd maml"
else
    MODELS="$MODELS_ARG"
fi

for MODEL in $MODELS; do
    case $MODEL in
        sgd)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_sgd
#SBATCH --output=logs/${DATASET}/sgd_%j.out
#SBATCH --error=logs/${DATASET}/sgd_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model sgd --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --skip_train
EOF
            ;;
        er)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_er
#SBATCH --output=logs/${DATASET}/er_%j.out
#SBATCH --error=logs/${DATASET}/er_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model er --buffer_size 500 --lr 0.1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --do_train
EOF
            ;;
        derpp)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_derpp
#SBATCH --output=logs/${DATASET}/derpp_%j.out
#SBATCH --error=logs/${DATASET}/derpp_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --do_train
EOF
            ;;
        ewc_on)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_ewc_on
#SBATCH --output=logs/${DATASET}/ewc_on_%j.out
#SBATCH --error=logs/${DATASET}/ewc_on_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model ewc-on --n_epochs 50 --lr 0.1 --e_lambda 10 --gamma 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --do_train
EOF
            ;;
        agem)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_agem
#SBATCH --output=logs/${DATASET}/agem_%j.out
#SBATCH --error=logs/${DATASET}/agem_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model agem --buffer_size 500 --lr 0.03 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task
EOF
            ;;
        lwf)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_lwf
#SBATCH --output=logs/${DATASET}/lwf_%j.out
#SBATCH --error=logs/${DATASET}/lwf_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model lwf --lr 0.03 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task \
    --do_train
EOF
            ;;
        mer)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_mer
#SBATCH --output=logs/${DATASET}/mer_%j.out
#SBATCH --error=logs/${DATASET}/mer_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model mer --lr 0.1 --beta 0.01 --gamma 0.03 \
    --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task
EOF
            ;;
        meta_sgd)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_meta_sgd
#SBATCH --output=logs/${DATASET}/meta_sgd_%j.out
#SBATCH --error=logs/${DATASET}/meta_sgd_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model meta-sgd --lr 0.1 \
    --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.2 \
    --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.1 --num_adapt_steps 10 \
    --savecheck task
EOF
            ;;
        no_meta_sgd)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_no_meta_sgd
#SBATCH --output=logs/${DATASET}/no_meta_sgd_%j.out
#SBATCH --error=logs/${DATASET}/no_meta_sgd_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model meta-sgd --lr 0.1 \
    --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task
EOF
            ;;
        maml)
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_maml
#SBATCH --output=logs/${DATASET}/maml_%j.out
#SBATCH --error=logs/${DATASET}/maml_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs/${DATASET}

source .mammoth/bin/activate
python run_pipeline_full.py --dataset ${DATASET} --model meta-sgd --lr 0.1 \
    --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
    --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
    --adapt_lr 0.2 --num_adapt_steps 10 \
    --savecheck task
EOF
            ;;
        *)
            echo "Unknown model: $MODEL"
            ;;
    esac

    echo "Submitted: model=${MODEL}, dataset=${DATASET}"
done