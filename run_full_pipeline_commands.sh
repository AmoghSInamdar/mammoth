#!/bin/bash

DATASET="${1:-smooth-mnist}"
MODELS_ARG="${2:-all}"

if [ "$MODELS_ARG" = "all" ]; then
    MODELS="sgd er derpp ewc_on agem lwf mer meta_sgd no_meta_sgd maml"
else
    MODELS="$MODELS_ARG"
fi

mkdir -p outputs

for MODEL in $MODELS; do
    case $MODEL in
        sgd)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                python run_pipeline_full.py --dataset $DATASET --model sgd --lr 0.1 \
                --adapt_lr 0.01 --num_adapt_steps 10 \
                --savecheck task \
                --do_train \
                >outputs/sgd_${DATASET}.out 2>outputs/sgd_${DATASET}.err &
            ;;
        er)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model=er --buffer_size=500 --lr=0.1 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                --do_train \
                >outputs/er_${DATASET}.out 2>outputs/er_${DATASET}.err &
            ;;
        derpp)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model derpp --lr 0.03 --buffer_size 500 --alpha 0.3 --beta 0.5 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                --do_train \
                >outputs/derpp_${DATASET}.out 2>outputs/derpp_${DATASET}.err &
            ;;
        ewc_on)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model=ewc_on --n_epochs=50 --lr=0.1 --e_lambda=10 --gamma=1 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                --do_train \
                >outputs/ewc_${DATASET}.out 2>outputs/ewc_${DATASET}.err &
            ;;
        agem)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model agem --buffer_size 500 --lr 0.03 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                >outputs/agem_${DATASET}.out 2>outputs/agem_${DATASET}.err &
            ;;
        lwf)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model lwf --lr 0.03 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                --do_train \
                >outputs/lwf_${DATASET}.out 2>outputs/lwf_${DATASET}.err &
            ;;
        mer)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                python run_pipeline_full.py --dataset $DATASET --model mer --lr 0.1 --beta 0.01 --gamma 0.03 --buffer_size 200 --minibatch_size 25 --n_epochs 1 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                >outputs/mer_${DATASET}.out 2>outputs/mer_${DATASET}.err &
            ;;
        meta_sgd)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                python run_pipeline_full.py --dataset $DATASET --model meta_sgd --lr 0.1 \
                --meta_method reptile --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.2 \
                --meta_adapt_lr 0.1 --meta_adapt_steps 10 --num_meta_examples 50 \
                --adapt_lr 0.1 --num_adapt_steps 10 \
                --savecheck task \
                >outputs/meta_sgd_${DATASET}.out 2>outputs/meta_sgd_${DATASET}.err &
            ;;
        no_meta_sgd)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                python run_pipeline_full.py --dataset $DATASET --model meta_sgd --lr 0.1 \
                --meta_method no_meta --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.1 \
                --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
                --adapt_lr 0.2 --num_adapt_steps 10 \
                --savecheck task \
                >outputs/no_meta_sgd_${DATASET}.out 2>outputs/no_meta_sgd_${DATASET}.err &
            ;;
        maml)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
                python run_pipeline_full.py --dataset $DATASET --model meta_sgd --lr 0.1 \
                --meta_method maml --meta_strategy parallel --num_lookahead_tasks 3 --meta_lr 0.001 \
                --meta_adapt_lr 0.2 --meta_adapt_steps 10 --num_meta_examples 50 \
                --adapt_lr 0.2 --num_adapt_steps 10 \
                --savecheck task \
                >outputs/maml_sgd_${DATASET}.out 2>outputs/maml_sgd_${DATASET}.err &
            ;;
        *)
            echo "Unknown model: $MODEL"
            ;;
    esac
done

wait