#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)
# This script is deprecated as the data mocking does not seem to work in for Bert
# Majority of the content is taken from the examples and changed according to the help with Pengle Zhang.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/root/lyk/checkpoints #<Specify path>
TENSORBOARD_LOGS_PATH=/root/lyk/tensorboard #<Specify path>
VOCAB_FILE=/root/lyk/bert/bert-large-cased-vocab.txt #<Specify path to file>/bert-vocab.json
DATA_PATH="" #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 512 
    --max-position-embeddings 512 
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 32 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --fp16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 

    # ZPL: We don't have transformer engine
    --transformer-impl local

    # ZPL: We don't have apex
    --no-persist-layer-norm 
    --no-gradient-accumulation-fusion 
    --no-masked-softmax-fusion 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    # --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --split 949,50,1
    --mock-data
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} /root/lyk/Megatron-LM/pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    