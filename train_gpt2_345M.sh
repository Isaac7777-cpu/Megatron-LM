 #!/bin/bash

# Runs the "175B" parameter model

# This script is direct adaption from the work of Pengle Zhang

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/root/lyk/checkpoints
TENSORBOARD_LOGS_PATH=/root/lyk/tensorboard
VOCAB_FILE=/root/lyk/gpt2/vocab.json
MERGE_FILE=/root/lyk/gpt2/merges.txt
DATA_PATH="haha"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 768 
    --num-attention-heads 16 
    --seq-length 2048
    --max-position-embeddings 2048 
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 16
    --train-iters 500
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 3e-4
    --lr-decay-style cosine 
    --min-lr 3.0e-5
    --lr-warmup-fraction .001 
    --lr-decay-iters 430
    
    # ZPL: We don't have transformer engine
    --transformer-impl local

    # ZPL: We don't have apex
    --no-persist-layer-norm 
    --no-gradient-accumulation-fusion 
    --no-masked-softmax-fusion 
)

MODEL_PARALLEL_ARGS=(
    # ZPL: Just 1 card
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    # ZPL: Use mock dataset for demo, you can try it with openwebtext, see tools/preprocess_data*.py
    # --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --mock-data 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} /root/lyk/Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
