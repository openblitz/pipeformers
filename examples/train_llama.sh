GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

mpirun --host localhost:${GPUS} -np ${GPUS}                                 \
    -x TORCH_SHOW_CPP_STACKTRACES=1                                         \
    -x WANDB_PROJECT=pipeformers                                            \
    -x WANDB_RUN_GROUP=llama-tiger-pretrain-en                              \
    python -m mpi4py -m pipeformers.trainer                                 \
        --base-model meta-llama/Meta-Llama-3.1-8B                           \
        --deepspeed-config examples/deepspeed_config.json                   \
        --epochs 1                                                          \
        --dataset Salesforce/wikitext                                       \
        --dataset-config wikitext-2-raw-v1                                  \
        --dataset-split 0.2                                                 \
        --output-dir .state_dicts/llama_tiger                               \
        --pipeline-stages ${GPUS}                                           \
        --sequence-length 5                                                 \
        --state_dict .state_dicts/meta-llama--Meta-Llama-3.1-8B-Instruct.pt
        # --wandb-logging


# 3.921875