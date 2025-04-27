#!/bin/bash
#SBATCH -A A-ccsc
#SBATCH --time=01:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH --mail-user=YOUR@EMAIL.ADDRESS
#SBATCH --mail-type=all
#SBATCH --nodelist=c611-041


# activate virtual env
#ml gcc/12.2.0
#source /work/09308/zhengmk/python_vir_envs/optimus-cc-frontera/bin/activate
#export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2019.5.281/linux/compiler/lib/intel64_lin:/opt/intel/itac/2019.5.041/intel64/bin/rtlib:/opt/apps/intel19/python3/3.9.2/lib:/opt/intel/oneapi/ccl/2021.9.0/lib/cpu_gpu_dpcpp:/opt/intel/oneapi/mpi/2021.9.0/libfabric/lib:/opt/intel/oneapi/mpi/2021.9.0/lib/release:/opt/intel/oneapi/mpi/2021.9.0/lib:/opt/apps/gcc/12.2.0/lib64:/opt/apps/gcc/12.2.0/lib:/opt/apps/hwloc/1.11.12/lib:/opt/apps/pmix/3.1.4/lib

source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate

ml cuda/12.4
export LD_LIBRARY_PATH=/opt/apps/gcc14/openmpi/5.0.5/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hcoll/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/home1/09308/zhengmk/usr:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/include:/opt/apps/cuda/12.4/targets/sbsa-linux/include
export CXX=g++
export TORCH_CUDA_ARCH_LIST=9.0

ln -s ~/bin/python3.9-aarch64-config python3-config
export PATH=/work/09308/zhengmk/optimus-cc/range_topk/slurm_scrips/tacc_scripts/vista/slurm_logs/355M/baseline:$PATH

which python3
#which gcc


GPUS_PER_NODE=1
NODE_LIST=$(srun hostname)
NODE_LIST_UNIQUE=$(echo "${NODE_LIST[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')

export GPUS_PER_NODE=$GPUS_PER_NODE
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$NODE_LIST_UNIQUE" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT=${MASTER_PORT}"

# Change for multinode config
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
DATA_PARALLEL=$(($WORLD_SIZE/$PIPELINE_PARALLEL/$TENSOR_PARALLEL))

echo "TENSOR_PARALLEL=${TENSOR_PARALLEL}"
echo "PIPELINE_PARALLEL=${PIPELINE_PARALLEL}"
echo "DATA_PARALLEL=${DATA_PARALLEL}"


# DATA_PATH=<Specify path and file prefix>_text_document
DATA_PATH=/scratch/09308/zhengmk/openwebtxt/preprocessed_data_text_document
CHECKPOINT_PATH=/scratch/09308/zhengmk/optimus-cc_checkpoints/GPT2/6.7B
# CHECKPOINT_PATH=/scratch/09308/zhengmk/optimus-cc_checkpoints/GPT2/6.7B/baseline_TP${TENSOR_PARALLEL}_PP${PIPELINE_PARALLEL}

output_dir=/work/09308/zhengmk/optimus-cc/range_topk/slurm_scrips/tacc_scripts/logs/6.7B/dense_srun_TP${TENSOR_PARALLEL}_PP${PIPELINE_PARALLEL}_DP${DATA_PARALLEL}
mkdir -p  $output_dir
mkdir -p  $output_dir/jsons

log_txt_file=$output_dir/dense_ppl_log.txt

global_batch_size=512
echo "global_batch_size=${global_batch_size}"

timestamp=$(date +%s)
log_txt_file=$output_dir/loss_ppl_log_${timestamp}.txt

srun python /work/09308/zhengmk/optimus-cc/range_topk/pretrain_gpt.py \
       --tensor-model-parallel-size $TENSOR_PARALLEL \
       --pipeline-model-parallel-size $PIPELINE_PARALLEL \
       --num-layers 60 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --micro-batch-size 8 \
       --global-batch-size $global_batch_size \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /work/09308/zhengmk/optimus-cc/range_topk/tools/gpt2-vocab.json \
       --merge-file /work/09308/zhengmk/optimus-cc/range_topk/tools/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --experiment_name GPT_6.7B_Baseline \
       --accumulate-allreduce-grads-in-fp32 \
       --bf16 \
       --log_txt_file $log_txt_file \
       |& tee $output_dir/log_${timestamp}.txt

set +x