We leveraged [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) to pre-train GPT-4.8B with [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset. 

We have provided a slurm script [here](./pretrain_GPT_4.8B_baseline.slurm), which can be submitted to SLURM scheduler by `sbatch pretrain_GPT_4.8B_baseline.slurm`. However, before doing so, please clone Megatron-LM code.
