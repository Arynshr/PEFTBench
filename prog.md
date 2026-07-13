# Performance Analysis of Parameter-Efficient Fine-Tuning of Small Foundation Models under GPU Resource Constraints

> **Note for AI assistants / Claude / Copilot etc.:** This README is written to give any LLM full project context in one read. Refer to it before generating code, configs, or suggestions for this repo.

## 2. Project Summary

Conventional full fine-tuning of foundation models is computationally expensive. This project evaluates **Parameter-Efficient Fine-Tuning (PEFT)** — specifically **LoRA** and **QLoRA** — on compact transformer models, then extends the comparison to **inference-time performance** across three setups:

1. **Baseline inference** — unmodified pretrained model
2. **Finetuned inference** — LoRA/QLoRA-adapted model
3. **Parallelized/optimized inference** — served via SGLang (continuous batching, RadixAttention KV-cache, CUDA graph optimizations; multi-GPU tensor parallelism only if multiple GPUs are available)

**Important scope clarification:** This project does **not** build custom CUDA kernels or true multi-node distributed training. "Parallelized/CUDA-enhanced" inference means *using* SGLang's built-in optimizations on available GPU(s), not authoring new kernels. Multi-GPU tensor parallelism is included only if the environment provides more than one GPU.

## 3. Objectives

- Compare LoRA vs QLoRA on: training time, GPU memory, throughput, validation loss, accuracy, F1
- Compare baseline vs finetuned vs SGLang-optimized inference on: latency, throughput, memory footprint
- Evaluate mixed precision (FP16/BF16), gradient checkpointing, batch size tuning
- Produce a reproducible, containerized pipeline usable on both Windows (via Docker Desktop + WSL2 backend) and native WSL2/Linux

## 4. Models, Data, Techniques

| Category | Options |
|---|---|
| Foundation models (candidates) | Gemma 2B, Phi-3 Mini, Qwen2.5-1.5B, SmolLM2 (final choice depends on VRAM) |
| PEFT techniques | LoRA (frozen base + trainable low-rank matrices), QLoRA (LoRA + 4-bit NF4 quantization via bitsandbytes) |
| Datasets (candidates) | Dolly 15K, Alpaca, Financial PhraseBank, MedQA subset |
| GPU optimizations | Mixed precision (FP16/BF16), gradient checkpointing, batch size tuning |
| Inference serving | SGLang (OpenAI-compatible API server, LoRA adapter serving/switching) |

## 5. Tools & Technologies

Python 3.11+, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, Hugging Face Datasets, Scikit-learn, CUDA/cuDNN/NCCL, PyTorch Distributed & Slurm, TensorBoard, nvidia-smi, NVIDIA Nsight Systems, SGLang, Docker + NVIDIA Container Toolkit, (optional) Prometheus + Grafana for serving metrics.

## 6. Workflow

```
1. Environment setup (Python, PyTorch, CUDA, PEFT libs)
2. Dataset preparation (select, clean, tokenize, train/val split)
3. Baseline model evaluation (zero-shot performance, resource usage)
4a. Fine-tuning with LoRA        4b. Fine-tuning with QLoRA
5. GPU optimization (mixed precision, gradient checkpointing, batch tuning)
6. Comparative analysis — training: LoRA vs QLoRA (loss, accuracy, F1, memory, throughput, time)
7. Inference deployment via SGLang — baseline vs finetuned vs optimized-serving
8. Comparative analysis — inference: latency, throughput, memory across 3 setups
9. Results export & freeze (before GPU access ends)
```

## 7. Directory Structure

```
hpc-peft-project/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml         # services: training, sglang-serve, tensorboard, (prometheus/grafana optional)
│   └── .dockerignore
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── scripts/
│   ├── setup_env.sh
│   ├── run_training.sh
│   ├── run_baseline.sh
│   └── entrypoint.sh
├── src/
│   ├── data/preprocess.py
│   ├── models/load_model.py
│   ├── training/train_lora.py
│   ├── training/train_qlora.py
│   ├── evaluation/evaluate.py
│   └── utils/gpu_monitor.py
├── data/              # mounted volume, gitignored
├── checkpoints/       # mounted volume, gitignored — export LoRA/QLoRA adapter weights before GPU cutoff
├── logs/              # mounted volume, gitignored — export as static CSV/JSON before GPU cutoff
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

## 8. GPU Access & Environment Strategy

**Access method:** Remote Linux GPU server, reached via VPN + MobaXterm (SSH/SFTP client). This is not a local Windows/WSL2 dual-target scenario — all training/inference runs on the remote machine; MobaXterm is only used for terminal access and file transfer (checkpoints/logs) off the server.

**Package management — uv (chosen) vs conda:**
- **uv** is the default choice: fast installs, works in user space without sudo, sufficient as long as the remote host's NVIDIA driver + system CUDA are already present and compatible with the required PyTorch/bitsandbytes wheels.
- **Before committing to uv:** run `nvidia-smi` on the remote box, note the CUDA version, and confirm compatible PyTorch/bitsandbytes wheels exist for it.
- **Fallback to conda/mamba** only if there's a CUDA version mismatch — conda can install an alternate CUDA toolkit into a user-space environment without needing sudo; uv cannot do this since it assumes system CUDA is already correct.
- `bitsandbytes` (required for QLoRA) is the most CUDA-version-sensitive dependency — verify this one first regardless of which tool is chosen.

**Containerization (optional, not required):**
- Docker is a clean-code/learning choice here, not a necessity, since everything runs on one remote GPU server rather than needing to work across Windows + WSL2.
- Requires **root/sudo on the remote server** to install Docker Engine (not Docker Desktop — that's Windows/Mac only) and the NVIDIA Container Toolkit.
- **Check sudo access first** (`sudo -v`) before committing to this path. Shared/cluster GPU servers often restrict Docker and may offer **Singularity/Apptainer** instead (rootless, common in HPC).
- If Docker is used: base image = official NVIDIA CUDA image matched to PyTorch/CUDA version; multi-stage Dockerfile (build → slim runtime); `docker-compose.yml` orchestrates training, SGLang serving, and TensorBoard as separate services.
- If Docker is *not* used: same separation of concerns still applies — training, SGLang serving, and TensorBoard run as separate processes/sessions on the remote server (e.g., via `tmux`/`screen` or Slurm jobs) rather than separate containers.
- Either way: config (hyperparameters, model/dataset choice) stays externalized to `configs/*.yaml`, not hardcoded; secrets/paths via `.env`, not committed.

## 9. Deployment & Observability

- **Training-time observability:** TensorBoard, nvidia-smi, NVIDIA Nsight Systems
- **Serving-time (SGLang):** built-in metrics (latency, throughput, request stats); optionally Prometheus + Grafana as additional compose services
- **Deployment:** SGLang runs the finetuned model behind an OpenAI-compatible API; supports loading/switching LoRA adapters at runtime without reloading the base model

## 10. Critical Constraint — GPU Access Is Temporary

No retraining or re-inference will be possible once GPU access ends. Before cutoff, the following must be captured/exported:

1. LoRA/QLoRA adapter weights (small, prioritize over full merged models)
2. TensorBoard scalars exported as static CSV/JSON (not just left in a TensorBoard instance)
3. A batch of saved inference input/output pairs from SGLang (static demo, since re-inference won't be possible later)
4. Final metrics tables (loss, accuracy, F1, memory, throughput, latency) written directly into the report
5. Exact library/framework versions and configs documented for reproducibility by others
6. Optional: screen recording of the SGLang server responding to live queries as deployment proof

## 11. Explicit Scope Boundaries

**In scope:** LoRA/QLoRA fine-tuning, mixed precision, gradient checkpointing, batch tuning, SGLang-based optimized serving (including multi-GPU tensor parallelism *if* multiple GPUs are available).

**Out of scope:** Training foundation models from scratch, reinforcement learning techniques, authoring custom CUDA kernels, true multi-node distributed training across clusters.

## 12. Evaluation Metrics

- **Model quality:** validation loss, accuracy, F1-score
- **Training efficiency:** training time, GPU memory utilization, throughput
- **Inference efficiency (3-way comparison):** latency, throughput, memory footprint — baseline vs finetuned vs SGLang-optimized
