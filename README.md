# TraceDoctor
This repository provides example code for our paper "Improving LLM-based Log Parsing by Learning from Errors in Reasoning Traces."

```
├── Analyzer
│   ├── cluster_err.py
│   ├── error_label.py
│   ├── error_summary.py
│   ├── errset.py
│   └── model_analyzer.py
├── Generator
│   ├── log_aug.py
│   └── seeds_select.py
├── Model
│   └── model_download.py
├── README.md
├── high-level-error-types.md
└── requirements.txt

# Installation
First create a conda environment with Python 3.10:
```bash
conda create -n tracedoctor python=3.10
conda activate tracedoctor
pip install -r requirements.txt
```

# Model Download
In this work, we tested the proposed method on five different models: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus), and [Skywork-o1-Open-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B). These models can be downloaded via the following code. If an access token is required, it can be obtained from the official website.
```bash
cd Model
python model_download.py --model_name "deepseek/DeepSeek-R1-Distill-Llama-8B" --base_model_path "mode_path_here" --hf_token "your_token_here" --gpu_ids 0 1 --max_memory_per_gpu 20 --use_8bit
```

# Datasets Download
The test set used in this paper utilizes the official loghub-2.0 large-scale dataset. [loghub-2.0](https://github.com/logpai/loghub-2.0) can be downloaded from the official website.

# Data Augmentation
## Analyzer (Summarize Error Types)
(1) Initial Dataset Analysis for Student Models
```bash
cd Analyzer
python model_analyzer.py --model_path "/root/models/llama-3-70b" --input_csv "/root/data/logs/system_logs.csv" --output_dir "/root/results/analysis_output" --batch_size 4 --temperature 0.01 --sample_size 1000 --max_length 4096 --max_new_tokens 1024 --gpu_ids 0 1 2 3
```

(2) Error Set Generation
```bash
python errset.py --api_key "your-api-key" --input_file "/path/to/input/diff.csv" --output_file "/path/to/output/log_analysis_results.csv" --model "deepseek-chat" --batch_size 10
```

(3) Error Summary Generation
```bash
python error_summary.py --api_key "your-api-key-here" --input_file "/path/to/input/log_analysis_results.csv" --output_file "/path/to/output/error_types_analysis.md" --model "deepseek-chat"
```

(4) Classification Based on Error Type
```bash
python error_label.py --api_key "your-api-key" --md_summary_path "/path/to/error_analysis.md" --input_file "/path/to/input.csv" --output_file "/path/to/output.csv"
```

(5) High-Level Semantic Embedding Aggregation
```bash
python cluster_err.py --openai_api_key "your-openai-api-key" --openai_base_url https://api.chatanywhere.tech/v1 --md_file "/path/to/error_types_analysis.md" --csv_file "/path/to/label.csv" --output_dir "/path/to/output" --similarity_threshold 0.9
```

## Generator (Data Generation)
(1) Seed_logs Select
```bash
cd ../Generator
python seeds_select.py --labeled_csv "/path/to/labeled_data.csv" --summary_csv "/path/to/summary_data.csv" --output_csv "/path/to/output_seeds.csv" --num_seeds 2 --random_seed 42
```

(2) Enhancing Data Based on Three Strategies
```bash
python log_aug.py --api_key "your-deepseek-api-key" --input_file "input.csv" --output_file "output.json" --error_types_file "path/to/error_types.md" --augmentation_count 7
```

# Fine-tuning
In our paper, TraceDoctor is implemented by fine-tuning five large language models: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus), and [Skywork-o1-Open-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B). Thanks to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for implementing an efficient tool to fine-tune LLMs. The training steps of TraceDoctor are as follows:

(1) Prepare the environment

Please make sure to install the packages required. For issues related to environment, please refer to LLaMA-Factory v0.9.3.

(2) Register the training dataset

Register the dataset in data/dataset_info.json:
```json
{
  "train_dataset": {
    "file_name": "augmnent_datasets.json",
    "file_sha1": "",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

(3) Start training (with multiple GPUs)
```bash
llamafactory-cli train \
--model_name_or_path /path/to/your/model \
--stage sft \
--do_train \
--finetuning_type lora \
--lora_rank 8 \
--lora_target all \
--dataset Linux \
--dataset_dir /path/to/LLaMA-Factory/data \
--template deepseek \
--cutoff_len 4096 \
--max_samples 20000 \
--overwrite_cache \
--preprocessing_num_workers 16 \
--output_dir /path/to/output/results \
--logging_steps 10 \
--save_steps 500 \
--plot_loss \
--overwrite_output_dir \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 1.0e-4 \
--num_train_epochs 5.0 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--bf16
```

(4) Inference
```bash
llamafactory-cli chat \
--model_name_or_path /path/to/your/model \
--adapter_name_or_path /path/to/saves/sft \
--template deepseek \
--infer_backend vllm \
--trust_remote_code
```

For more information, please refer to LLaMA-Factory v0.9.3.
```
