## Vul2Prompt

This directory contains scripts to generate vulnerability-inducing prompts based on vulnerable code examples and then post-process them.

### Data Generation

Launch a local model server using sglang and generate prompts from vulnerable code examples using various attack strategies.

```bash
# --- TMUX SESSION "sgl" ---
tmux new -s sgl
docker run --gpus all --shm-size 32g -p 30000:30000 --network=host \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-0528 \
  --tp 8 --trust-remote-code --port 30000 \
  --torch-compile-max-bs 8 & tmux detach
# --------------------------

# --- TMUX SESSION "main" ---
tmux at -t main || tmux new -s main

python datagen/vul2prompt/vul2prompt.py --parallel 256 --output_path outputs/vul2prompt/vul2prompt.jsonl --depth 1 --strategies "general"

tmux detach
# ---------------------------
```

**Note:**
- You can configure other helpful-only models to generate prompts by adjusting the model parameter in the docker command.
- The `--strategies` argument supports single strategy (e.g., `general`), multiple strategies separated by spaces (e.g., `"general benign2vul"`), or `all` to run all available strategies. Available strategies are: general, benign2vul, vul2vul.
- You can define custom attack strategies in `datagen/vul2prompt/attack_strategies.py`.

### Data Post-processing

Process the generated data by extracting prompts, adding metadata from the seed code file, and splitting the output into separate files based on the attack strategy.

```bash
python datagen/vul2prompt/post_process.py --input_path outputs/vul2prompt/vul2prompt.jsonl --seed_code_path outputs/rule2code/cwe2code.processed.jsonl
```

**Note:**
- The output files will be generated with strategy-specific suffixes automatically (e.g., `.general.jsonl`, `.benign2vul.jsonl`, `.vul2vul.jsonl`).
- The `--seed_code_path` should point to seed code data with analyzer results processed by `datagen/rule2code/post_process.py`.
