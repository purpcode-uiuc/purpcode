## Rule2Code

This directory contains scripts to generate code examples based on security rules (e.g., CWE rules, CodeGuru detectors) and then post-process them.

### Data Generation

Launch a local model server using sglang and generate vulnerable code examples based on security rules.

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

# Generate vulnerable code examples based on CWE rules.
python datagen/rule2code/cwe2code.py --parallel 256 --output_path outputs/rule2code/cwe2code.jsonl --depth 1

# Generate vulnerable code examples based on CodeGuru detectors.
python datagen/rule2code/guru2code.py --parallel 256 --output_path outputs/rule2code/guru2code.jsonl --depth 1

tmux detach
# ---------------------------
```

**Note:**
- You can configure other helpful-only models to generate code examples by adjusting the model parameter in the docker command.

### Data Post-processing

Scrape bandit rules from the Ruff documentation, then process the generated data by extracting code examples, running static analysis, adding analyzer results to the examples, and reformatting them.

```bash
# Scrape bandit rules.
python datagen/rule2code/get_bandit_rules.py --output_file bandit_rules.json

# Process the generated cwe2code data.
python datagen/rule2code/post_process.py --input_path outputs/rule2code/cwe2code.jsonl --ruff_rules_path bandit_rules.json --source cwe2code

# Process the generated guru2code data.
python datagen/rule2code/post_process.py --input_path outputs/rule2code/guru2code.jsonl --ruff_rules_path bandit_rules.json --source guru2code
```

**Note:**
- The processed output file will be generated with `.processed.jsonl` suffix.
