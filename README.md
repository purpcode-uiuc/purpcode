# 🔮 PurpCode: Reasoning for Safer Code Generation

This repo includes the training and evaluation infrastructure for PurpCode. For other related resources, please check out:

* [📝 Paper](https://arxiv.org/abs/2507.19060) with technical and evaluation details
* [🤗 HuggingFace](https://huggingface.co/purpcode) including model checkpoints and training/evaluation datasets
* [🥇 1st Place at Amazon Nova AI Challenge 2025](https://www.amazon.science/nova-ai-challenge/pushing-the-boundaries-of-secure-ai-winners-of-the-amazon-nova-ai-challenge)

## Overview

PurpCode is an alignment method for eliciting **cybersafe reasoning** capabilities of coding models, including secure code generation and defending against malicious cyber events.
PurpCode includes two alignment stages:

1. **[Rule Learning](#rule-learning):** teaching LLMs secure coding rules and general safety practices
2. **[Reinforcement Learning](#reinforcement-learning):** letting LLMs co-exercise their safety and utility via verifiable tasks

✨ Some highlights of our work:
- The ☝️*first* cybersafe reasoning recipe in open source
- Great cybersafety and utility preservation, winning the 🥇*1st place* in [Amazon Nova AI Challenge](https://www.amazon.science/nova-ai-challenge/pushing-the-boundaries-of-secure-ai-winners-of-the-amazon-nova-ai-challenge)
- Fully 👐open-sourced, from models, data, to training/evaluation code and data synthesizers
- 🏎️ Fast RL with *Single-Step Dynamic Sampling* -- 12% faster, 15% less sample wasting, and better results than [DAPO](https://arxiv.org/abs/2503.14476)
- 📚 Supporting 13 evals, 90 CWEs, and 4 training objectives & rewards, covering cybersafety, utility, and overrefusal
- 🙅‍♂️ XSCode -- our home-made evaluator and the *first* benchmark for checking overrefusal in secure code generation
- ... and more details in the [paper](https://arxiv.org/abs/2507.19060)!

## Initial Setup

```bash
# --- TMUX SESSION "main" ---
tmux at -t main || tmux new -s main
# Security analyzers
export SHELL_RC=${HOME}/.bashrc # or ~/.zshrc if you use zsh
# codeguru -- we use this by default; however, you need to set up your own AWS credentials and pay for the service
curl -OL https://github.com/aws/aws-codeguru-cli/releases/download/0.2.4/aws-codeguru-cli.zip
unzip aws-codeguru-cli.zip -d ${HOME}
export PATH=$PATH:${HOME}/aws-codeguru-cli/bin/
sed -i '1i export PATH=$PATH:${HOME}/aws-codeguru-cli/bin/' ${SHELL_RC}

# codeql -- if you don't want to use codeguru, you can use codeql instead which only eats CPUs but the analyzer completeness and soundness can be different
#        -- you also need to set environment variable to PURPCODE_CODE_ANALYZER=codeql
wget https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.21.0/codeql-bundle-linux64.tar.gz
tar -xf codeql-bundle-linux64.tar.gz -C ${HOME}
export PATH=$PATH:${HOME}/codeql/
sed -i '1i export PATH=$PATH:${HOME}/codeql/' ${SHELL_RC}

tmux detach
# --------------------------

# --- TMUX SESSION "sandbox" ---
tmux new -s sandbox
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20241204
tmux detach
# ------------------------------
```

## Rule Learning

We will go through the example based on `Qwen/Qwen2.5-14B-Instruct-1M`:

### Rejection Sampling

```bash
# --- TMUX SESSION "sgl" ---
conda create -n sgl python=3.12 -y
conda activate sgl
pip install --upgrade pip
pip install "sglang[all]>=0.4.9.post2" "sglang-router" "huggingface-hub"

huggingface-cli download Qwen/Qwen2.5-14B-Instruct-1M
python3 -m sglang_router.launch_server --model Qwen/Qwen2.5-14B-Instruct-1M --dp-size 8 --port 30000 --host 0.0.0.0 | tmux detach
# --------------------------

# --- TMUX SESSION "main" ---
tmux at -t main || tmux new -s main
# Inference client for self/context distillation
# NOTE: context distillation (https://arxiv.org/abs/2209.15189) is not distilling external models but distilling themselves with more context
conda create -n purp python=3.12 -y
conda activate purp
pip install -r requirements.txt
# Sampling
python datagen/ctxdistill/distill_main.py --model openai/Qwen/Qwen2.5-14B-Instruct-1M --sample-per-prompt 8 --concurrency 400
# ---------------------------

# --- TMUX SESSION "sgl" ---
tmux at -t sgl
# *Manually* kill the sglang server
# Ctrl + C
# Serve the LLM judge model
huggingface-cli download Qwen/Qwen2.5-32B-Instruct
python3 -m sglang_router.launch_server --model Qwen/Qwen2.5-32B-Instruct --dp-size 8 --port 30000 --host 0.0.0.0
tmux detach
# --------------------------

# --- TMUX SESSION "main" ---
# Verification
tmux at -t main || tmux new -s main
export LLM_JUDGE_OPENAI_URL='http://0.0.0.0:30000/v1'
python datagen/ctxdistill/post.py --generation-path Qwen2.5-14B-Instruct-1M.distill.train.jsonl
# ----------------------------
```

### Running SFT

```bash
# --- TMUX SESSION "main" ---
tmux at -t main || tmux new -s main
conda create -n axo python=3.12 -y
conda activate axo
git clone git@github.com:axolotl-ai-cloud/axolotl.git
cd axolotl
pip3 install torch --index-url https://download.pytorch.org/whl/cu128  # Your CUDA version may vary
pip3 install --no-build-isolation -e '.[flash-attn,deepspeed]'

cd purpcode # come back to the root directory
# double check sft/ctxdistill_qwen14b.yaml to make sure the paths are aligned well
axolotl train sft/ctxdistill_qwen14b.yaml --deepspeed deepspeed_configs/zero3.json
# --> outputs/purpcode-14b-ctxdistill
```

## Reinforcement Learning

```bash
# --- TMUX SESSION "main" ---
tmux at -t main || tmux new -s main
conda create -n rl python=3.12 -y
conda activate rl

git clone git@github.com:ganler/verl.git
cd verl
git checkout opt

pip install -e . --upgrade
pip install vllm==0.8.2
pip install flash-attn --no-build-isolation --upgrade

cd purpcode # come back to the root directory
python rl/data/merge.py --datasets purpcorn/code-r1-46k-leetcode2k-kodcode purpcorn/rl-codesec-78k purpcorn/rl-secqa-11k purpcorn/rl-safety-8k-single-turn \
                        --skip     Qwen2.5-14B-Instruct-1M.ez_task_ids.txt

# ---------------------------

# --- TMUX SESSION "sgl" (remote machine) ---
# Do it in another machine (assuming ip=a.b.c.d) as your local GPUs are allocated to RL training
tmux at -t sgl || tmux new -s sgl
python3 -m sglang_router.launch_server --model Qwen/Qwen2.5-32B-Instruct --dp-size 8 --port 30000 --host 0.0.0.0 | tmux detach
# -------------------------------------------

# --- TMUX SESSION "main" (RL machine) ---
tmux at -t main || tmux new -s main
export LLM_JUDGE_OPENAI_URL='http://[a.b.c.d]:30000/v1' # replace [a.b.c.d] with a true IP address
conda activate rl
bash rl/main_grpo_qwen14b.sh
# -------------------------------------------
```

## Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python eval/main.py --task "purpcode/CyberSecEval-SCG"   --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/CodeLMSec"          --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/CWEval"             --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/CyberSecEval-MITRE" --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/CyberSecEval-FRR"   --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/XSCode"             --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/XSTest"             --model purpcode/purpcode-14b-rl
python eval/main.py --task "purpcode/PHTest"             --model purpcode/purpcode-14b-rl
```

Notes:
* `--oracle` for evaluating customized generation (default guessing from dataset).

## Acknowledgements

- [Amazon Nova AI Challenge](https://www.amazon.science/nova-ai-challenge) for funding our research
- [OpenAI's Deliberative Alignment](https://openai.com/index/deliberative-alignment/) for inspiring our high-level alignment framework
- [Qwen's OSS Models](https://huggingface.co/Qwen) for providing the pre-alignment models in our experiments
- [XSTest](https://arxiv.org/abs/2308.01263) for inspiring our XSCode dataset

## References

```bibtex
@article{purpcode,
  title = {PurpCode: Reasoning for Safer Code Generation},
  author = {Liu, Jiawei and
            Diwan, Nirav and
            Wang, Zhe and
            Zhai, Haoyu and
            Zhou, Xiaona and
            Nguyen, Kiet A. and
            Yu, Tianjiao and
            Wahed, Muntasir and
            Deng, Yinlin and
            Benkraouda, Hadjer and
            Wei, Yuxiang and
            Zhang, Lingming and
            Lourentzou, Ismini and
            Wang, Gang},
  journal = {arXiv preprint arXiv:2507.19060},
  year = {2025},
}
```
