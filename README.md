# 🔮 PurpCode: Reasoning for Safer Code Generation

This repo includes the training and evaluation infrastructure for PurpCode. For other resources, please check out:

* [📝 Paper](https://arxiv.org/abs/2507.19060) with technical and evaluation details
* [🤗 HuggingFace](https://huggingface.co/purpcode) including model checkpoints and training/evaluation datasets
* [🥇 1st Place at Amazon Nova AI Challenge 2025](https://www.amazon.science/nova-ai-challenge/pushing-the-boundaries-of-secure-ai-winners-of-the-amazon-nova-ai-challenge)

## Overview

PurpCode is an alignment method and a fully open-source recipe (data, model, and code) for eliciting **cybersafe reasoning** capabilities of coding models, including secure code generation and defending against malicious cyber events.
PurpCode includes two alignment stages:

1. **[Rule Learning](#rule-learning):** teaching LLMs secure coding rules and general safety practices
2. **[Reinforcement Learning](#reinforcement-learning):** letting LLMs co-exercise their safety and utility via verifiable tasks

We also curate comprehensive safety data via internal red teaming and use various evaluators covering cybersafety, utility, and overrefusal.

## Rule Learning

TBD

## Reinforcement Learning

TBD

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
