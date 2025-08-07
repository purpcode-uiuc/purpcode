### XSCode
[XSCode](https://huggingface.co/datasets/purpcode/XSCode) is an overrefusal benchmark for secure code generation. While benchmarks like CyberSecEval FRR prompts are lengthy and specifically target malicious cyber activities, XSCode contains 589 short and harmless code-generation prompts that do not contain any built-in code security bias.

### XSCode Generation

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python eval/compile_xscode/main.py
```

Generation requires AWS bedrock access.

### XSCode Eval

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python eval/main.py --task "purpcode/xscode" --model <MODEL_NAME>
```
