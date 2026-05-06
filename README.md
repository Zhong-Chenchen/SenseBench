# SenseBench: A Benchmark for Remote Sensing Low-Level Visual Perception and Description in Large Vision-Language Models

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Zhongchenchen/SenseBench)

> We will regularly maintain and update SenseBench and this repository to foster a comprehensive remote sensing community.
---

## 📢 Latest Updates
- **05-05-2026**: We release the complete SenseBench benchmark in the [Hugging Face Dataset](https://huggingface.co/datasets/Zhongchenchen/SenseBench). 🔥🔥

---

## ✨ Overview

<p align="center">
  <img src="docs/imgs/perdes.png" width="100%" alt="Overview of the SenseBench evaluation framework">
</p>

<p align="justify">
  <b>Figure:</b> Overview of the SenseBench evaluation framework. The upper part shows the <i>SensePerception</i> task taxonomy across input formats, distortion settings, and <i>whether</i>/<i>what</i>/<i>how</i> question types. The lower part illustrates <i>SenseDescription</i> examples for single and paired inputs, where responses are evaluated by <i>completeness</i>, <i>correctness</i>, and <i>faithfulness</i>, with <span style="color:red;">red</span> text indicating incorrect or unsupported statements.
</p>

---
## 🛠️ Evaluation Workflow

---
## LLM-as-Judge Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server Unbabel/M-Prometheus-7B --server-port 23333 --tp 2 --chat-template mistral
```

```bash
python eval.py --models gpt --use-llm --description-llm
```
