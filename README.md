# SenseBench: A Benchmark for Remote Sensing Low-Level Visual Perception and Description in Large Vision-Language Models

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/Zhongchenchen/SenseBench)


---

## 📢 Latest Updates
- **05-05-2026**: We release the complete SenseBench benchmark in the [Hugging Face Dataset](https://huggingface.co/datasets/Zhongchenchen/SenseBench). 🔥🔥

---

## LLM-as-Judge Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server Unbabel/M-Prometheus-7B --server-port 23333 --tp 2 --chat-template mistral
```

```bash
python eval.py --models gpt --use-llm --description-llm
```
