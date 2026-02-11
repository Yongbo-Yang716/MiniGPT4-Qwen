MiniGPT4-Qwen is a modular Vision-Language Model (VLM) that integrates:

- EVA-CLIP Vision Encoder
- Q-Former for visual token compression
- Qwen-Chat as the language backbone
- Optional LoRA-based fine-tuning

Built on top of the LAVIS framework, this project focuses on clean architecture, modular design, and research-friendly extensibility.

---

## 🧠 Architecture Overview

Image  
→ Vision Encoder (ViT)  
→ Visual Patch Tokens  
→ Q-Former Cross-Attention  
→ Query Tokens  
→ Linear Projection  
→ Qwen-Chat (LLM)  
→ Text Generation  

---

## ✨ Features

- Fully modular Vision-Language pipeline
- Config-driven training via YAML
- Single-GPU friendly
- Optional DeepSpeed pipeline parallelism
- LoRA integration support
- ChatML format compatibility

---
