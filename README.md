<div align="center">

#  📹 Awesome Video World Models with AR Diffusion
</div>

### Overview
This repository surveys autoregressive-diffusion video generation models (AR video diffusion). They combine diffusion-level quality with autoregressive controllability, and become real-time after few-step distillation—emerging as a new paradigm for video generation and video world models. We cover recent work on both multi-step AR video diffusion and distilled few-step AR models, from core algorithms to diverse downstream tasks.

### Table of Contents

- [1. Algorithm](#1-algorithm)
    - [1.1 AR Diffusion (native pretraining)](#11-ar-diffusion-for-streaming-generation)
    - [1.2 AR Diffusion Distillation for Real-time Generation (post training)](#12-ar-diffusion-distillation-for-real-time-streaming-generation)
    - [1.3 Long Video Generation](#13-long-video-generation)
- [2. Application](#2-application)
    - [2.1 Open-source AR Video Foundation Models](#21-general-video-world-model)
    - [2.2 Video Action World Model](#22-video-action-world-model)
    - [2.2 Avtar](#23-avtar)
- [3. Infrastructure](#3-infrastructure)
    - [3.1 Sparse Attention](#31-sparse-attention)
    - [3.2 Caching](#32-caching)
    - [3.2 Quantized Attention](#33-quantized-attention)
- [Contributing](#contributing)
- [Acknowledgment](#acknowledgment)


## 1. Algorithm
## 1.1 AR Diffusion (native pretraining)

These methods focus on basic **AR Diffusion (where each chunk/frame is generated via diffusion and the frames are AR)**, which, while enabling streaming generation, still rely on multi-step sampling, making real-time generation challenging.

* **Diffusion Forcing**: "Next-token Prediction Meets Full-Sequence Diffusion". [![arXiv](https://img.shields.io/badge/arXiv-2407.01392-b31b1b.svg)](https://arxiv.org/abs/2407.01392) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://boyuan.space/diffusion-forcing) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/buoyancy99/diffusion-forcing)
* **DFoT**, "Diffusion Forcing Transformer with History Guidance". [![arXiv](https://img.shields.io/badge/arXiv-2502.01392-b31b1b.svg)](https://arxiv.org/abs/2502.06764) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://boyuan.space/history-guidance/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/kwsong0113/diffusion-forcing-transformer)
* **AR-Diffusion**, "AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion". [![arXiv](https://img.shields.io/badge/arXiv-2503.07418-b31b1b.svg)](https://arxiv.org/abs/2503.07418)[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/iva-mzsun/AR-Diffusion)
* **PFVG**, "Pack and force your memory: Long-form and consistent video generation". [![arXiv](https://img.shields.io/badge/arXiv-2510.01784-b31b1b.svg)](https://arxiv.org/abs/2510.01784) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://wuxiaofei01.github.io/PFVG/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/wuxiaofei01/PFVG)
* **BAgger**, "BAgger: Backwards Aggregation for
Mitigating Drift in Autoregressive Video Diffusion Models". [![arXiv](https://img.shields.io/badge/arXiv-2512.12080-b31b1b.svg)](https://arxiv.org/abs/2512.12080) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://ryanpo.com/bagger/)
* **Resampling Forcing**, "End-to-End Training for Autoregressive Video Diffusion via Self-Resampling". [![arXiv](https://img.shields.io/badge/arXiv-2512.15702-b31b1b.svg)](https://arxiv.org/abs/2512.15702) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://guoyww.github.io/projects/resampling-forcing/) 


## 1.2 🔥 AR Diffusion Distillation for Real-time Generation (post training)
This category of algorithms focuses on **distilling multi-step bidirectional diffusion models into few-step AR models**, specifically tailored for **real-time streaming generation**.

- From Multi-step Bidirectional Diffusion to Few-step Autoregressive Generators:
    * [⭐] **CausVid**, "From Slow Bidirectional to Fast Causal Video Generators". [![arXiv](https://img.shields.io/badge/arXiv-2412.07772-b31b1b.svg)](https://arxiv.org/abs/2412.07772) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://causvid.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/tianweiy/CausVid)
    * [⭐] **Self Forcing**, "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion". [![arXiv](https://img.shields.io/badge/arXiv-2506.08009-b31b1b.svg)](https://arxiv.org/abs/2506.08009) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://seaweed-apt.com/2) 
    * [⭐] **Causal Forcing**, "Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2602.02214-b31b1b.svg)](https://arxiv.org/abs/2602.02214) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://thu-ml.github.io/CausalForcing.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/thu-ml/Causal-Forcing) 

<div align=center>
<img width="582" height="59" alt="image" src="https://github.com/user-attachments/assets/cae08ae6-8adb-4249-b1b4-232dc332f943" />
</div>
<br>

- Further Improvements:
    * (Adversarial distillation) **Seaweed APT2**, "Autoregressive Adversarial Post-Training
    for Real-Time Interactive Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2506.08009-b31b1b.svg)](https://arxiv.org/pdf/2506.09350) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://seaweed-apt.com/2)
    * (One-step distillation) **ASD**, "Towards One-Step Causal Video Generation via Adversarial Self-Distillation". [![arXiv](https://img.shields.io/badge/arXiv-2511.01419-b31b1b.svg)](https://arxiv.org/abs/2511.01419) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/BigAandSmallq/SAD)
    * (Reinforcement learning) **Reward Forcing**, "Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation". [![arXiv](https://img.shields.io/badge/arXiv-2512.04678-b31b1b.svg)](https://arxiv.org/abs/2512.04678) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://reward-forcing.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/JaydenLyh/Reward-Forcing)

## 1.3 Long Video Generation

* Long video quality

    * **LongLive**, "LongLive: Real-time Interactive Long Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2509.22622-b31b1b.svg)](https://arxiv.org/abs/2509.22622) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://nvlabs.github.io/LongLive/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/NVlabs/LongLive) 
    * **Rolling Forcing**, "Rolling Forcing: Autoregressive Long Video Diffusion in Real Time". [![arXiv](https://img.shields.io/badge/arXiv-2509.25161-b31b1b.svg)](https://arxiv.org/abs/2509.25161) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://kunhao-liu.github.io/Rolling_Forcing_Webpage/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/TencentARC/RollingForcing)
    * **Self Forcing++**, "Self-Forcing++: Towards Minute-Scale High-Quality Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2510.02283-b31b1b.svg)](https://arxiv.org/abs/2510.02283) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://self-forcing-plus-plus.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/justincui03/Self-Forcing-Plus-Plus)
    * **Infinite Forcing**, [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/SOTAMak1r/Infinite-Forcing)
    * **Infinity-RoPE**, "Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout". [![arXiv](https://img.shields.io/badge/arXiv-2511.20649-b31b1b.svg)](https://arxiv.org/abs/2511.20649)  [![Website](https://img.shields.io/badge/Website-Link-blue)](https://infinity-rope.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/yesiltepe-hidir/infinity-rope) 
    * **Deep Forcing**, "Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression". [![arXiv](https://img.shields.io/badge/arXiv-2512.05081-b31b1b.svg)](https://arxiv.org/abs/2512.05081) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://cvlab-kaist.github.io/DeepForcing/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/cvlab-kaist/DeepForcing) 

* Long-term Memory
  * **WORLDMEM**, "WORLDMEM: Long-term Consistent
World Simulation with Memory". [![arXiv](https://img.shields.io/badge/arXiv-2504.12369-b31b1b.svg)](https://arxiv.org/pdf/2504.12369) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://xizaoqu.github.io/worldmem/)[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/xizaoqu/WorldMem)
    * **VRAG**, "Learning World Models for Interactive Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2505.21996-b31b1b.svg)](https://arxiv.org/abs/2505.21996) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://sites.google.com/view/vrag)[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/yeyutaihan/vrag) 
    * **Context as Memory**, "Context as Memory: Scene-Consistent Interactive Long Video
Generation with Memory Retrieval". [![arXiv](https://img.shields.io/badge/arXiv-2506.03141-b31b1b.svg)](https://arxiv.org/pdf/2506.03141) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://context-as-memory.github.io/)
    * **Memory Forcing**, "Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft". [![arXiv](https://img.shields.io/badge/arXiv-2510.03198-b31b1b.svg)](https://arxiv.org/abs/2510.03198) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://junchao-cs.github.io/MemoryForcing-demo/)
    * **MemFlow**, "MemFlow: Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives". [![arXiv](https://img.shields.io/badge/arXiv-2512.14699-b31b1b.svg)](https://arxiv.org/abs/2512.14699) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://sihuiji.github.io/MemFlow.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/KlingTeam/MemFlow)
    * **StableWorld**, "StableWorld: Towards Stable and Consistent Long Interactive Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2601.15281-b31b1b.svg)](https://arxiv.org/abs/2601.15281) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://sd-world.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/xbyym/StableWorld)
    * **LIVE**, "LIVE: Long-horizon Interactive Video World Modeling". [![arXiv](https://img.shields.io/badge/arXiv-2602.03747-b31b1b.svg)](https://arxiv.org/abs/2602.03747) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://junchao-cs.github.io/LIVE-demo/) 
    * **Infinite-World**, "Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory". [![arXiv](https://img.shields.io/badge/arXiv-2602.02393-b31b1b.svg)](https://arxiv.org/abs/https://arxiv.org/pdf/2602.02393)
    * **Context Forcing**, "Context Forcing: Consistent Autoregressive Video Generation with Long Context". [![arXiv](https://img.shields.io/badge/arXiv-2602.06028-b31b1b.svg)](https://arxiv.org/abs/2602.06028) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://chenshuo20.github.io/Context_Forcing/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/TIGER-AI-Lab/Context-Forcing)
## 2. Application

## 2.1 Open-source AR Video Foundation Models
* **SkyReels**, "SkyReels-V2: Infinite-length Film Generative Model". [![arXiv](https://img.shields.io/badge/arXiv-2504.13074-b31b1b.svg)](https://arxiv.org/abs/2504.13074) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://www.skyreels.ai/home?utm_campaign=github_SkyReels_V2) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/SkyworkAI/SkyReels-V2)
* **MAGI-1**, "MAGI-1: Autoregressive Video Generation at Scale". [![arXiv](https://img.shields.io/badge/arXiv-2505.13211-b31b1b.svg)](https://arxiv.org/abs/2505.13211) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://sand.ai/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/SandAI-org/MAGI-1)


## 2.2 Video Action World Model

* **Genie3**, "Genie3: A general purpose world model that can generate an unprecedented diversity of interactive environments". [![Website](https://img.shields.io/badge/Website-Link-blue)](https://deepmind.google/models/genie/) 

* **Matrix-game 2.0**, "Matrix-game 2.0: An open-source real-time and
streaming interactive world model". [![arXiv](https://img.shields.io/badge/arXiv-2508.13009-b31b1b.svg)](https://arxiv.org/pdf/2508.13009) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://matrix-game-v2.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) 

* **PAN**, "PAN: A World Model for General, Interactable, and Long-Horizon
World Simulation". [![arXiv](https://img.shields.io/badge/arXiv-2511.09057-b31b1b.svg)](https://arxiv.org/pdf/2511.09057v1) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://panworld.ai/)

* **RELIC**, "RELIC: Interactive Video World Model
with Long-Horizon Memory". [![arXiv](https://img.shields.io/badge/arXiv-2512.04040-b31b1b.svg)](https://arxiv.org/pdf/2512.04040) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://relic-worldmodel.github.io/)

* **HY-WorldPlay**, "WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling". [![arXiv](https://img.shields.io/badge/arXiv-2512.14614-b31b1b.svg)](https://arxiv.org/abs/2512.14614) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://3d-models.hunyuan.tencent.com/world/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/Tencent-Hunyuan/HY-WorldPlay) 

* **Yume 1.5**, "Yume-1.5: A Text-Controlled Interactive World Generation Model". [![arXiv](https://img.shields.io/badge/arXiv-2512.22096-b31b1b.svg)](https://arxiv.org/abs/2512.22096) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://stdstu12.github.io/YUME-Project/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/stdstu12/YUME)

* **LingBot-World**, "Advancing Open-source World Models". [![arXiv](https://img.shields.io/badge/arXiv-2601.20540-b31b1b.svg)](https://arxiv.org/abs/2601.20540) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://lingbotai.world/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/Robbyant/lingbot-world)



## 2.3 Avtar
* **RealVideo**, [![Website](https://img.shields.io/badge/Website-Link-blue)](https://z.ai/blog/realvideo) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/zai-org/RealVideo) 
* **LiveAvatar**, "Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length". [![arXiv](https://img.shields.io/badge/arXiv-2512.04677-b31b1b.svg)](https://arxiv.org/abs/2512.04677) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://liveavatar.github.io/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/Alibaba-Quark/LiveAvatar)
* **SoulX-FlashTalk**, "SoulX-FlashTalk: Real-Time Infinite Streaming of Audio-Driven Avatars via Self-Correcting Bidirectional Distillation". [![arXiv](https://img.shields.io/badge/arXiv-2512.23379-b31b1b.svg)](https://arxiv.org/abs/2512.23379) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://soul-ailab.github.io/soulx-flashtalk/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/Soul-AILab/SoulX-FlashTalk)
* **LiveTalk**, "LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation". [![arXiv](https://img.shields.io/badge/arXiv-2512.23576-b31b1b.svg)](https://arxiv.org/abs/2512.23576) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/GAIR-NLP/LiveTalk)
* **Avatar Forcing**, "Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation". [![arXiv](https://img.shields.io/badge/arXiv-2601.00664-b31b1b.svg)](https://arxiv.org/abs/2601.00664) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://taekyungki.github.io/AvatarForcing/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/TaekyungKi/AvatarForcing)
## 3 Infrastructure

## 3.1 Sparsity

* **Dummy Forcing**, "Efficient Autoregressive Video Diffusion with Dummy Head". [![arXiv](https://img.shields.io/badge/arXiv-2601.20499-b31b1b.svg)](https://arxiv.org/abs/2601.20499) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://csguoh.github.io/project/DummyForcing/) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/csguoh/DummyForcing)
* **Light Forcing**, "Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention". [![arXiv](https://img.shields.io/badge/arXiv-2602.04789-b31b1b.svg)](https://arxiv.org/abs/2602.04789) [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/chengtao-lv/LightForcing)
* **Fast Autoregressive Video Diffusion and World Models with Temporal Cache Compression and Sparse Attention**, "Fast Autoregressive Video Diffusion and World Models with Temporal Cache Compression and Sparse Attention". [![arXiv](https://img.shields.io/badge/arXiv-2602.01801-b31b1b.svg)](https://arxiv.org/abs/2602.01801) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://dvirsamuel.github.io/fast-auto-regressive-video/)
* **TokenTrim**, "TokenTrim: Inference-Time Token Pruning for Autoregressive Long Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2602.00268-b31b1b.svg)](https://arxiv.org/abs/2602.00268) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://arielshaulov.github.io/TokenTrim/)
* **PaFu-KV**, "Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion". [![arXiv](https://img.shields.io/badge/arXiv-2601.21896-b31b1b.svg)](https://arxiv.org/abs/2601.21896)
* **MonarchRT**, "MonarchRT: Efficient Attention for Real-Time Video Generation". [![arXiv](https://img.shields.io/badge/arXiv-2602.12271-b31b1b.svg)](https://arxiv.org/abs/2602.12271)
  
## 3.2 Caching
* **FlowCache**, "Flow caching for autoregressive video generation". [![arXiv](https://img.shields.io/badge/arXiv-2602.10825-b31b1b.svg)](https://arxiv.org/abs/2602.10825)  [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/mikeallen39/FlowCache)


## 3.3 Quantization
* **Quant VideoGen**, "Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization". [![arXiv](https://img.shields.io/badge/arXiv-2602.02958-b31b1b.svg)](https://arxiv.org/abs/2602.02958)

### 3.4 Others
* **SCD**, "Causality in Video Diffusers is Separable from Denoising". [![arXiv](https://img.shields.io/badge/arXiv-2602.10095-b31b1b.svg)](https://arxiv.org/abs/2602.10095)

---
### Contributing
We have not yet compiled an exhaustive list of all related work; we apologize for any omissions and welcome pull requests to merge them in. We also welcome high-level categorization and synthesis.
### Acknowledgment
We refer to the format of [Awesome-World-Models](https://github.com/knightnemo/Awesome-World-Models). 
