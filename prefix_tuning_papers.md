# NeurIPS 2025 papers using soft prompts or prefix-based fine-tuning

Found 16 papers matching keywords: soft prompt, soft-prompt, softprompt, prefix tuning, prompt tuning, p-tuning, p tuning, soft prefix, prefix prompt, prompt-based tuning.

## All You Need is One: Capsule Prompt Tuning with a Single Vector

- Authors: Yiyang Liu, James Liang, Heng Fan, Wenhao Yang, Yiming Cui, Xiaotian Han, Lifu Huangg, Dongfang Liu, Qifan Wang, Cheng Han
- Session: Poster
- Matched terms: prompt tuning
- Critique: Relies on a single capsule prompt without clear ablations on robustness across highly diverse tasks, so generality claims may be optimistic.

Prompt-based learning has emerged as a parameter-efficient finetuning (PEFT) approach to facilitate Large Language Model (LLM) adaptation to downstream tasks by conditioning generation with task-aware guidance. Despite its successes, current prompt-based learning methods heavily rely on laborious grid searching for optimal prompt length and typically require considerable number of prompts, introducing additional computational burden. Worse yet, our pioneer findings indicate that the task-aware prompt design is inherently limited by its absence of instance-aware information, leading to a subtle attention interplay with the input sequence. In contrast, simply incorporating instance-aware information as a part of the guidance can enhance the prompt-tuned model performance without additional fine-tuning. Moreover, we find an interesting phenomenon, namely "attention anchor", that incorporating instance-aware tokens at the earliest position of the sequence can successfully preserve strong attention to critical structural information and exhibit more active attention interaction with all input tokens. In light of our observation, we introduce Capsule Prompt-Tuning (CaPT), an efficient and effective solution that leverages off-the-shelf, informative instance semantics into prompt-based learning. Our approach innovatively integrates both instance-aware and task-aware information in a nearly parameter-free manner (i.e., one single capsule prompt).Empirical results demonstrate that our method can exhibit superior performance across various language tasks (e.g., 84.03\% average accuracy on T5-Large), serving as an "attention anchor," while enjoying high parameter efficiency (e.g., 0.003\% of model parameters on Llama3.2-1B).

## Defending Multimodal Backdoored Models by Repulsive Visual Prompt Tuning

- Authors: Zhifang Zhang, Shuo He, Haobo Wang, Bingquan Shen, Lei Feng
- Session: Poster
- Matched terms: prompt tuning
- Critique: Defense depends on few-shot clean data but does not explore how sensitive the approach is to label noise or domain shifts in that clean subset.

Multimodal contrastive learning models (e.g., CLIP) can learn high-quality representations from large-scale image-text datasets, while they exhibit significant vulnerabilities to backdoor attacks, raising serious safety concerns. In this paper, we reveal that CLIP's vulnerabilities primarily stem from its tendency to encode features beyond in-dataset predictive patterns, compromising its visual feature resistivity to input perturbations. This makes its encoded features highly susceptible to being reshaped by backdoor triggers. To address this challenge, we propose Repulsive Visual Prompt Tuning (RVPT), a novel defense approach that employs deep visual prompt tuning with a specially designed feature-repelling loss. Specifically, RVPT adversarially repels the encoded features from deeper layers while optimizing the standard cross-entropy loss, ensuring that only predictive features in downstream tasks are encoded, thereby enhancing CLIP’s visual feature resistivity against input perturbations and mitigating its susceptibility to backdoor attacks. Unlike existing multimodal backdoor defense methods that typically require the availability of poisoned data or involve fine-tuning the entire model, RVPT leverages few-shot downstream clean samples and only tunes a small number of parameters. Empirical results demonstrate that RVPT tunes only 0.27\% of the parameters in CLIP, yet it significantly outperforms state-of-the-art defense methods, reducing the attack success rate from 89.70\% to 2.76\% against the most advanced multimodal attacks on ImageNet and effectively generalizes its defensive capabilities across multiple datasets. Our code is available on https://anonymous.4open.science/r/rvpt-anonymous.

## GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining

- Authors: Chunyu Wei, Wenji Hu, Xingjia Hao, Xin Wang, Yifan Yang, Yunhai Wang, Yang Tian, Yueguo Chen
- Session: Poster
- Matched terms: soft prompt
- Critique: The tool-chaining framework could introduce heavy engineering overhead, yet the paper provides little analysis of latency or failure recovery when tools misbehave.

Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We introduce GraphChain, a novel framework enabling LLMs to analyze large graphs by orchestrating dynamic sequences of specialized tools, mimicking human exploratory processes. GraphChain incorporates two core technical contributions: (1) Progressive Graph Distillation, a reinforcement learning approach that learns to generate tool sequences balancing task relevance and intermediate state compression, thereby overcoming LLM context limitations. (2) Structure-aware Test-Time Adaptation (STTA), a mechanism using a lightweight, self-supervised adapter conditioned on graph spectral properties to efficiently adapt a frozen LLM policy to diverse graph structures via soft prompts without retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis.

## MixPrompt: Efficient Mixed Prompting for Multimodal Semantic Segmentation

- Authors: Zhiwei Hao, Zhongyu Xiao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Dan Zeng
- Session: Poster
- Matched terms: prompt tuning
- Critique: MixPrompt freezes the RGB backbone, so gains may diminish on domains where backbone features are misaligned; the paper lacks cross-domain evaluations.

Recent advances in multimodal semantic segmentation show that incorporating auxiliary inputs—such as depth or thermal images—can significantly improve performance over single-modality (RGB-only) approaches. However, most existing solutions rely on parallel backbone networks and complex fusion modules, greatly increasing model size and computational demands. Inspired by prompt tuning in large language models, we introduce \textbf{MixPrompt}: a prompting-based framework that integrates auxiliary modalities into a pretrained RGB segmentation model without modifying its architecture. MixPrompt uses a lightweight prompting module to extract and fuse information from auxiliary inputs into the main RGB backbone. This module is initialized using the early layers of a pretrained RGB feature extractor, ensuring a strong starting point. At each backbone layer, MixPrompt aligns RGB and auxiliary features in multiple low-rank subspaces, maximizing information use with minimal parameter overhead. An information mixing scheme enables cross-subspace interaction for further performance gains. During training, only the prompting module and segmentation head are updated, keeping the RGB backbone frozen for parameter efficiency. Experiments across NYU Depth V2, SUN-RGBD, MFNet, and DELIVER datasets show that MixPrompt achieves improvements of 4.3, 1.1, 0.4, and 1.1 mIoU, respectively, over two-branch baselines, while using nearly half the parameters. MixPrompt also outperforms recent prompting-based methods under similar compute budgets.

## Noise Matters: Optimizing Matching Noise for Diffusion Classifiers

- Authors: Yanghao Wang, Long Chen
- Session: Poster
- Matched terms: prompt tuning
- Critique: Noise optimization is dataset-specific, raising concerns about scalability to rapidly changing data or continual learning settings.

Although today's pretrained discriminative vision-language models (e.g., CLIP) have demonstrated strong perception abilities, such as zero-shot image classification, they also suffer from the bag-of-words problem and spurious bias. To mitigate these problems, some pioneering studies leverage powerful generative models (e.g., pretrained diffusion models) to realize generalizable image classification, dubbed Diffusion Classifier (DC). Specifically, by randomly sampling a Gaussian noise, DC utilizes the differences of denoising effects with different category conditions to classify categories. Unfortunately, an inherent and notorious weakness of existing DCs is noise instability: different random sampled noises lead to significant performance changes. To achieve stable classification performance, existing DCs always ensemble the results of hundreds of sampled noises, which significantly reduces the classification speed. To this end, we firstly explore the role of noise in DC, and conclude that: there are some ``good noises'' that can relieve the instability. Meanwhile, we argue that these good noises should meet two principles: 1) Frequency Matching: noise should destroy the specific frequency signals; 2) Spatial Matching: noise should destroy the specific spatial areas. Regarding both principles, we propose a novel Noise Optimization method to learn matching (i.e., good) noise for DCs: NoOp. For frequency matching, NoOp first optimizes a dataset-specific noise: Given a dataset and a timestep $t$, optimize one randomly initialized parameterized noise. For Spatial Matching, NoOp trains a Meta-Network that adopts an image as input and outputs image-specific noise offset. The sum of optimized noise and noise offset will be used in DC to replace random noise. Extensive ablations on various datasets demonstrated the effectiveness of NoOp. It is worth noting that our noise optimization is orthogonal to existing optimization methods (e.g., prompt tuning), our NoOP can even benefit from these methods to further boost performance.

## PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs

- Authors: Jaewon Chu, Seunghun Lee, Hyunwoo J. Kim
- Session: Poster
- Matched terms: soft prompt
- Critique: PRESTO assumes access to plentiful black-box queries for preimage exploration, but cost and rate limits for commercial APIs are not quantified.

Large language models (LLMs) have achieved remarkable success across diverse domains, due to their strong instruction-following capabilities. This raised interest in optimizing instructions for black-box LLMs, whose internal parameters are inaccessible but popular for their strong performance and ease of use. Recent approaches leverage white-box LLMs to assist instruction optimization for black-box LLMs by generating instructions from soft prompts. However, white-box LLMs often map different soft prompts to the same instruction, leading to redundant queries to the black-box model. While previous studies regarded this many-to-one mapping as a redundancy to be avoided, we reinterpret it as useful prior knowledge that can enhance the optimization performance. To this end, we introduce PREimage-informed inSTruction Optimization (PRESTO), a novel framework that leverages the preimage structure of soft prompts to improve query efficiency. PRESTO consists of three key components: (1) score sharing, which shares the evaluation score with all soft prompts in a preimage; (2) preimage-based initialization, which select initial data points that maximize search space coverage using preimage information; and (3) score consistency regularization, which enforces prediction consistency within each preimage. By leveraging preimages, PRESTO observes 14 times more scored data under the same query budget, resulting in more efficient optimization. Experimental results on 33 instruction optimization tasks demonstrate the superior performance of PRESTO.

## PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning

- Authors: Zongqian Li, Yixuan Su, Nigel Collier
- Session: Poster
- Matched terms: prompt tuning
- Critique: The integration of MoE and decomposition increases architectural complexity, yet deployment-time memory and inference impacts are not reported.

Parameter-efficient fine-tuning (PEFT) methods have shown promise in adapting large language models, yet existing approaches exhibit counter-intuitive phenomena: integrating either matrix decomposition or mixture-of-experts (MoE) individually decreases performance across tasks, though decomposition improves results on specific domains despite reducing parameters, while MoE increases parameter count without corresponding decrease in training efficiency. Motivated by these observations and the modular nature of PT, we propose PT-MoE, a novel framework that integrates matrix decomposition with MoE routing for efficient PT. Evaluation results across 17 datasets demonstrate that PT-MoE achieves state-of-the-art performance in both question answering (QA) and mathematical problem solving tasks, improving F1 score by 1.49 points over PT and 2.13 points over LoRA in QA tasks, while improving mathematical accuracy by 10.75 points over PT and 0.44 points over LoRA, all while using 25% fewer parameters than LoRA. Our analysis reveals that while PT methods generally excel in QA tasks and LoRA-based methods in math datasets, the integration of matrix decomposition and MoE in PT-MoE yields complementary benefits: decomposition enables efficient parameter sharing across experts while MoE provides dynamic adaptation, collectively enabling PT-MoE to demonstrate cross-task consistency and generalization abilities. These findings, along with ablation studies on routing mechanisms and architectural components, provide insights for future PEFT methods.

## Prompt Tuning Decision Transformers with Structured and Scalable Bandits

- Authors: Finn Rietz, Oleg Smirnov, Sara Karimi, Lele Cao
- Session: Poster
- Matched terms: prompt tuning
- Critique: Bandit-driven prompt selection assumes reliable off-policy reward signals from demonstrations, which may overestimate performance in sparse-reward environments.

Prompt tuning has emerged as a key technique for adapting large pre-trained Decision Transformers (DTs) in offline Reinforcement Learning (RL), particularly in multi-task and few-shot settings. The Prompting Decision Transformer (PDT) enables task generalization via trajectory prompts sampled uniformly from expert demonstrations -- without accounting for prompt informativeness. In this work, we propose a bandit-based prompt-tuning method that learns to construct optimal trajectory prompts from demonstration data at inference time. We devise a structured bandit architecture operating in the trajectory prompt space, achieving linear rather than combinatorial scaling with prompt size.  Additionally, we show that the pre-trained PDT itself can serve as a powerful feature extractor for the bandit, enabling efficient reward modeling across various environments. We theoretically establish regret bounds and demonstrate empirically that our method consistently enhances performance across a wide range of tasks, high-dimensional environments, and out-of-distribution scenarios, outperforming existing baselines in prompt tuning.

## Prompt Tuning Transformers for Data Memorization

- Authors: Haiyu Wang, Yuanyuan Lin
- Session: Poster
- Matched terms: prompt tuning
- Critique: Memorization analysis focuses on theoretical capacity, but practical risks like overfitting or privacy leakage in prompt-tuned transformers are not experimentally studied.

One research direction for understanding Transformers' representational capacity involves quantifying their data memorization ability. However, the memorization capacity of Transformers with prompts is not yet well understood. In this work, we extend the existing results on prompt tuning Transformers to simulate ReLU neural networks with the Autoregressive algorithm in \citep{nakada2025theoretical} to a more general case where the width of the ReLU neural network can be a small constant. Based on this, we prove that a constant-size Transformer with prompts whose length is $\tilde{O}(\sqrt{nN})$ can memorize any $N$ input sequence of length $n$. Our results discard the reliance on large feed-forward layers. Besides, we also theoretically prove that a trade-off exists between the prompt length and computational efficiency. Our findings, supported by experiments, demonstrate that a single-layer randomly initialized Transformer with prompts can possess competitive data memorization ability compared with models trained from scratch. In addition, we validate that the prompt length is potentially reduced if the low-rank structure exists.

## Reconstructing Robust Vision-Language Models from Natural Latent Spaces

- Authors: Zhangyun Wang, Ni Ding, Aniket Mahanti
- Session: Poster
- Matched terms: prompt tuning
- Critique: CoAPT relies on improved TV filtering, yet robustness gains may stem from preprocessing rather than prompt tuning; ablations isolating each component are missing.

Pre-trained Vision-Language Models (VLMs) exhibit significant vulnerability to imperceptible adversarial perturbations. Current advanced defense strategies typically employ adversarial prompt tuning to improve the adversarial robustness of VLMs, which struggle to simultaneously maintain generalization across both natural and adversarial examples under different benchmarks and downstream tasks. We propose a Collaborative Adversarial Prompt Tuning (CoAPT) approach from pre-trained models to target robust models. Specifically, we adopt an improved fast total variation (TV) technique to suppress and eliminate high-frequency details from images while preserving edge structures, thereby disrupting the adversarial perturbation space. Subsequently, guided by the high-level image and text representations in the latent space of the pre-trained VLMs, the corrupted natural features are restored while inheriting the superior generalization capability. Compared to existing state-of-the-art methods, CoAPT achieves a superior trade-off between robustness and generalization. Across four benchmark tasks, CoAPT improves robustness by 28.37\% (few-shot-16), 31.67\% (base-to-novel), 20.01\% (zero-shot), and 17.37\% (out-of-distribution), while maintaining an average increase of 9.83\% in natural accuracy.

## SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes

- Authors: Yifan Yang, Zhen Zhang, Rupak Swaminathan, Jing Liu, Nathan Susanj, Zheng Zhang
- Session: Poster
- Matched terms: prompt tuning
- Critique: Forward-only optimization reduces hardware needs but may suffer from gradient-free noise; comparative convergence curves versus BP-free baselines are not provided.

Fine-tuning vision language models (VLMs) has achieved remarkable performance across various downstream tasks; yet, it requires access to model gradients through  backpropagation (BP), making them unsuitable for memory-constrained, inference-only edge devices. To address this limitation, previous work has explored various BP-free fine-tuning methods. However, these approaches often rely on high-variance evolutionary strategies (ES) or zeroth-order (ZO) optimization, and often fail to achieve satisfactory performance. In this paper, we propose a hybrid Sharpness-aware Zeroth-order optimization (SharpZO) approach, specifically designed to enhance the performance of ZO VLM fine-tuning via a sharpness-aware warm-up training. SharpZO features a two-stage optimization process: a sharpness-aware ES stage that globally explores and smooths the loss landscape to construct a strong initialization, followed by a fine-grained local search via sparse ZO optimization. The entire optimization relies solely on forward passes. Detailed theoretical analysis and extensive experiments on CLIP models demonstrate that SharpZO significantly improves accuracy and convergence speed, achieving up to 7\% average gain over state-of-the-art forward-only methods.

## Test-Time Adaptive Object Detection with Foundation Model

- Authors: Yingjie Gao, Yanan Zhang, Zhi Cai, Di Huang
- Session: Poster
- Matched terms: prompt tuning
- Critique: The method stores high-quality pseudo-labels, but lacks analysis on memory growth or forgetting when encountering long test streams.

In recent years, test-time adaptive object detection has attracted increasing attention due to its unique advantages in online domain adaptation, which aligns more closely with real-world application scenarios. However, existing approaches heavily rely on source-derived statistical characteristics while making the strong assumption that the source and target domains share an identical category space. In this paper, we propose the first foundation model-powered test-time adaptive object detection method that eliminates the need for source data entirely and overcomes traditional closed-set limitations. Specifically, we design a Multi-modal Prompt-based Mean-Teacher framework for vision-language detector-driven test-time adaptation, which incorporates text and visual prompt tuning to adapt both language and visual representation spaces on the test data in a parameter-efficient manner. Correspondingly, we propose a Test-time Warm-start strategy tailored for the visual prompts to effectively preserve the representation capability of the vision branch. Furthermore, to guarantee high-quality pseudo-labels in every test batch, we maintain an Instance Dynamic Memory (IDM) module that stores high-quality pseudo-labels from previous test samples, and propose two novel strategies-Memory Reinforcement and Memory Hallucination-to leverage IDM's high-quality instances for enhancing original predictions and hallucinating images without available pseudo-labels, respectively. Extensive experiments on cross-corruption and cross-dataset benchmarks demonstrate that our method consistently outperforms previous state-of-the-art methods, and can adapt to arbitrary cross-domain and cross-category target data. The code and models will be made publicly available.

## Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models

- Authors: Konstantinos Dafnis, Dimitris Metaxas
- Session: Poster
- Matched terms: prompt tuning
- Critique: Spectrum-aware steering is evaluated on standard protocols; robustness to severe distribution shifts or adversarial text prompts is not discussed.

Vision–language models (VLMs) excel at zero-shot inference but often degrade under test-time domain shifts. For this reason, episodic test-time adaptation strategies have recently emerged as powerful techniques for adapting VLMs to a single unlabeled image. However, existing adaptation strategies, such as test-time prompt tuning, typically require backpropagating through large encoder weights or altering core model components. In this work, we introduce \textbf{S}pectrum-Aware \textbf{T}est-Time \textbf{S}teering (\textbf{STS}), a \textit{lightweight adaptation framework} that extracts a spectral subspace from the textual embeddings to define principal semantic directions, and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views. STS operates entirely at inference in the latent space, without backpropagation through or modification of the frozen encoders. Building on standard evaluation protocols, our comprehensive experiments demonstrate that STS largely surpasses or compares favorably against state-of-the-art test-time adaptation methods, while introducing only a handful of additional parameters and achieving inference speeds up to 8× faster with a 12× smaller memory footprint than conventional test-time prompt tuning.

## Understanding Prompt Tuning and In-Context Learning via Meta-Learning

- Authors: Tim Genewein, Kevin Li, Jordi Grau-Moya, Anian Ruoss, Laurent Orseau, Marcus Hutter
- Session: Poster
- Matched terms: prompt tuning, soft prefix
- Critique: While offering theory, the experiments use educational toy settings, leaving open how insights translate to modern large-scale models.

Prompting is one of the main ways to adapt a pretrained model to target tasks. Besides manually constructing prompts, many prompt optimization methods have been proposed in the literature. Method development is mainly empirically driven, with less emphasis on a conceptual understanding of prompting. In this paper we discuss how optimal prompting can be understood through a Bayesian view, which also implies some fundamental limitations of prompting that can only be overcome by tuning weights. The paper explains in detail how meta-trained neural networks behave as Bayesian predictors over the pretraining distribution, whose hallmark feature is rapid in-context adaptation. Optimal prompting can be studied formally as conditioning these Bayesian predictors, yielding criteria for target tasks where optimal prompting is and is not possible. We support the theory with educational experiments on LSTMs and Transformers, where we compare different versions of prefix-tuning and different weight-tuning methods. We also confirm that soft prefixes, which are sequences of real-valued vectors outside the token alphabet, can lead to very effective prompts for trained and even untrained networks by manipulating activations in ways that are not achievable by hard tokens. This adds an important mechanistic aspect beyond the conceptual Bayesian theory.

## VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion

- Authors: Jaekyun Park, Hye Won Chung
- Session: Poster
- Matched terms: prompt tuning
- Critique: VIPAMIN is validated on self-supervised backbones, but the paper does not compare against strong supervised or hybrid pretraining baselines.

In the era of large-scale foundation models, fully fine-tuning pretrained networks for each downstream task is often prohibitively resource-intensive. Prompt tuning offers a lightweight alternative by introducing tunable prompts while keeping the backbone frozen. However, existing visual prompt tuning methods often fail to specialize the prompts or enrich the representation space--especially when applied to self-supervised backbones. We show that these limitations become especially pronounced in challenging tasks and data-scarce settings, where effective adaptation is most critical. In this work, we introduce VIPAMIN, a visual prompt initialization strategy that enhances adaptation of self-supervised models by (1) aligning prompts with semantically informative regions in the embedding space, and (2) injecting novel representational directions beyond the pretrained subspace. Despite its simplicity--requiring only a single forward pass and lightweight operations--VIPAMIN consistently improves performance across diverse tasks and dataset sizes, setting a new state of the art in visual prompt tuning.

## VaMP: Variational Multi-Modal Prompt Learning

- Authors: Silin Cheng, Kai Han
- Session: Poster
- Matched terms: prompt tuning
- Critique: Variational prompts introduce sampling noise; the paper lacks calibration or uncertainty quality metrics to justify the added stochasticity.

Vision-language models (VLMs), such as CLIP, have shown strong generalization under zero-shot settings, yet adapting them to downstream tasks with limited supervision remains a significant challenge. Existing multi-modal prompt learning methods typically rely on fixed, shared prompts and deterministic parameters, which limits their ability to capture instance-level variation or model uncertainty across diverse tasks and domains. We propose VaMP, a variational framework for prompt adaptation that enables sample-specific, uncertainty-aware tuning in multi-modal representation learning. VaMP generates instance-conditioned prompts by sampling from a learned posterior distribution, allowing the model to personalize its behavior based on input content. To guide this process with both local and global semantics, we introduce a class-aware prior constructed from the instance representation and class prototype. We formulate prompt tuning as variational inference over latent prompt representations and train the entire framework end-to-end via reparameterized sampling. Experiments on few-shot and domain generalization benchmarks show that VaMP achieves state-of-the-art performance among prompt tuning methods, highlighting the benefits of modeling both uncertainty and task structure in multi-modal prompt adaptation.
