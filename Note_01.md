# CS336 Lecture 1: 课程概览与基础概念

## Kernel（内核）

- Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens

## Parallelism（并行）

在**多GPU上分布训练计算**的技术，由于模型的参数很大，单个GPU无法完成训练，所以需要**多个GPU协同工作**。

- GPU间通信比计算慢得多
- 四种主要并行策略

| 策略                       | 切分对象       | 适用场景            |
| ------------------------ | ---------- | --------------- |
| **Data Parallelism**     | 数据批次       | 模型能放进单GPU，想加速训练 |
| **Tensor Parallelism**   | 层内参数（如矩阵列） | 单层太大，单GPU放不下    |
| **Pipeline Parallelism** | 不同层        | 模型很深，不同GPU负责不同层 |
| **Sequence Parallelism** | 序列长度维度     | 处理超长序列          |

## Inference（推理）

- Goal：生成 tokens
- Two phases：prefill and decode（预填充和解码）
  - prefill：预填充，输入是给定的 prompt 的所有 token，可以并行处理所有 token，**计算密集型**
  - decode：解码，逐个生成新 token，一次只生成一个，**内存密集型**
  - 示例：
    ```
    Prompt: "The capital of France is"
           ↓ [Prefill: 一次性处理所有 prompt tokens]
    Generate: "Paris"
           ↓ [Decode: 逐个生成，每次用前一步的输出作为下一步输入]
    Generate: "." (结束)
    ```
- Methods to speed up decoding：
  - 模型压缩：剪枝（去掉不重要的参数），量化（用更低位数表示参数），蒸馏（训练小模型模仿大模型）
  - 投机解码（Speculative Decoding）：用"草稿"小模型快速生成多个候选 token，同时大模型并行验证，保留好的 token
  - 系统优化：KV Cache（缓存注意力计算的 key/value）、Batching（批处理，同时处理多个请求）

## Scaling Laws（扩展定律）

- Goal：给定固定的 FLOPs 预算 C，应该使用**更大的模型**（更多参数 N），还是**训练更多的数据**（更多 token D）
  - FLOPs 预算 C：Floating Point Operations（浮点运算次数）— 训练模型需要的计算量
- Chinchilla 最优定律
  $$D^* = 20 \times N^*$$
  **含义**：模型参数量和训练数据量应保持固定比例。
  **例子**：
  - 1.4B 参数模型 → 应训练 28B tokens
  - 70B 参数模型（Chinchilla）→ 训练 1.4T tokens
- Meaning：
  1. **超参数调优**：在小模型上做实验，预测大模型的最佳设置
  2. **资源分配**：确定最优的模型大小和数据量组合
  3. **成本优化**：避免浪费计算在过大模型或过多数据上
  4. **注意**：Chinchilla 只考虑训练成本，没考虑**推理成本**（实际部署中推理往往更贵）

## Data（数据）

- Goal：我们希望模型具备什么能力？不同的能力需求决定了需要什么样的训练数据。
- Evaluation：

| 类型                            | 方法                                    |
| ----------------------------- | ------------------------------------- |
| **困惑度 (Perplexity)**          | 语言模型的教科书式评估                           |
| **标准化测试**                     | MMLU（多任务理解）、HellaSwag（常识推理）、GSM8K（数学） |
| **指令遵循**                      | AlpacaEval、IFEval、WildBench           |
| **Scaling test-time compute** | Chain-of-thought、ensembling           |
| **LM-as-a-judge**             | 评估生成任务                                |
| **完整系统**                      | RAG、agents                            |

- Data Curation（数据收集与整理）
- Data Processing（数据处理流程）
  - Transformation：将 HTML/PDF 转换成文本
  - Filtering：保留高质量数据、通过分类器删除有害内容
  - Deduplication：使用 Bloom filters（布隆过滤器）或 MinHash（最小哈希），目的是节省计算、避免记忆

## Alignment（对齐）

**让预训练好的"基础模型"真正变得有用**的过程。

- 基础模型 vs 对齐后模型

| 阶段       | 特点                    | 例子                                            |
| -------- | --------------------- | --------------------------------------------- |
| **基础模型** | 只会"续写"，raw potential   | 问"法国首都是？" → 输出"巴黎是一个美丽的城市..."（继续编故事） |
| **对齐后**  | 能回答问题、遵循指令            | 问"法国首都是？" → 输出"巴黎"                     |

- Goal：
  - Instruction Following
  - 调整风格（format, length, tone, etc.）
  - Incorporate safety（拒答有害问题）
- Two Phases：
  - Phase 1 — Supervised Finetuning（SFT，监督微调）
    - 用（prompt，response）对的数据
    - 数据通常需要人工标注
    - 直觉：基础模型已有技能，只需要少量标注示例"激活"
  - Phase 2 — Learn From Feedback（从反馈学习）
    - 不依赖昂贵的人工标注，通过比较生成结果来优化
    - Preference Data（偏好数据）
    - Verifiers（验证器）
      - Formal verifiers（形式验证器）：代码能否编译运行？数学题答案是否正确？
      - Learned verifiers（学习验证器）：训练一个 LM 作为裁判（LM-as-a-judge）
- Algorithms（对齐算法）
  - PPO（Proximal Policy Optimization，近端策略优化）：传统 RL，复杂，需要训练 value function
  - DPO（Direct Preference Optimization，直接偏好优化）：直接使用偏好数据，更简单
  - GRPO（Group Relative Preference Optimization，群体相对偏好优化）：DeepSeek-R1 所用，去掉 value function
- **总结**：Learning from feedback = 不依赖昂贵的人工标注，通过比较模型生成的答案哪个更好（偏好数据），用 DPO/GRPO 算法进一步优化模型。

## Tokenization（分词）

- Tokenizer: strings <-> tokens (indices)
- Tokenizer 是语言模型的**入口和出口**，负责：
  - **Encode（编码）**：`str → list[int]`（文本 → token 索引）
  - **Decode（解码）**：`list[int] → str`（token 索引 → 文本）
- 常见 Tokenizer 类型：
  - Character Tokenizer（字符级）
    - **词汇表**：~150K Unicode 字符
    - **问题**：词汇表太大，稀有字符浪费空间
  - Byte Tokenizer（字节级）
    - **词汇表**：256（0-255）
    - **问题**：压缩率 = 1，序列太长（中文1字=3字节）
  - Word Tokenizer（词级）
    - **词汇表**：训练数据中所有不同的词
    - **问题**：词汇表无限增长，UNK（未知词）问题
  - BPE Tokenizer（字节对编码）✓
    - Basic Idea：在原始文本上训练，自动确定词汇表
    - Intuition：**常见的字符序列用一个 token 表示，罕见的字符序列用多个 token 表示。**
    - Example：
      - 训练数据："the cat in the hat"
      - "the" 出现 2 次 → 合并成新 token 256
      - " c"（空格+c）出现 1 次 → 保持两个 tokens
      - 新文本 "the quick brown fox"：`"the"` → `[256]`，`"quick"` → `[113, 117, 105, 99, 107]`（拆成 bytes）

| 序列频率      | 表示方式             | 例子                                           |
| --------- | ---------------- | -------------------------------------------- |
| **常见**    | **单个 token**     | `"the"` → `[256]`                            |
| **中等**    | **2-3 个 tokens** | `"theater"` → `[256, 72, 81]`（the + at + er） |
| **罕见/新词** | **拆成 bytes/字符**  | `"量子"` → `[201, 173, 202, 135]`              |

- BPE 训练过程（Train the Tokenizer）
  - 开始：每个字节独立
  - 第1轮：发现 "t"+"h" 总在一起 → 合并成 "th"
  - 第2轮：发现 "i"+"n"+"g" 常见 → 先合并 "in"，再 "ing"
  - 最终：常用词/词根是单 token，生僻词是多 token 的组合
- BPE 使用流程（Using the Tokenizer）
  - 新文本 "the quick brown fox" → [encode] → tokens [256, 32, 101, ...] → 输入模型训练/推理 → [decode] → 还原成 "the quick brown fox"
  - **可逆性检查**：`assert string == reconstructed_string`，Tokenizer 必须保证**无损转换**，否则信息会丢失。
  - Pre-tokenization（预分词）
    - 原始文本 "Hello, world! I'll go." → [预分词 ← 用正则表达式切成块] → ["Hello", ", ", "world", "! ", "I'll", " ", "go", "."] → [在每个块内做 BPE]
    - **预分词是用规则先做"粗切分"，避免 BPE 跨词乱合并，让子词边界更合理。**
