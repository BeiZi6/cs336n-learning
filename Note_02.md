# CS336 Lecture 2: 训练基础 — 从张量到训练循环

> 上节回顾：课程概览、Tokenization
>
> 本节目标：自底向上介绍训练模型所需的所有**基本组件**，从张量 → 模型 → 优化器 → 训练循环，并始终关注**资源效率**。

## 核心资源

训练过程中需要关注两类资源：

- Memory（内存，单位 GB）
- Compute（计算量，单位 FLOPs）

## 参考资料

本节不讲 Transformer 本身，推荐阅读：

- [Assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)
- [Mathematical description](https://johnthickstun.com/docs/transformers.pdf)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

## 学习要点

- Mechanics：直接用 PyTorch 实现
- Mindset：养成资源核算的习惯
- Intuitions：建立大致直觉（不涉及大模型）

---

## 一、Motivating Questions（热身估算）

### Q1：70B 模型在 1024 张 H100 上训练 15T tokens 需要多久？

```python
total_flops = 6 * 70e9 * 15e12           # 总计算量
h100_flop_per_sec = 1979e12 / 2           # H100 峰值（非稀疏）
mfu = 0.5                                 # 模型利用率
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
days = total_flops / flops_per_day        # ≈ 约数十天
```

### Q2：8 张 H100 最大能训多大的模型（AdamW，朴素方式）？

```python
h100_bytes = 80e9                          # 每张 80GB
bytes_per_parameter = 4 + 4 + (4 + 4)     # 参数 + 梯度 + 优化器状态 = 16 bytes
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

- Caveat 1：可以用 bf16 存参数和梯度（2+2），再保留一份 float32 副本（4），总共不省内存但更快
- Caveat 2：未计入 activations（取决于 batch size 和 sequence length）

---

## 二、Memory Accounting（内存核算）

### 2.1 Tensor 基础

张量是存储一切的基本单元：参数、梯度、优化器状态、数据、激活值。

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])   # 从数据创建
x = torch.zeros(4, 8)                        # 全零
x = torch.ones(4, 8)                         # 全一
x = torch.randn(4, 8)                        # 标准正态采样
x = torch.empty(4, 8)                        # 分配但不初始化
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断正态初始化
```

### 2.2 数据类型与内存

内存 = 元素数量 × 每个元素的字节数

#### float32（单精度，默认）

- 4 bytes/element
- GPT-3 的一个 FFN 矩阵 (12288×4, 12288)：**2.3 GB**

#### float16（半精度）

- 2 bytes/element，内存减半
- 问题：动态范围差，小数值会下溢
  ```python
  x = torch.tensor([1e-8], dtype=torch.float16)  # → 0（下溢！）
  ```

#### bfloat16（Brain Floating Point）

- 2 bytes/element，与 float16 相同内存
- 与 float32 相同的动态范围，精度稍差但对深度学习影响不大
  ```python
  x = torch.tensor([1e-8], dtype=torch.bfloat16)  # → 非零（无下溢）
  ```

#### fp8

- 2022 年标准化，专为 ML 设计
- H100 支持两种变体：E4M3（范围 [-448, 448]）和 E5M2（[-57344, 57344]）

#### 训练启示

- float32 可靠但内存大
- fp8/float16/bfloat16 有不稳定风险
- 解决方案：**混合精度训练**（见后文）

---

## 三、Compute Accounting（计算量核算）

### 3.1 GPU 上的张量

```python
x = torch.zeros(32, 32)                    # 默认在 CPU
y = x.to("cuda:0")                         # 移到 GPU
z = torch.zeros(32, 32, device="cuda:0")   # 直接在 GPU 创建
```

### 3.2 Tensor 操作

#### Storage（存储结构）

PyTorch 张量 = 指向内存的指针 + 描述如何访问元素的元数据（stride）。

```python
x.stride(0)  # 行步长：跳过多少元素到下一行
x.stride(1)  # 列步长：跳过多少元素到下一列
# 定位元素：index = r * stride(0) + c * stride(1)
```

#### Slicing（切片 — 视图操作）

很多操作只是提供张量的不同**视图（view）**，不复制数据：

```python
y = x[0]              # 取行 → 共享存储
y = x[:, 1]           # 取列 → 共享存储
y = x.view(3, 2)      # 重塑 → 共享存储
y = x.transpose(1, 0) # 转置 → 共享存储（但可能非连续）
```

- 修改 x 会同时影响 y（共享存储）
- 非连续张量需要 `.contiguous()` 后才能 `.view()`
- **视图免费，复制消耗内存和计算**

#### Elementwise（逐元素操作）

```python
x.pow(2)    # 平方
x.sqrt()    # 开方
x.rsqrt()   # 1/sqrt(x)
x + x       # 加法
x * 2       # 标量乘
x.triu()    # 上三角（用于 causal attention mask）
```

#### Matmul（矩阵乘法 — 深度学习的核心）

```python
x = torch.ones(4, 8, 16, 32)  # batch, seq, tokens, hidden
w = torch.ones(32, 2)
y = x @ w                      # → (4, 8, 16, 2)
# 对前两个维度逐一迭代，每个做 16×32 @ 32×2
```

### 3.3 Einops

传统 PyTorch 维度操作容易出错（`transpose(-2, -1)` 是什么？），Einops 用命名维度解决。

#### jaxtyping（维度文档化）

```python
x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
# 仅文档作用，无运行时检查
```

#### einsum（广义矩阵乘法）

```python
# 旧写法
z = x @ y.transpose(-2, -1)
# Einops 写法
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# 输出中未出现的维度会被求和
```

#### reduce（归约）

```python
# 旧写法
y = x.sum(dim=-1)
# Einops 写法
y = reduce(x, "... hidden -> ...", "sum")
```

#### rearrange（重排）

```python
# 拆分维度：total_hidden → heads × hidden1
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
# 合并维度
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

### 3.4 FLOPs 核算

FLOP = 一次基本浮点运算（加法或乘法）

- FLOPs：浮点运算次数（计算量）
- FLOP/s：每秒浮点运算次数（硬件速度）

#### 直觉

- 训练 GPT-3（2020）：3.14e23 FLOPs
- 训练 GPT-4（2023，推测）：2e25 FLOPs
- A100 峰值：312 TFLOP/s
- H100 峰值：1979 TFLOP/s（稀疏），非稀疏约一半

#### 矩阵乘法 FLOPs

```
y = x @ w    # x: (B, D), w: (D, K)
FLOPs = 2 * B * D * K   # 每个 (i,j,k) 一次乘法 + 一次加法
```

- 逐元素操作：O(m × n)
- 矩阵乘法在大矩阵时远超其他操作

#### 推广到 Transformer

- Forward FLOPs ≈ 2 × (tokens 数) × (参数量)

### 3.5 MFU（Model FLOPs Utilization）

$$\text{MFU} = \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}$$

- MFU ≥ 0.5 算不错
- bfloat16 比 float32 实际 FLOP/s 更高
- FLOP/s 取决于硬件（H100 >> A100）和数据类型（bfloat16 >> float32）

### 3.6 Gradients（梯度）

#### 基础

```python
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
loss.backward()   # 反向传播计算梯度
# w.grad 现在有值了
```

#### 梯度 FLOPs

对于模型 `x → w1 → h1 → w2 → h2 → loss`：

- Forward：2 × (数据点数) × (参数量) FLOPs
- Backward：4 × (数据点数) × (参数量) FLOPs
- **总计：6 × (数据点数) × (参数量) FLOPs**

反向传播中每个参数矩阵需要两次矩阵乘法：
1. 计算参数梯度：`w2.grad[j,k] = sum_i h1[i,j] * h2.grad[i,k]`
2. 计算输入梯度：`h1.grad[i,j] = sum_k w2[j,k] * h2.grad[i,k]`

---

## 四、Models（模型）

### 4.1 参数初始化

```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
```

- 问题：`output = x @ w` 的每个元素会随 `input_dim` 增大而增大 → 梯度爆炸
- 解决：缩放 `1/sqrt(input_dim)`（Xavier 初始化）
  ```python
  w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
  ```
- 更安全：截断正态分布 [-3, 3]，避免极端值
  ```python
  w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim),
                                          std=1/np.sqrt(input_dim), a=-3, b=3))
  ```

### 4.2 自定义模型示例

```python
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x):
        return x @ self.weight

class Cruncher(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Linear(dim, dim) for _ in range(num_layers)])
        self.final = Linear(dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final(x).squeeze(-1)
```

- 参数量：`num_layers × D² + D`
- 记得 `model.to(device)` 移到 GPU

---

## 五、Training Loop & Best Practices

### 5.1 随机性管理

随机性出现在：参数初始化、dropout、数据顺序等。为了可复现性，同时设置三处种子：

```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

### 5.2 数据加载

语言模型的数据 = 整数序列（tokenizer 输出）。

- 用 `np.memmap` 懒加载（LLaMA 数据 2.8TB，不能全部载入内存）
- Data loader 从数据中随机采样 batch_size 个长度为 sequence_length 的片段

#### Pinned Memory

```python
x = x.pin_memory()                    # 固定到物理内存
x = x.to(device, non_blocking=True)   # 异步传输到 GPU
```

好处：可以并行执行「取下一批数据到 CPU」和「在 GPU 上处理当前数据」。

### 5.3 优化器

优化器族谱：

| 优化器 | 特点 |
|--------|------|
| **SGD** | 基础：`w -= lr * grad` |
| **Momentum** | SGD + 梯度指数平均 |
| **AdaGrad** | SGD + 按 grad² 累积缩放 |
| **RMSProp** | AdaGrad + grad² 指数平均 |
| **Adam** | RMSProp + Momentum |

#### 内存核算（以 AdaGrad 为例）

| 组成 | 大小 |
|------|------|
| 参数 | num_parameters |
| 激活值 | B × D × num_layers |
| 梯度 | num_parameters |
| 优化器状态 | num_parameters |

总内存（float32）= 4 × (参数 + 激活 + 梯度 + 优化器状态)

#### 单步计算量

$$\text{FLOPs} = 6 \times B \times \text{num\_parameters}$$

### 5.4 训练循环

```python
for t in range(num_train_steps):
    x, y = get_batch(B=B)              # 1. 获取数据
    pred_y = model(x)                  # 2. 前向传播
    loss = F.mse_loss(pred_y, y)       # 3. 计算损失
    loss.backward()                    # 4. 反向传播
    optimizer.step()                   # 5. 更新参数
    optimizer.zero_grad(set_to_none=True)  # 6. 清空梯度
```

### 5.5 Checkpointing（检查点）

训练时间长，必然会崩溃，定期保存进度：

```python
# 保存
checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
torch.save(checkpoint, "model_checkpoint.pt")

# 加载
loaded_checkpoint = torch.load("model_checkpoint.pt")
```

### 5.6 Mixed Precision Training（混合精度训练）

精度权衡：
- 高精度（float32）：准确稳定，但内存大、计算慢
- 低精度（fp8/float16/bfloat16）：内存小、计算快，但可能不稳定

解决方案 — 混合使用：
- Forward pass（激活值）：用 bfloat16 / fp8
- 其余（参数、梯度）：用 float32

PyTorch 提供 AMP（Automatic Mixed Precision）库，NVIDIA Transformer Engine 支持 FP8。
