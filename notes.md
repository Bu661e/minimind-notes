# LLM 学习笔记

> 由 `llm-learning` skill 在对话中静默记录。

## 1. 归一化

### Q1. RMSNorm 和 LayerNorm 有什么区别？

**A**：LayerNorm 减均值再除以标准差；RMSNorm 只按均方根缩放、不减均值，少一次求均值，也无 bias，效果接近而更快。Llama 系列与 MiniMind 都用它。

**公式**

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \odot \gamma
$$

**PyTorch 调用**

```python
import torch.nn as nn

norm = nn.RMSNorm(512)          # PyTorch ≥ 2.4
y = norm(x)
```

**从 tensor 手写**

```python
def rms_norm(x, gamma, eps=1e-6):
    # x: (..., n)
    x = x * torch.rsqrt(
            x.pow(2) # a.pow(b) = a ** b
            .mean(dim=-1, keepdim=True)  # mean = 按照维度求均值，keepdim=True 保持维度不变
            + eps # 1e-6 防止除零
        )
    return x * gamma            # 逐元素缩放
```
