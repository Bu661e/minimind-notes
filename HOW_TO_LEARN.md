# 怎么学 MiniMind

> 一份可执行的学习指引。所有命令都在本机实测过。

---

## 0. 先搞清楚目录

**本指引只涉及 `minimind-notes` 这一个仓库。** 从头到尾在仓库内操作，不碰外面的 `minimind/` 训练仓库。

```
minimind-notes/    # ← 只学这个
├── modules/01-foundation/   # 4 个基础模块：归一化、位置编码、注意力、前馈
├── modules/common/          # 实验公用工具
├── reference/               # 源码副本（教学文档按行号引用它们）
└── notes.md                 # 你的学习笔记（skill 自动记录）
```

> 唯一在仓库外的依赖是 Python 环境 `.venv/`（位于工作区根），下一节说明。

---

## 1. 环境（已完成，无需重做）

环境**已经装好**，位于 `/home/dht/dht_workspace/minimind/.venv`：

- Python 3.10.12（`/usr/bin/python3.10`，非 conda）
- torch 2.14.1+cu130，CUDA 可用（RTX 4090）
- 实验所需的 matplotlib / seaborn / numpy 等齐全

**每次开始学习，第一件事是激活它**：

```bash
source /home/dht/dht_workspace/minimind/.venv/bin/activate
```

> ⚠️ **必须激活**。模块里的 `run_all.sh` 调用的是裸 `python`，不激活会指向系统 python（没装 torch），直接 `ModuleNotFoundError: No module named 'torch'`。

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出 2.14.1+cu130 True
```

环境已一次性装好，重建不在本指引范围内（需要跨仓库操作，一般用不到）。

---

## 2. 数据（已下载，无需重做）

实验数据已缓存在 `minimind-notes/modules/common/data/`（共 63 MB）：TinyShakespeare + TinyStories 10MB/50MB 两档。

**不要重复运行下载脚本** —— 它会走 `datasets` 流式下载路径，触发一个解释器退出时的崩溃（`Fatal Python error: PyGILState_Release`，退出码 134）。数据已在，直接跳过去。

---

## 3. 学习顺序

严格按这个顺序走，每个模块都是「跑实验 → 读理论 → 自测」：

| 顺序 | 模块 | 实验数 | 核心问题 |
|---|---|---|---|
| 1️⃣ | `01-normalization` | 4 | 深层网络为什么难训练？为什么用 RMSNorm？Pre-LN vs Post-LN？ |
| 2️⃣ | `02-position-encoding` | 4 | Attention 排列不变，怎么知道词序？RoPE 为什么能外推长度？ |
| 3️⃣ | `03-attention` | 3 | Q/K/V 是什么？为什么要多头？GQA 省在哪？ |
| 4️⃣ | `04-feedforward` | 1 | FFN 存什么知识？为什么要先扩张再压缩？ |

### 之后：架构组装

`modules/02-architecture/` —— ⚠️ 该目录**只有 `index.md` 占位**，实验和教学文档尚未开发。先跳过，或对照 `reference/model/model_minimind.py` 的 `MiniMindBlock` 自行理解残差连接与组装顺序。

> 本仓库的教学内容到这里为止。实际训练（pretrain / SFT / LoRA）由 `reference/trainer/` 下的脚本承担，但需要额外的数据集，**不在本指引范围内** —— 先把上面 4 个模块学完再说。

---

## 4. 每个模块怎么学

以 `01-normalization` 为例（其他模块同理，把路径里的编号换掉）：

```bash
# ① 激活环境（每次都要）
source /home/dht/dht_workspace/minimind/.venv/bin/activate

# ② 进入模块的实验目录（必须在 experiments/ 下运行！）
cd /home/dht/dht_workspace/minimind/minimind-notes/modules/01-foundation/01-normalization/experiments

# ③ 跑实验
bash run_all.sh                    # 一键跑完本模块所有实验
python exp1_gradient_vanishing.py  # 或单独跑某个
```

> ⚠️ **必须在 `experiments/` 目录下运行**。脚本用 `sys.path.append('../../..')` 这种相对路径找项目根，换目录会导入失败。

**② 读理论**：同模块目录下的
- `teaching.md` —— 原理讲解（Why / What / How）
- `code_guide.md` —— 对照 MiniMind 源码逐段走读
- `quiz.md` —— 自测题

**③ 看结果**：图表输出到 `experiments/results/*.png`。

### 实测状态

| 模块 | 状态 |
|---|---|
| 01-normalization | ✅ 4 个实验全部跑通（`run_all.sh` 实测通过） |
| 02-position-encoding | ✅ exp1 实测通过，其余 3 个同类型 |
| 03-attention | ✅ 3 个实验已修复并跑通 |
| 04-feedforward | ✅ exp1 实测通过 |

> **已修复**：`03-attention/exp1_attention_basics.py` 原先在 `# TODO(human)` 处留白，却在下文直接引用未定义的 `attention_weights`，导致 `NameError` 崩溃。现已补上注意力计算（`scores = Q @ K.T` + `F.softmax`），运行结果为 "it" 最关注 "animal"（0.446），正好演示该实验要讲的代词消歧。

---

## 5. 记笔记

问答由 `llm-learning` skill 在你提问时**静默**记录到 `minimind-notes/notes.md`（按章节归类，每条含公式 + PyTorch 调用 + 从 tensor 手写）。

你不用管它；想手动记就说「记录一下」。

---

## 6. 一句话速查

```bash
# 每次开始
source /home/dht/dht_workspace/minimind/.venv/bin/activate
cd /home/dht/dht_workspace/minimind/minimind-notes

# 顺序：01-normalization → 02-position-encoding → 03-attention → 04-feedforward
# 每个模块：
cd modules/01-foundation/<模块名>/experiments && bash run_all.sh
# 然后读 ../teaching.md、../code_guide.md，做 ../quiz.md

# 注意：必须在 experiments/ 下运行；必须激活 venv
```
