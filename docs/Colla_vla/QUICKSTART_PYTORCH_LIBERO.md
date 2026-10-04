# 使用 PyTorch 训练 - 快速开始

本文档包含两个训练示例：
1. **使用 π₀.₅ 训练 LIBERO 数据**（pi05_libero）
2. **使用 π₀-DROID 权重训练**（pi0_droid）

---

## 示例 1: 使用 π₀.₅ 训练 LIBERO 数据

### 前置步骤

### 1. 设置 PyTorch 环境

运行设置脚本（会自动检测 Python 版本并应用补丁）：

```bash
bash scripts/setup_pytorch_training.sh
```

或者手动应用 transformers 库的补丁：

```bash
# 自动检测 Python 版本
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python${PYTHON_VERSION}/site-packages/transformers/
```

### 2. 转换 JAX 基础模型到 PyTorch

首先需要下载并转换 JAX 基础模型：

```bash
# 下载 JAX 基础模型（如果还没有）
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_base \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --output_path /18141169908/wyh/weights/pi05_base_pytorch
```

或者如果你已经下载了 JAX 模型：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_base \
    --checkpoint_dir /path/to/jax/pi05_base \
    --output_path /18141169908/wyh/weights/pi05_base_pytorch
```

### 3. 修改配置以使用 PyTorch 模型

编辑 `src/openpi/training/config.py`，找到 `pi05_libero` 配置（约第 735 行），修改 `pytorch_weight_path`：

```python
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="openpi/libero",  # 确保这个匹配你的数据集名称
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    # ... 其他配置 ...
    pytorch_weight_path="/18141169908/wyh/weights/pi05_base_pytorch",  # 改为你的 PyTorch 模型路径
    # ...
),
```

## 训练步骤

### 步骤 1: 设置环境变量

```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
```

### 步骤 2: 计算归一化统计（如果还没有）

```bash
export HF_LEROBOT_HOME=/18141169908/wyh
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

**注意**：`compute_norm_stats.py` 可能使用 JAX，但这对 PyTorch 训练也适用。

### 步骤 3: 开始 PyTorch 训练

#### 单 GPU 训练：

```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
uv run scripts/train_pytorch.py pi05_libero --exp_name my_pytorch_experiment --save_interval 5000
```

#### 多 GPU 训练（单节点）：

```bash
export HF_LEROBOT_HOME=/18141169908/wyh
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_libero --exp_name my_pytorch_experiment --save_interval 5000
```

将 `--nproc_per_node=2` 改为你的 GPU 数量。

#### 恢复训练：

```bash
uv run scripts/train_pytorch.py pi05_libero --exp_name my_pytorch_experiment --resume
```

## PyTorch vs JAX 训练的区别

| 特性 | JAX | PyTorch |
|------|-----|---------|
| 训练脚本 | `scripts/train.py` | `scripts/train_pytorch.py` |
| 模型格式 | JAX checkpoint | PyTorch safetensors |
| 精度 | 混合精度（默认） | bfloat16 或 float32 |
| 多 GPU | FSDP | DDP (torchrun) |
| EMA | 支持 | 不支持 |
| LoRA | 支持 | 不支持 |

## 检查点位置

训练完成后，检查点会保存在：
```
checkpoints/pi05_libero/my_pytorch_experiment/<step>/
```

每个检查点目录包含：
- `model.safetensors` - PyTorch 模型权重
- 其他配置文件

## 使用训练好的模型

训练完成后，可以直接使用 PyTorch 检查点进行推理：

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("pi05_libero")
checkpoint_dir = "checkpoints/pi05_libero/my_pytorch_experiment/30000"

# 自动检测 PyTorch 格式
policy = policy_config.create_trained_policy(config, checkpoint_dir)
```

---

## 示例 2: 使用 π₀-DROID 权重训练

如果你已经下载了 `pi0_droid` 的权重（在 `/18141169908/wyh/weights/pi0_droid/`），可以使用以下步骤进行训练。

### 前置步骤

#### 1. 设置 PyTorch 环境

```bash
bash scripts/setup_pytorch_training.sh
```

#### 2. 转换已下载的 pi0_droid JAX 模型到 PyTorch

由于你已经下载了 `pi0_droid` 的 JAX 权重，可以直接转换。

**推荐方式（带进度输出）**：

```bash
bash scripts/convert_pi0_droid_to_pytorch.sh
```

这个脚本会：
- 先测试参数加载是否正常
- 显示详细的进度信息
- 保存转换日志到 `/tmp/convert_log.txt`

**或者直接使用原始命令**：

```bash
PYTHONUNBUFFERED=1 uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi0_droid \
    --checkpoint_dir /18141169908/wyh/weights/pi0_droid \
    --output_path /18141169908/wyh/pi0_droid_pytorch
```

**注意**：
- 转换 11GB 模型可能需要 10-30 分钟
- 如果看起来卡住了，可能是正在加载大模型，请耐心等待
- 使用 `PYTHONUNBUFFERED=1` 确保输出不被缓冲
- 如果长时间没有输出，可以检查 `/tmp/convert_log.txt` 查看进度
- **重要**：转换脚本会自动复制 `assets` 目录（包含归一化统计）。如果转换后缺少 `assets` 目录，需要手动复制：
  ```bash
  cp -r /18141169908/wyh/weights/pi0_droid/assets /18141169908/wyh/weights/pi0_droid_pytorch/
  ```

**注意**：`pi0_droid` 配置使用的是 DROID 数据格式，不是 LIBERO。如果你想用 `pi0_droid` 权重训练 LIBERO 数据，需要创建一个新的配置。

#### 3. 创建自定义配置（使用 pi0_droid 权重训练 LIBERO）

如果你想用 `pi0_droid` 的权重来训练 LIBERO 数据，需要在 `src/openpi/training/config.py` 中添加一个新配置：

```python
TrainConfig(
    name="pi0_libero_from_droid",
    model=pi0_config.Pi0Config(action_horizon=10),  # pi0 配置，不是 pi05
    data=LeRobotLiberoDataConfig(
        repo_id="openpi/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    pytorch_weight_path="/18141169908/wyh/weights/pi0_droid_pytorch",  # 使用转换后的 PyTorch 模型
    num_train_steps=30_000,
),
```

#### 4. 或者使用 pi0_droid 训练 DROID 数据

如果你想用 `pi0_droid` 权重继续训练 DROID 数据，可以直接使用现有配置，但需要修改 `pytorch_weight_path`：

编辑 `src/openpi/training/config.py`，找到或创建 `pi0_droid` 配置，添加 `pytorch_weight_path`：

```python
TrainConfig(
    name="pi0_droid",
    model=pi0_config.Pi0Config(action_horizon=10),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
    pytorch_weight_path="/18141169908/wyh/weights/pi0_droid_pytorch",  # 添加这一行
),
```

### 训练步骤

#### 步骤 1: 设置环境变量

```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
```

#### 步骤 2: 计算归一化统计

**如果训练 LIBERO 数据**（使用自定义配置 `pi0_libero_from_droid`）：
```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
uv run scripts/compute_norm_stats.py --config-name pi0_libero_from_droid
```

**如果训练 DROID 数据**（使用 `pi0_droid` 配置）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_droid
```

#### 步骤 3: 开始 PyTorch 训练

**训练 LIBERO 数据**：
```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
uv run scripts/train_pytorch.py pi0_libero_from_droid --exp_name pi0_libero_from_droid_exp --save_interval 5000
```

**训练 DROID 数据**：
```bash
uv run scripts/train_pytorch.py pi0_droid --exp_name pi0_droid_exp --save_interval 5000
```

**多 GPU 训练**：
```bash
export HF_LEROBOT_HOME=/18141169908/wyh/dataset
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_libero_from_droid --exp_name pi0_libero_from_droid_exp --save_interval 5000
```

#### 恢复训练：
```bash
uv run scripts/train_pytorch.py pi0_libero_from_droid --exp_name pi0_libero_from_droid_exp --resume
# 或
uv run scripts/train_pytorch.py pi0_droid --exp_name pi0_droid_exp --resume
```

### pi0_droid vs pi05_libero 的区别

| 特性 | π₀-DROID | π₀.₅-LIBERO |
|------|----------|-------------|
| 基础模型 | π₀ | π₀.₅ |
| 原始训练数据 | DROID | 多种机器人数据 |
| 配置名称 | `pi0_droid` | `pi05_libero` |
| 模型大小 | 较小 | 较大 |
| 适用场景 | DROID 机器人 | LIBERO 基准测试 |

---

## 通用信息

### PyTorch vs JAX 训练的区别

| 特性 | JAX | PyTorch |
|------|-----|---------|
| 训练脚本 | `scripts/train.py` | `scripts/train_pytorch.py` |
| 模型格式 | JAX checkpoint | PyTorch safetensors |
| 精度 | 混合精度（默认） | bfloat16 或 float32 |
| 多 GPU | FSDP | DDP (torchrun) |
| EMA | 支持 | 不支持 |
| LoRA | 支持 | 不支持 |

### 检查点位置

训练完成后，检查点会保存在：
```
checkpoints/<config_name>/<exp_name>/<step>/
```

每个检查点目录包含：
- `model.safetensors` - PyTorch 模型权重
- 其他配置文件

### 使用训练好的模型

训练完成后，可以直接使用 PyTorch 检查点进行推理：

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("pi0_libero_from_droid")  # 或 "pi0_droid", "pi05_libero"
checkpoint_dir = "checkpoints/<config_name>/<exp_name>/30000"

# 自动检测 PyTorch 格式
policy = policy_config.create_trained_policy(config, checkpoint_dir)
```

## 故障排除

1. **转换模型失败**：确保 JAX 模型已完全下载
2. **找不到 PyTorch 模型**：检查 `pytorch_weight_path` 配置是否正确
3. **CUDA 错误**：确保 PyTorch 正确安装了 CUDA 支持
4. **内存不足**：可以尝试：
   - 减小 `batch_size`
   - 使用 `pytorch_training_precision="float32"`（虽然会使用更多内存）
5. **配置不存在**：如果使用自定义配置名称，确保已在 `config.py` 中正确定义

