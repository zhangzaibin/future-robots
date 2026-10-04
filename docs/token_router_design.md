# Token Router 设计方案 (Hybrid Approach)

## 1. 背景与目标

当前 `pi0_multi_agent_attn.py` 使用**静态硬编码**的注意力路由规则：
- 每个 agent 的 tokens 只能 attend 到固定的 token 类型（global image、自己的 wrist、自己的 prompt、LCT 等）
- 路由规则是预定义的，无法根据输入内容动态调整

**目标**：在保留现有静态路由作为基础的同时，引入一个轻量的 **Token Router** 模块，根据输入动态调制注意力掩码。

---

## 2. 设计方案

### 2.1 核心思路：Hybrid Mask Modulation

```
final_attn_mask = base_static_mask * router_output
```

- `base_static_mask`：现有 `make_per_agent_attn_mask` 生成的静态掩码（0/1 二值）
- `router_output`：Router 预测的调制系数（0-1 连续值）
- `final_attn_mask`：逐元素相乘，实现动态调制

这样做的好处：
1. 保留静态路由的归纳偏置（inductive bias）
2. Router 只学习"调制"，训练更稳定
3. 推理时可以选择退化为纯静态路由

### 2.2 Router 输入特征

Router 需要知道**谁在 attend（query）** 和 **可以被 attend 到什么（key）**：

```python
@dataclasses.dataclass
class RouterContext:
    """Router 的输入上下文"""
    # Query 侧特征
    query_emb: jnp.ndarray           # [B, Q, D] query token embeddings
    query_type: jnp.ndarray           # [B, Q] token types (enum: GLOBAL=0, WRIST=1, LANG=2, LCT=3, STATE=4, ACTION=5)
    query_agent_id: jnp.ndarray      # [B, Q] agent IDs (0, 1, ... 或 -1 表示无归属)

    # Key 侧特征
    key_emb: jnp.ndarray             # [B, K, D] key token embeddings
    key_type: jnp.ndarray            # [B, K] token types
    key_agent_id: jnp.ndarray        # [B, K] agent IDs

    # 可选：任务提示（从 prompt 提取的全局语义）
    task_hint: jnp.ndarray | None    # [B, D] 来自 prompt 的任务 embedding
```

### 2.3 Token Type 分类

```python
class TokenType(Enum):
    GLOBAL = 0      # 全局图像 tokens
    WRIST = 1       # 手腕图像 tokens
    LANG = 2        # 语言/prompt tokens
    LCT = 3         # Latent Collaboration Tokens
    STATE = 4       # 状态 tokens
    ACTION = 5      # 动作 tokens
```

### 2.4 Router 网络结构

```
Query Embedding     Key Embedding
      │                  │
      ▼                  ▼
 ┌─────────┐        ┌─────────┐
 │  Q Proj │        │  K Proj │
 └────┬────┘        └────┬────┘
      │                  │
      ▼                  ▼
 ┌─────────────────────────────┐
 │  Add Type Embeddings        │  (可学习的 type interaction matrix)
 │  Add Agent Embeddings       │
 │  (Optional) Mul Task Gate  │
 └──────────────┬──────────────┘
                │
                ▼
         ┌───────────┐
         │  Q @ K^T  │  → [B, Q, K] scores
         └─────┬─────┘
               │
               ▼
         ┌───────────┐
         │  Sigmoid  │  → [B, Q, K] router_output (0~1)
         └───────────┘
```

**详细结构**：

```python
class TokenRouter(nnx.Module):
    def __init__(self, config: Pi0MultiAgentAttnConfig):
        emb_dim = config.emb_dim          # e.g., 2048
        router_dim = config.router_dim     # e.g., 128 (远小于 emb_dim)

        # 投影层
        self.q_proj = nnx.Linear(emb_dim, router_dim)
        self.k_proj = nnx.Linear(emb_dim, router_dim)

        # Type embeddings: [6, router_dim] (6 种 token type)
        self.type_emb = nnx.Embedding(num_types=6, emb_dim=router_dim)

        # Agent embeddings: [num_agents, router_dim]
        self.agent_emb = nnx.Embedding(num_agents=2, emb_dim=router_dim)

        # Task conditioning (可选)
        self.task_gate = nnx.Sequential([
            nnx.Linear(emb_dim, router_dim),
            nnx.sigmoid
        ])

        # Type interaction bias: 可学习的 type pair 偏好
        # 形状 [6, 6]，表示 (query_type, key_type) 的偏好强度
        self.type_bias = nnx.Param(
            jax.random.normal(rngs(), (6, 6)) * 0.02
        )

        # 输出调制系数
        self.scale = nnx.Param(jnp.array(1.0))

    def forward(self, ctx: RouterContext) -> jnp.ndarray:
        # 投影
        q = self.q_proj(ctx.query_emb)  # [B, Q, router_dim]
        k = self.k_proj(ctx.key_emb)    # [B, K, router_dim]

        # 添加 type 和 agent embedding
        q = q + self.type_emb(ctx.query_type) + self.agent_emb(ctx.query_agent_id)
        k = k + self.type_emb(ctx.key_type) + self.agent_emb(ctx.key_agent_id)

        # Task gating (可选)
        if ctx.task_hint is not None:
            gate = self.task_gate(ctx.task_hint)  # [B, router_dim]
            q = q * gate[:, None, :]
            k = k * gate[:, None, :]

        # 计算注意力分数
        scores = jnp.einsum("bqd,bkd->bqk", q, k)  # [B, Q, K]

        # 添加 type interaction bias
        type_bias = self.type_bias[ctx.query_type, ctx.key_type]  # [B, Q, K]
        scores = scores + type_bias

        # 转换为 0-1 调制系数
        router_output = jax.nn.sigmoid(scores * self.scale)

        return router_output  # [B, Q, K]
```

---

## 3. 与现有代码的集成

### 3.1 配置新增

```python
# 在 Pi0MultiAgentAttnConfig 中新增
@dataclasses.dataclass(frozen=True)
class Pi0MultiAgentAttnConfig(_model.BaseModelConfig):
    # ... existing fields ...

    # Token Router 配置
    use_token_router: bool = False          # 是否启用 router
    router_dim: int = 128                   # Router 内部维度
    router_train_temp: float = 1.0           # 训练时 temperature（用于伯努利采样）
    router_use_hard_mask: bool = False       # 推理时是否用 hard mask (0/1)
```

### 3.2 模型集成

```python
class Pi0MultiAgentAttn(_model.BaseModel):
    def __init__(self, config: Pi0MultiAgentAttnConfig, rngs: nnx.Rngs):
        # ... existing init ...

        # 新增 Token Router
        if config.use_token_router:
            self.token_router = TokenRouter(config)

    def compute_loss(self, observation, actions, ...):
        # ... prefix/suffix embedding ...

        # 生成静态注意力掩码
        static_mask = make_per_agent_attn_mask(...)  # [B, total_len, total_len]

        # Router 动态调制
        if self.config.use_token_router:
            router_ctx = RouterContext(
                query_emb=all_tokens,       # 需要分别构建 prefix/suffix context
                query_type=all_types,
                query_agent_id=all_agent_ids,
                key_emb=all_tokens,
                key_type=all_types,
                key_agent_id=all_agent_ids,
                task_hint=task_embedding,   # 可选
            )
            router_output = self.token_router(router_ctx)

            # 逐元素调制
            final_mask = static_mask * router_output
        else:
            final_mask = static_mask

        # 后续使用 final_mask 进行注意力计算
        ...
```

---

## 4. 训练策略

### 4.1 损失函数

```python
def compute_loss_with_router(router_output, static_mask, task_loss, config):
    """
    router_output: [B, Q, K] 路由器输出的调制系数
    static_mask:   [B, Q, K] 静态掩码 (0/1)
    task_loss:     原始任务 loss
    """

    # 1. 基础任务 loss（不变）
    total_loss = task_loss

    # 2. 稀疏性正则：鼓励 router 输出接近 0 或 1
    #    - 太接近 0.5 会导致所有 token 都 attend，浪费计算
    sparsity_loss = -jnp.mean(router_output * jnp.log(router_output + 1e-8) +
                              (1 - router_output) * jnp.log(1 - router_output + 1e-8))

    # 3. 一致性正则：鼓励 router 不要完全推翻静态路由
    #    - 静态 mask=0 的位置，router 应该也接近 0
    #    - 静态 mask=1 的位置，router 可以自由调整
    mask_zero_loss = jnp.mean((1 - static_mask) * router_output**2)
    mask_one_variance = jnp.mean(static_mask * (router_output - 0.5)**2)

    total_loss = total_loss + 0.01 * sparsity_loss + 0.1 * mask_zero_loss

    return total_loss
```

### 4.2 训练流程

```
1. 初始化：Router 权重随机初始化，静态 mask 的规则被保留
2. 冷启动：前几个 epoch 逐渐增加 router_loss 的权重，避免破坏预训练
3. 微调：可以用较小的学习率单独训练 router
```

---

## 5. 推理时的行为

### 5.1 Soft Mode（默认）

```python
# 推理时使用 soft mask（连续值）
final_mask = static_mask * router_output
```

### 5.2 Hard Mode

```python
# 推理时将 router_output 二值化（0 或 1）
if config.router_use_hard_mask:
    router_output = jnp.round(router_output)  # 0 或 1
final_mask = static_mask * router_output
```

### 5.3 Fallback Mode

```python
# 如果 router 不可靠或用于蒸馏，直接用静态 mask
final_mask = static_mask
```

---

## 6. 部署考虑

### 6.1 额外计算开销

- Router 本身：`O(B * Q * K * router_dim)` 远小于标准 attention `O(B * Q * K * emb_dim)`
- 当 `router_dim=128, emb_dim=2048` 时，开销约为原始 attention 的 **6%**

### 6.2 可选优化

1. **预计算 Router 输出**：对于相同的输入配置，可以缓存 router 结果
2. **稀疏化**：只对部分 layer 使用 router，或只对 action tokens 使用
3. **知识蒸馏**：用静态 mask 作为 teacher，训练 router 近似静态 mask 的效果

---

## 7. 实现步骤

### Phase 1: 基础框架
- [ ] 新增 `RouterContext` dataclass
- [ ] 新增 `TokenRouter` 模块
- [ ] 新增 `use_token_router` 配置选项
- [ ] 在 `Pi0MultiAgentAttn.__init__` 中实例化 router
- [ ] 在 `compute_loss` 中集成 router 调用

### Phase 2: 训练
- [ ] 实现带 router loss 的训练循环
- [ ] 调整超参数（lambda_reg 等）
- [ ] 验证 router 能学习到合理的调制

### Phase 3: 优化
- [ ] 实现 Hard Mode
- [ ] 实现 Fallback Mode
- [ ] 性能测试和优化

---

## 8. 预期效果

| 场景 | 静态 Mask | Hybrid Router |
|------|-----------|---------------|
| 简单独立任务 | 正常工作 | Router 学会保持/轻微调制 |
| 需要跨臂协作 | LCT 作为唯一桥接 | Router 可以增强/减弱协作强度 |
| 某个 wrist 未使用 | 固定 attend 规则 | Router 可以学会抑制不必要的 attend |
| 新任务类型 | 完全依赖预定义规则 | Router 可以根据 prompt 动态调整 |

---

## 9. 附录：Type Interaction Matrix 初始化

基于现有静态规则的初始化值（可学习调整）：

```python
# 初始值：0=不允许，0.5=中立，1=允许
INITIAL_TYPE_BIAS = jnp.array([
    #    GLB  WRT  LNG  LCT  STT  ACT
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0],  # GLOBAL attends to all
    [1.0,  0.5,  0.5,  0.0,  0.0,  0.0],  # WRIST sees: global, self wrist, self lang
    [1.0,  0.5,  0.5,  1.0,  0.0,  0.0],  # LANG sees: global, self wrist, self lang, LCT
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0],  # LCT sees all (cross-arm bridge)
    [1.0,  0.5,  0.5,  1.0,  0.5,  0.5],  # STATE sees: global, self wrist, self lang, LCT, self
    [1.0,  0.5,  0.5,  1.0,  0.5,  0.5],  # ACTION sees: global, self wrist, self lang, LCT, self
])
```

经过训练后，router 会学习到更适合具体任务的路由规则。
