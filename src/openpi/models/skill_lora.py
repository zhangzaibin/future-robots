"""Skill-LoRA bank: per-skill LoRA adapters composed via per-token mixture weights."""

import math
import re

import flax.linen as nn
import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class SkillLoRAConfig:
    """Configuration for a bank of skill-specific LoRA adapters."""

    num_skills: int
    rank: int
    alpha: float = 1.0
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


def _mix_skill_deltas(mixture: jnp.ndarray, deltas: jnp.ndarray) -> jnp.ndarray:
    """Mix per-skill deltas while preserving the original output layout.

    ``deltas`` is produced by vmapping one LoRA path and therefore has shape
    ``[num_skills, *result.shape]``. The batch/token axes are consecutive inside
    ``result.shape`` but are not always the first two axes (e.g. qkv is
    ``[3, B, S, ...]``). We mix only over the skill axis and keep each
    batch/token position independent.
    """
    k_mix = mixture.shape[-1]
    if deltas.shape[0] != k_mix:
        raise ValueError(f"Skill bank size {deltas.shape[0]} != mixture K {k_mix}")
    batch_size, seq_len = mixture.shape[:2]
    output_shape = deltas.shape[1:]
    candidate_axes = [
        i for i in range(len(output_shape) - 1) if output_shape[i : i + 2] == (batch_size, seq_len)
    ]
    if not candidate_axes:
        raise ValueError(
            f"Could not find batch/token axes {mixture.shape[:2]} in skill delta shape {output_shape}"
        )
    batch_axis = candidate_axes[0]
    token_axis = batch_axis + 1

    # Move result's B/S axes right after the skill axis: [K, B, S, *rest].
    deltas_b_s_first = jnp.moveaxis(deltas, (batch_axis + 1, token_axis + 1), (1, 2))
    mixed = jnp.einsum("bsk,kbs...->bs...", mixture, deltas_b_s_first)

    # Restore the original result axis order.
    output_axes = list(range(len(output_shape)))
    current_axes = [batch_axis, token_axis] + [i for i in output_axes if i not in (batch_axis, token_axis)]
    restore_perm = [current_axes.index(i) for i in output_axes]
    return jnp.transpose(mixed, restore_perm)


def _make_lora_eqns(eqn: str, axes: tuple[int, int], label: str) -> tuple[str, str]:
    if label in eqn:
        raise ValueError(f"{label} already in eqn: {eqn}")
    if not (m := re.match("(.*),(.*)->(.*)", eqn)):
        raise ValueError(f"Unsupported einsum eqn: {eqn}")
    lhs, rhs, out = m.groups()
    a_label, b_label = (rhs[x] for x in axes)
    a_rhs = rhs.replace(b_label, label)
    a_out = out.replace(b_label, label)
    eqn_a = f"{lhs},{a_rhs}->{a_out}"
    b_rhs = rhs.replace(a_label, label)
    eqn_b = f"{a_out},{b_rhs}->{out}"
    return eqn_a, eqn_b


class SkillBankEinsum(nn.Module):
    """Einsum with K skill-specific LoRA banks, mixed per token via mixture [B, T, K]."""

    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    skill_lora_config: SkillLoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)
        if config := self.skill_lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            k = config.num_skills
            self.w_a_bank = self.param("skill_lora_a_bank", config.init_fn, (k, *shape_a))
            self.w_b_bank = self.param("skill_lora_b_bank", nn.initializers.zeros, (k, *shape_b))

    @nn.compact
    def __call__(self, eqn: str, x, mixture: at.Float[at.Array, "b sk k"] | None = None):
        dtype = x.dtype
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.skill_lora_config:
            if mixture is None:
                return result
            eqn_a, eqn_b = _make_lora_eqns(eqn, config.axes, config.label)

            def one_skill(w_a, w_b):
                lora_a = jnp.einsum(eqn_a, x, w_a.astype(dtype))
                return jnp.einsum(eqn_b, lora_a, w_b.astype(dtype))

            # deltas: [K, *result.shape]
            deltas = jax.vmap(one_skill, in_axes=(0, 0))(self.w_a_bank, self.w_b_bank)
            delta = _mix_skill_deltas(mixture, deltas)
            result = result + delta.astype(dtype) * config.scaling_value

        return result


class SkillBankFeedForward(nn.Module):
    """Feed-forward with per-skill LoRA banks, mixed per token."""

    features: int
    hidden_dim: int
    skill_lora_config: SkillLoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.skill_lora_config:
            cfg = self.skill_lora_config
            k = cfg.num_skills
            self.w_gating_lora = (
                self.param("gating_einsum_skill_lora_a_bank", cfg.init_fn, (k, 2, self.features, cfg.rank)),
                self.param("gating_einsum_skill_lora_b_bank", nn.initializers.zeros, (k, 2, cfg.rank, self.hidden_dim)),
            )
            self.w_linear_lora = (
                self.param("linear_skill_lora_a_bank", cfg.init_fn, (k, self.hidden_dim, cfg.rank)),
                self.param("linear_skill_lora_b_bank", nn.initializers.zeros, (k, cfg.rank, self.features)),
            )

    def _dot_skill(
        self,
        x: at.Array,
        w: at.Array,
        lora_banks: tuple[at.Array, at.Array] | None,
        mixture: at.Float[at.Array, "b sk k"] | None,
    ) -> at.Array:
        dtype = x.dtype
        base = jnp.dot(x, w.astype(dtype))
        if lora_banks is None or mixture is None:
            return base
        w_a_bank, w_b_bank = lora_banks
        k = w_a_bank.shape[0]
        # x: [B, T, in], w_a_bank: [K, in, rank]
        # For gating: w_a_bank [K, 2, features, rank]
        if w_a_bank.ndim == 4:
            # gating path: x [B,T,F], w_a [K,2,F,R], w_b [K,2,R,H]
            b, t, _ = x.shape
            # Process each gating slice separately - caller uses _dot per slice
            raise NotImplementedError("Use _dot per gating slice for SkillBankFeedForward")
        # linear path: w_a [K, hidden, rank], w_b [K, rank, features]
        # lora per skill: [B, T, K, rank]
        lora_a = jnp.einsum("btf,kfr->btkr", x, w_a_bank.astype(dtype))
        lora_out = jnp.einsum("btkr,krf->btkf", lora_a, w_b_bank.astype(dtype))
        delta = jnp.einsum("btk,btkf->btf", mixture, lora_out)
        cfg = self.skill_lora_config
        assert cfg is not None
        return base + delta.astype(x.dtype) * cfg.scaling_value

    def _dot_gating_skill(
        self,
        x: at.Array,
        w: at.Array,
        lora_banks: tuple[at.Array, at.Array],
        mixture: at.Float[at.Array, "b sk k"] | None,
        gate_idx: int,
    ) -> at.Array:
        dtype = x.dtype
        base = jnp.dot(x, w.astype(dtype))
        if mixture is None:
            return base
        w_a_bank, w_b_bank = lora_banks
        w_a = w_a_bank[:, gate_idx]  # [K, F, R]
        w_b = w_b_bank[:, gate_idx]  # [K, R, H]
        lora_a = jnp.einsum("btf,kfr->btkr", x, w_a.astype(dtype))
        lora_out = jnp.einsum("btkr,krh->btkh", lora_a, w_b.astype(dtype))
        delta = jnp.einsum("btk,btkh->bth", mixture, lora_out)
        cfg = self.skill_lora_config
        assert cfg is not None
        return base + delta.astype(x.dtype) * cfg.scaling_value

    @nn.compact
    def __call__(self, x, mixture: at.Float[at.Array, "b sk k"] | None = None):
        dtype = x.dtype
        if self.w_gating_lora is None:
            w_gating = self.w_gating.astype(dtype)
            ff_gate = jnp.dot(x, w_gating[0])
            gate_value = nn.gelu(ff_gate)
            ff1 = jnp.dot(x, w_gating[1])
            activations = gate_value * ff1
            outputs = jnp.dot(activations, self.w_linear.astype(dtype))
            return outputs

        ff_gate = self._dot_gating_skill(x, self.w_gating[0], self.w_gating_lora, mixture, 0)
        gate_value = nn.gelu(ff_gate)
        ff1 = self._dot_gating_skill(x, self.w_gating[1], self.w_gating_lora, mixture, 1)
        activations = gate_value * ff1
        outputs = self._dot_skill(activations, self.w_linear, self.w_linear_lora, mixture)
        assert outputs.dtype == dtype
        return outputs


class SkillBankLinear(nnx.Module):
    """NNX Linear with K skill LoRA banks, mixed per token via mixture [B, T, K]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        skill_lora_config: SkillLoRAConfig | None = None,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.skill_lora_config = skill_lora_config
        self.kernel = nnx.Param(jax.random.normal(rngs(), (in_features, out_features)) * 0.02)
        if skill_lora_config is not None:
            k, r = skill_lora_config.num_skills, skill_lora_config.rank
            self.lora_a_bank = nnx.Param(
                jax.random.normal(rngs(), (k, in_features, r)) * 0.01
            )
            self.lora_b_bank = nnx.Param(
                jnp.zeros((k, r, out_features))
            )
        else:
            self.lora_a_bank = None
            self.lora_b_bank = None

    def __call__(
        self,
        x: at.Float[at.Array, "b t d_in"],
        mixture: at.Float[at.Array, "b sk k"] | None = None,
    ) -> at.Float[at.Array, "b t d_out"]:
        dtype = x.dtype
        result = x @ self.kernel.value.astype(dtype)
        if self.skill_lora_config is None or mixture is None:
            return result
        cfg = self.skill_lora_config
        assert self.lora_a_bank is not None and self.lora_b_bank is not None
        lora_a = jnp.einsum("btf,kfr->btkr", x, self.lora_a_bank.value.astype(dtype))
        lora_out = jnp.einsum("btkr,kro->btko", lora_a, self.lora_b_bank.value.astype(dtype))
        delta = jnp.einsum("btk,btko->bto", mixture, lora_out)
        return result + delta * cfg.scaling_value
