"""Pi0.5 Skill-LoRA Composer: dynamic per-skill LoRA composition for multi-arm VLA."""

import dataclasses
import logging
from typing import Sequence

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.skill_lora as skill_lora
from openpi.models.pi0_multi_agent_attn import (
    TokenBoundaries,
    make_attn_mask,
    make_per_agent_attn_mask,
    make_per_agent_prefix_attn_mask,
    posemb_sincos,
)
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

DEFAULT_SKILL_NAMES = (
    "pick",
    "place",
    "lift",
    "align",
    "handover",
    "receive",
    "wait",
    "press",
    "pull",
    "push",
)


@dataclasses.dataclass(frozen=True)
class Pi0SkillLoRAComposerConfig(_model.BaseModelConfig):
    """Configuration for Pi0.5 with Skill-LoRA Composer."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    pi05: bool = True
    discrete_state_input: bool = None  # type: ignore

    num_agents: int = 2
    action_dim_per_agent: int = 8
    agent_to_wrist_image: tuple[tuple[int, str], ...] = (
        (0, "left_wrist_0_rgb"),
        (1, "right_wrist_0_rgb"),
    )
    global_image_key: str = "base_0_rgb"
    use_arm_wise_prefix_attn: bool = True
    use_arm_wise_suffix_attn: bool = True
    use_latent_collab_tokens: bool = True
    num_latent_collab_tokens: int = 4
    use_per_agent_prompt: bool = True
    max_token_len_per_agent: int | None = None

    num_skills: int = 10
    skill_lora_rank: int = 16
    skill_lora_alpha: float = 16.0
    apply_skill_lora_to_action_expert: bool = True
    apply_skill_lora_to_decoder: bool = True
    router_loss_weight: float = 0.1
    router_use_text_only: bool = False
    router_per_arm: bool = False

    def __post_init__(self):
        object.__setattr__(self, "pi05", True)
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", True)

    def get_max_token_len_per_agent(self) -> int:
        if self.max_token_len_per_agent is not None:
            return self.max_token_len_per_agent
        return self.max_token_len // self.num_agents

    def get_skill_lora_config(self) -> skill_lora.SkillLoRAConfig:
        return skill_lora.SkillLoRAConfig(
            num_skills=self.num_skills,
            rank=self.skill_lora_rank,
            alpha=self.skill_lora_alpha,
        )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0SkillLoRAComposer":
        return Pi0SkillLoRAComposer(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        images = {self.global_image_key: image_spec}
        image_masks = {self.global_image_key: image_mask_spec}
        for _, wrist_key in self.agent_to_wrist_image:
            images[wrist_key] = image_spec
            image_masks[wrist_key] = image_mask_spec
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=images,
                image_masks=image_masks,
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                skill_labels=jax.ShapeDtypeStruct([batch_size, self.num_agents], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_agent_to_wrist_dict(self) -> dict[int, str]:
        return dict(self.agent_to_wrist_image)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.Nothing


class SkillRouter(nnx.Module):
    """Per-arm skill router: h_{i,t} -> logits over K skills."""

    def __init__(self, width: int, num_skills: int, *, rngs: nnx.Rngs):
        hidden = max(width // 2, 64)
        self.fc1 = nnx.Linear(width, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, num_skills, rngs=rngs)

    def __call__(self, h: at.Float[at.Array, "b a d"]) -> at.Float[at.Array, "b a k"]:
        x = nnx.swish(self.fc1(h))
        return self.fc2(x)


class Pi0SkillLoRAComposer(_model.BaseModel):
    """Pi0.5 multi-agent model with Skill-LoRA composition."""

    def __init__(self, config: Pi0SkillLoRAComposerConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = True

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        if config.apply_skill_lora_to_action_expert:
            skill_cfg = config.get_skill_lora_config()
            action_expert_config = dataclasses.replace(
                action_expert_config,
                skill_lora_configs={"attn": skill_cfg, "ffn": skill_cfg},
            )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        skill_cfg = config.get_skill_lora_config() if config.apply_skill_lora_to_decoder else None
        expert_width = _gemma.get_config(config.action_expert_variant).width
        self.action_in_proj = skill_lora.SkillBankLinear(
            config.action_dim, expert_width, skill_lora_config=skill_cfg, rngs=rngs
        )
        self.action_out_proj = skill_lora.SkillBankLinear(
            expert_width, config.action_dim, skill_lora_config=skill_cfg, rngs=rngs
        )
        self.time_mlp_in = nnx.Linear(expert_width, expert_width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(expert_width, expert_width, rngs=rngs)

        self._agent_to_wrist_image = config.get_agent_to_wrist_dict()

        if config.use_latent_collab_tokens and config.num_latent_collab_tokens > 0:
            self.latent_collab_tokens = nnx.Param(
                jax.random.normal(rngs(), (config.num_latent_collab_tokens, paligemma_config.width)) * 0.02
            )
        else:
            self.latent_collab_tokens = None

        if config.router_per_arm:
            self.skill_routers = nnx.Dict(
                {
                    str(i): SkillRouter(paligemma_config.width, config.num_skills, rngs=rngs)
                    for i in range(config.num_agents)
                }
            )
            self.skill_router = None
        else:
            self.skill_router = SkillRouter(paligemma_config.width, config.num_skills, rngs=rngs)
            self.skill_routers = None
        self.deterministic = True

    def _pool_per_arm_context(
        self,
        prefix_out: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        boundaries: TokenBoundaries,
    ) -> jnp.ndarray:
        """Mean-pool prefix tokens per arm -> [B, num_agents, width]."""
        batch_size = prefix_out.shape[0]
        width = prefix_out.shape[-1]
        num_agents = self.config.num_agents
        h = jnp.zeros((batch_size, num_agents, width), dtype=prefix_out.dtype)
        lct_start, lct_end = boundaries.latent_collab_range

        for agent_id in range(num_agents):
            indices = []
            if not self.config.router_use_text_only:
                global_start, global_end = boundaries.global_image_range
                indices.extend(range(global_start, global_end))
                wrist_name = self._agent_to_wrist_image.get(agent_id)
                if wrist_name and wrist_name in boundaries.image_boundaries:
                    ws, we = boundaries.image_boundaries[wrist_name]
                    indices.extend(range(ws, we))
            if agent_id in boundaries.per_agent_language_ranges:
                ls, le = boundaries.per_agent_language_ranges[agent_id]
                indices.extend(range(ls, le))
            if not self.config.router_use_text_only and lct_start < lct_end:
                indices.extend(range(lct_start, lct_end))
            if not indices:
                continue
            idx = jnp.array(indices, dtype=jnp.int32)
            tokens = prefix_out[:, idx, :]
            mask = prefix_mask[:, idx]
            mask_f = mask.astype(prefix_out.dtype)[..., None]
            pooled = jnp.sum(tokens * mask_f, axis=1) / (jnp.sum(mask_f, axis=1) + 1e-8)
            h = h.at[:, agent_id, :].set(pooled)
        return h

    def _compute_skill_mixtures(self, h: jnp.ndarray) -> dict[str, jnp.ndarray]:
        if self.config.router_per_arm:
            # h: [B, A, D] — each arm's dedicated router processes only its own slice
            logits_list = [
                self.skill_routers[str(i)](h[:, i, :])  # [B, K]
                for i in range(self.config.num_agents)
            ]
            r_logits = jnp.stack(logits_list, axis=1)  # [B, A, K]
        else:
            r_logits = self.skill_router(h)  # [B, A, K]
        idx = jnp.argmax(r_logits, axis=-1)  # [B, A]
        m_final = jax.lax.stop_gradient(jax.nn.one_hot(idx, self.config.num_skills))
        return {"r_logits": r_logits, "m_final": m_final}

    def _build_per_token_skill_mixture(
        self,
        boundaries: TokenBoundaries,
        m_final: jnp.ndarray,
        suffix_len: int,
    ) -> jnp.ndarray:
        """Build [B, suffix_len, K] mixture from per-arm m_final [B, A, K]."""
        batch_size = m_final.shape[0]
        k = self.config.num_skills
        mixture = jnp.zeros((batch_size, suffix_len, k), dtype=m_final.dtype)
        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
            start, end = boundaries.action_token_boundaries[agent_id]
            end = min(end, suffix_len)
            if start >= suffix_len:
                continue
            agent_mix = m_final[:, agent_id, :]
            mixture = mixture.at[:, start:end, :].set(
                einops.repeat(agent_mix, "b k -> b t k", t=end - start)
            )
        return mixture

    def _skill_aux_loss(
        self,
        mix: dict[str, jnp.ndarray],
        skill_labels: jnp.ndarray | None,
    ) -> jnp.ndarray:
        if skill_labels is None:
            return jnp.array(0.0, dtype=jnp.float32)
        valid = skill_labels >= 0  # [B, A] bool, traced
        r_logits = mix["r_logits"]
        log_probs = jax.nn.log_softmax(r_logits, axis=-1)
        # Clamp labels to 0 so one_hot is well-defined even for unlabeled (-1) entries;
        # their contribution is zeroed out by the valid mask.
        labels_safe = jnp.where(valid, skill_labels, 0)
        per_arm_loss = -jnp.sum(jax.nn.one_hot(labels_safe, self.config.num_skills) * log_probs, axis=-1)
        num_valid = jnp.sum(valid).astype(jnp.float32)
        loss_r = jnp.sum(valid * per_arm_loss) / (num_valid + 1e-8)
        return self.config.router_loss_weight * loss_r

    @at.typecheck
    def embed_prefix_with_boundaries(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        TokenBoundaries,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        image_boundaries = {}
        global_image_range = (0, 0)
        per_agent_language_ranges = {}
        current_pos = 0

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            num_tokens = image_tokens.shape[1]
            start_pos = current_pos
            end_pos = current_pos + num_tokens
            if name == self.config.global_image_key or "base" in name or "global" in name:
                global_image_range = (start_pos, end_pos)
            else:
                image_boundaries[name] = (start_pos, end_pos)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=num_tokens)
            )
            ar_mask += [False] * num_tokens
            current_pos = end_pos

        if obs.tokenized_prompt is not None:
            if self.config.use_per_agent_prompt:
                max_len_per_agent = self.config.get_max_token_len_per_agent()
                for agent_id in range(self.config.num_agents):
                    start_in_prompt = agent_id * max_len_per_agent
                    end_in_prompt = start_in_prompt + max_len_per_agent
                    agent_prompt = obs.tokenized_prompt[:, start_in_prompt:end_in_prompt]
                    agent_prompt_mask = obs.tokenized_prompt_mask[:, start_in_prompt:end_in_prompt]
                    agent_tokens = self.PaliGemma.llm(agent_prompt, method="embed")
                    tokens.append(agent_tokens)
                    input_mask.append(agent_prompt_mask)
                    ar_mask += [False] * max_len_per_agent
                    per_agent_language_ranges[agent_id] = (current_pos, current_pos + max_len_per_agent)
                    current_pos += max_len_per_agent
            else:
                tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                tokens.append(tokenized_inputs)
                input_mask.append(obs.tokenized_prompt_mask)
                ar_mask += [False] * tokenized_inputs.shape[1]
                lang_end = current_pos + tokenized_inputs.shape[1]
                for agent_id in range(self.config.num_agents):
                    per_agent_language_ranges[agent_id] = (current_pos, lang_end)
                current_pos = lang_end

        latent_collab_start = current_pos
        if self.config.use_latent_collab_tokens and self.latent_collab_tokens is not None:
            batch_size = list(obs.images.values())[0].shape[0]
            lct_tokens = einops.repeat(self.latent_collab_tokens.value, "n e -> b n e", b=batch_size)
            num_lct = self.config.num_latent_collab_tokens
            tokens.append(lct_tokens)
            input_mask.append(jnp.ones((batch_size, num_lct), dtype=jnp.bool_))
            ar_mask += [False] * num_lct
            current_pos += num_lct
        latent_collab_end = current_pos

        boundaries = TokenBoundaries(
            image_boundaries=image_boundaries,
            global_image_range=global_image_range,
            prefix_length=current_pos,
            language_range=(0, current_pos),
            latent_collab_range=(latent_collab_start, latent_collab_end),
            per_agent_language_ranges=per_agent_language_ranges,
        )
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, boundaries

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        per_token_mixture: at.Float[at.Array, "b sk k"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
        dict[int, tuple[int, int]],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        action_token_boundaries = {}
        batch_size = obs.state.shape[0]
        current_suffix_idx = 0

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        adarms_cond = time_emb

        for agent_id in range(self.config.num_agents):
            agent_action_start = current_suffix_idx
            if per_token_mixture is not None:
                mix_slice = per_token_mixture[:, agent_action_start : agent_action_start + self.action_horizon, :]
                agent_action_tokens = self.action_in_proj(noisy_actions, mix_slice)
            else:
                agent_action_tokens = self.action_in_proj(noisy_actions)
            tokens.append(agent_action_tokens)
            input_mask.append(jnp.ones((batch_size, self.action_horizon), dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (self.action_horizon - 1))
            action_end = current_suffix_idx + self.action_horizon
            action_token_boundaries[agent_id] = (agent_action_start, action_end)
            current_suffix_idx = action_end

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond, action_token_boundaries

    def _decode_action_outputs(
        self,
        suffix_out: jnp.ndarray,
        boundaries: TokenBoundaries,
        per_token_mixture: jnp.ndarray | None,
    ) -> jnp.ndarray:
        agent_velocities = []
        num_agents = self.config.num_agents
        dim_per_agent = self.config.action_dim_per_agent
        for agent_id in range(num_agents):
            action_start, action_end = boundaries.action_token_boundaries[agent_id]
            agent_action_out = suffix_out[:, action_start:action_end, :]
            if per_token_mixture is not None:
                mix_slice = per_token_mixture[:, action_start:action_end, :]
                agent_v_full = self.action_out_proj(agent_action_out, mix_slice)
            else:
                agent_v_full = self.action_out_proj(agent_action_out)
            if agent_id < num_agents - 1:
                agent_v = agent_v_full[:, :, :dim_per_agent]
            else:
                remaining_dims = self.action_dim - agent_id * dim_per_agent
                agent_v = agent_v_full[:, :, :remaining_dims]
            agent_velocities.append(agent_v)
        return jnp.concatenate(agent_velocities, axis=-1)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=image_keys
        )

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, boundaries = self.embed_prefix_with_boundaries(observation)

        prefix_attn_mask = make_per_agent_prefix_attn_mask(
            prefix_mask,
            prefix_ar_mask,
            boundaries,
            num_agents=self.config.num_agents,
            agent_to_wrist_image=self._agent_to_wrist_image,
            apply_prefix_mask=self.config.use_arm_wise_prefix_attn,
        )
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, adarms_cond=[None, None]
        )
        assert prefix_out is not None

        h = self._pool_per_arm_context(prefix_out, prefix_mask, boundaries)
        mix = self._compute_skill_mixtures(h)
        m_final = mix["m_final"]

        suffix_len = self.config.num_agents * self.action_horizon
        boundaries.action_token_boundaries = {
            i: (i * self.action_horizon, (i + 1) * self.action_horizon) for i in range(self.config.num_agents)
        }
        boundaries.action_horizon_per_agent = self.action_horizon
        boundaries.num_state_tokens = 0
        per_token_mixture = self._build_per_token_skill_mixture(boundaries, m_final, suffix_len)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, action_token_boundaries = self.embed_suffix(
            observation, x_t, time, per_token_mixture
        )
        boundaries.action_token_boundaries = action_token_boundaries

        # Reuse prefix KV cache from the first forward — prefix token representations are
        # identical (no LoRA on backbone, prefix can't attend to suffix), so running prefix
        # through the 2B model a second time would be pure wasted compute.
        suffix_len_local = suffix_tokens.shape[1]
        suffix_attn_mask = self._make_suffix_self_attn_mask(
            suffix_mask, suffix_ar_mask, boundaries, suffix_len_local
        )
        cross_attn_mask = self._make_cross_attn_mask(
            prefix_mask, suffix_mask, boundaries, suffix_len_local
        )
        full_attn_mask = jnp.concatenate([cross_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            skill_mixtures=[None, per_token_mixture],
        )
        v_t = self._decode_action_outputs(suffix_out, boundaries, per_token_mixture)
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        aux_loss = self._skill_aux_loss(mix, observation.skill_labels)
        return action_loss + aux_loss, {"router_loss": aux_loss}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_skill_mixture: bool = True,
    ) -> _model.Actions | tuple[_model.Actions, jnp.ndarray]:
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=image_keys)
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask, boundaries = self.embed_prefix_with_boundaries(observation)
        prefix_attn_mask = make_per_agent_prefix_attn_mask(
            prefix_mask,
            prefix_ar_mask,
            boundaries,
            num_agents=self.config.num_agents,
            agent_to_wrist_image=self._agent_to_wrist_image,
            apply_prefix_mask=self.config.use_arm_wise_prefix_attn,
        )
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, adarms_cond=[None, None]
        )
        assert prefix_out is not None

        h = self._pool_per_arm_context(prefix_out, prefix_mask, boundaries)
        mix = self._compute_skill_mixtures(h)
        m_final = mix["m_final"]

        suffix_len = self.config.num_agents * self.action_horizon
        boundaries.action_token_boundaries = {
            i: (i * self.action_horizon, (i + 1) * self.action_horizon) for i in range(self.config.num_agents)
        }
        boundaries.action_horizon_per_agent = self.action_horizon
        boundaries.num_state_tokens = 0
        per_token_mixture = self._build_per_token_skill_mixture(boundaries, m_final, suffix_len)

        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, action_token_boundaries = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), per_token_mixture
            )
            boundaries.action_token_boundaries = action_token_boundaries
            suffix_len_local = suffix_tokens.shape[1]
            prefix_len = prefix_tokens.shape[1]
            suffix_attn_mask = self._make_suffix_self_attn_mask(
                suffix_mask, suffix_ar_mask, boundaries, suffix_len_local
            )
            cross_attn_mask = self._make_cross_attn_mask(
                prefix_mask, suffix_mask, boundaries, suffix_len_local
            )
            full_attn_mask = jnp.concatenate([cross_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                skill_mixtures=[None, per_token_mixture],
            )
            v_t = self._decode_action_outputs(suffix_out, boundaries, per_token_mixture)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if return_skill_mixture:
            return x_0, m_final
        return x_0

    def _make_suffix_self_attn_mask(
        self,
        suffix_mask: jnp.ndarray,
        suffix_ar_mask: jnp.ndarray,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> jnp.ndarray:
        base_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        if not self.config.use_arm_wise_suffix_attn:
            return base_mask
        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
            agent_action_start, agent_action_end = boundaries.action_token_boundaries[agent_id]
            if agent_action_end > suffix_len:
                agent_action_end = suffix_len
            if agent_action_start >= suffix_len:
                continue
            for query_idx in range(agent_action_start, agent_action_end):
                if query_idx >= suffix_len:
                    break
                allowed_mask = jnp.zeros(suffix_len, dtype=jnp.bool_)
                for key_idx in range(agent_action_start, query_idx + 1):
                    allowed_mask = allowed_mask.at[key_idx].set(True)
                allowed_mask_batched = jnp.logical_and(allowed_mask[None, :], suffix_mask)
                base_mask = base_mask.at[:, query_idx, :].set(allowed_mask_batched)
        return base_mask

    def _make_cross_attn_mask(
        self,
        prefix_mask: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> jnp.ndarray:
        batch_size = prefix_mask.shape[0]
        prefix_len = prefix_mask.shape[1]
        base_cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        if not self.config.use_arm_wise_suffix_attn:
            return base_cross_mask

        def create_agent_prefix_mask(agent_id):
            allowed_prefix_mask = jnp.zeros(prefix_len, dtype=jnp.bool_)
            global_start, global_end = boundaries.global_image_range
            allowed_prefix_mask = allowed_prefix_mask.at[global_start:global_end].set(True)
            if agent_id in self._agent_to_wrist_image:
                wrist_name = self._agent_to_wrist_image[agent_id]
                if wrist_name in boundaries.image_boundaries:
                    ws, we = boundaries.image_boundaries[wrist_name]
                    allowed_prefix_mask = allowed_prefix_mask.at[ws:we].set(True)
            if agent_id in boundaries.per_agent_language_ranges:
                ls, le = boundaries.per_agent_language_ranges[agent_id]
                allowed_prefix_mask = allowed_prefix_mask.at[ls:le].set(True)
            lct_start, lct_end = boundaries.latent_collab_range
            if lct_start < lct_end:
                allowed_prefix_mask = allowed_prefix_mask.at[lct_start:lct_end].set(True)
            return allowed_prefix_mask

        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
            agent_action_start, agent_action_end = boundaries.action_token_boundaries[agent_id]
            if agent_action_end > suffix_len:
                agent_action_end = suffix_len
            if agent_action_start >= suffix_len:
                continue
            allowed_prefix_mask = create_agent_prefix_mask(agent_id)
            combined_mask_batched = jnp.logical_and(allowed_prefix_mask[None, :], prefix_mask)
            for suffix_idx in range(agent_action_start, agent_action_end):
                base_cross_mask = base_cross_mask.at[:, suffix_idx, :].set(combined_mask_batched)
        return base_cross_mask
