"""Sanity tests for Pi0 Skill-LoRA Composer."""

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0_skill_lora_composer import Pi0SkillLoRAComposerConfig


def _make_config() -> Pi0SkillLoRAComposerConfig:
    return Pi0SkillLoRAComposerConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=4,
        max_token_len=48,
        num_agents=2,
        action_dim_per_agent=8,
        num_skills=4,
        skill_lora_rank=4,
        use_latent_collab_tokens=True,
        num_latent_collab_tokens=2,
        use_per_agent_prompt=True,
        max_token_len_per_agent=24,
    )


def test_skill_lora_composer_forward_and_sample():
    config = _make_config()
    rng = jax.random.key(0)
    model = config.create(rng)

    obs = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)
    # Valid token ids for embedder (avoid OOB)
    obs = _model.Observation(
        images=obs.images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=jnp.zeros_like(obs.tokenized_prompt),
        tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask, dtype=jnp.bool_),
        skill_labels=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
    )

    loss_rng, sample_rng = jax.random.split(rng)
    loss = model.compute_loss(loss_rng, obs, actions, train=False)
    assert loss.shape == (2, config.action_horizon)

    actions_out, m_final = model.sample_actions(sample_rng, obs, num_steps=2)
    assert actions_out.shape == (2, config.action_horizon, config.action_dim)
    assert m_final.shape == (2, config.num_agents, config.num_skills)
    # m_final must be one-hot: each arm selects exactly one skill
    assert jnp.allclose(m_final.sum(axis=-1), jnp.ones((2, config.num_agents)))
    assert jnp.all((m_final == 0) | (m_final == 1))
