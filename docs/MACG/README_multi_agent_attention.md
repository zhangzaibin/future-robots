# Pi0 Multi-Agent Attention

## Overview

This module implements **per-agent selective attention** for multi-robot scenarios. Each agent's action tokens only attend to:

1. **Global view image tokens** - Shared across all agents
2. **Their own wrist view image tokens** - Agent-specific visual input
3. **Their own prompt portion** - Agent-specific language instruction

They do **NOT** attend to other agents' wrist view tokens.

## Architecture

```
Token Sequence:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Global Image │ Agent0 Wrist │ Agent1 Wrist │ Language Tokens │ State │ Actions      │
│  256 tokens  │  256 tokens  │  256 tokens  │   ~48 tokens    │   1   │   50 tokens  │
└─────────────────────────────────────────────────────────────────────────────────────┘

Attention Pattern:
                           Keys (Prefix)
                    ┌──────────────────────────────────────────┐
                    │ Global │ Agent0 │ Agent1 │   Language    │
Queries             │  Image │ Wrist  │ Wrist  │   Tokens      │
────────────────────┼────────┼────────┼────────┼───────────────┤
Agent0 Actions      │   ✓    │   ✓    │   ✗    │  agent0 part  │
────────────────────┼────────┼────────┼────────┼───────────────┤
Agent1 Actions      │   ✓    │   ✗    │   ✓    │  agent1 part  │
────────────────────┴────────┴────────┴────────┴───────────────┘

✓ = Can attend
✗ = Cannot attend (masked out)
```

## Usage

### Configuration

```python
from openpi.models.pi0_multi_agent_attn import Pi0MultiAgentAttnConfig

config = Pi0MultiAgentAttnConfig(
    action_dim=32,
    action_horizon=50,
    max_token_len=250,
    num_agents=2,
    action_dim_per_agent=8,
    # Mapping: agent ID -> wrist image key
    agent_to_wrist_image=(
        (0, "left_wrist_0_rgb"),
        (1, "right_wrist_0_rgb"),
    ),
    global_image_key="base_0_rgb",
    parse_prompt_boundaries=True,  # Parse "agent0:..." from prompt
)
```

### Training Configuration

The training config is defined in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi0_multi_agent_attn_2arms_placebowls",
    model=Pi0MultiAgentAttnConfig(
        action_dim=32,
        action_horizon=50,
        max_token_len=250,
        num_agents=2,
        action_dim_per_agent=8,
        agent_to_wrist_image=(
            (0, "left_wrist_0_rgb"),
            (1, "right_wrist_0_rgb"),
        ),
        global_image_key="base_0_rgb",
        parse_prompt_boundaries=True,
    ),
    data=SimpleDataConfig(
        repo_id="your_dataset_path",
        # ... other data config
    ),
    # ... other training config
)
```

### Running Training

```bash
# Train the multi-agent attention model
uv run python scripts/train.py --config pi0_multi_agent_attn_2arms_placebowls
```

## Prompt Format

The prompt should follow this format for proper per-agent parsing:

```
agent0: Wait for the other robots. agent1: Place the bowl on the target.
```

The parser looks for `agent0:`, `agent1:`, `agent2:`, etc. markers to split the prompt.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int | 2 | Number of robot agents |
| `action_dim_per_agent` | int | 8 | Action dimensions per agent |
| `agent_to_wrist_image` | tuple | - | Mapping from agent ID to wrist image key |
| `global_image_key` | str | "base_0_rgb" | Key for the global/shared image |
| `parse_prompt_boundaries` | bool | True | Whether to parse prompt for agent markers |

## For 3-arm Setup

```python
config = Pi0MultiAgentAttnConfig(
    num_agents=3,
    action_dim_per_agent=8,
    agent_to_wrist_image=(
        (0, "left_wrist_0_rgb"),
        (1, "right_wrist_0_rgb"),
        (2, "third_wrist_0_rgb"),
    ),
    global_image_key="base_0_rgb",
    # ...
)
```

## Implementation Details

### Key Files

- `src/openpi/models/pi0_multi_agent_attn.py` - JAX implementation
- `src/openpi/models_pytorch/pi0_multi_agent_attn_pytorch.py` - PyTorch implementation
- `src/openpi/training/config.py` - Training configurations

### TokenBoundaries

The `TokenBoundaries` dataclass tracks where different token types are located:

```python
@dataclasses.dataclass
class TokenBoundaries:
    image_boundaries: dict[str, tuple[int, int]]  # image_name -> (start, end)
    prompt_boundaries: dict[int, tuple[int, int]]  # agent_id -> (start, end)
    global_image_range: tuple[int, int]
    prefix_length: int
    language_range: tuple[int, int]
```

### Per-Agent Attention Mask

The `make_per_agent_attn_mask` function creates the custom attention mask:

1. Start with standard attention mask (prefix bidirectional, suffix causal)
2. For each agent's action tokens, restrict attention to:
   - Global image tokens
   - Own wrist image tokens
   - Own prompt tokens (if parsed)
   - State token
   - Own previous action tokens (causal)




