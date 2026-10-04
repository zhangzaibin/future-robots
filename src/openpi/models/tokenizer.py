import logging
import os
import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        # Check for local tokenizer path first (for offline use)
        local_tokenizer_path = os.getenv("PALIGEMMA_TOKENIZER_PATH")
        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            path = pathlib.Path(local_tokenizer_path)
            logging.info(f"Using local tokenizer from: {path}")
        else:
            try:
                path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            except Exception as e:
                error_msg = (
                    f"Failed to download tokenizer. Error: {e}\n"
                    f"To use a local tokenizer, set environment variable:\n"
                    f"  export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model\n"
                    f"Or download manually using:\n"
                    f"  bash scripts/download_tokenizer.sh /path/to/download/dir"
                )
                logging.error(error_msg)
                raise ConnectionError(error_msg) from e
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class PaligemmaPerAgentTokenizer:
    """Tokenizer for per-agent prompts in multi-agent settings.
    
    This tokenizer splits a prompt by "|" separator and tokenizes each agent's
    prompt portion independently, then pads each to max_len_per_agent.
    
    Example:
        Input: "Pick up the red block | Place the blue cube"
        Output: [agent0_tokens (padded to 24)] [agent1_tokens (padded to 24)]
        
    The output format is: [B, num_agents * max_len_per_agent]
    """
    
    def __init__(self, max_len_per_agent: int = 24, num_agents: int = 2, action_dim_per_agent: int = 8):
        self._max_len_per_agent = max_len_per_agent
        self._num_agents = num_agents
        self._action_dim_per_agent = action_dim_per_agent
        self._total_len = max_len_per_agent * num_agents
        
        # Check for local tokenizer path first (for offline use)
        local_tokenizer_path = os.getenv("PALIGEMMA_TOKENIZER_PATH")
        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            path = pathlib.Path(local_tokenizer_path)
            logging.info(f"Using local tokenizer from: {path}")
        else:
            try:
                path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            except Exception as e:
                error_msg = (
                    f"Failed to download tokenizer. Error: {e}\n"
                    f"To use a local tokenizer, set environment variable:\n"
                    f"  export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model\n"
                    f"Or download manually using:\n"
                    f"  bash scripts/download_tokenizer.sh /path/to/download/dir"
                )
                logging.error(error_msg)
                raise ConnectionError(error_msg) from e
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    
    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize a prompt with "|" separator into per-agent token sequences.
        
        Args:
            prompt: Full prompt string, e.g., "Pick up red block | Place blue cube"
                    Agents are separated by "|"
            state: Optional state array for pi0.5 discrete state input.
                   When provided, each agent's prompt includes only its own state slice:
                   "Task: {agent_prompt}, State: {agent_state_str};\nAction: "
        
        Returns:
            tokens: Array of shape [num_agents * max_len_per_agent]
            masks: Boolean mask array of same shape
        """
        parts = prompt.split("|")
        parts = [p.strip() for p in parts]
        
        while len(parts) < self._num_agents:
            parts.append(parts[-1] if parts else "")
        
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        
        all_tokens = []
        all_masks = []
        
        for agent_id in range(self._num_agents):
            agent_prompt = parts[agent_id] if agent_id < len(parts) else parts[-1]
            cleaned_text = agent_prompt.strip().replace("_", " ").replace("\n", " ")
            
            if state is not None:
                start = agent_id * self._action_dim_per_agent
                end = start + self._action_dim_per_agent
                agent_state = state[start:end]
                discretized = np.digitize(agent_state, bins=bins) - 1
                agent_state_str = " ".join(map(str, discretized))
                full_prompt = f"Task: {cleaned_text}, State: {agent_state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=(agent_id == 0))
            elif agent_id == 0:
                tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
            else:
                tokens = self._tokenizer.encode(cleaned_text) + self._tokenizer.encode("\n")
            
            tokens_len = len(tokens)
            
            if tokens_len < self._max_len_per_agent:
                padding = [False] * (self._max_len_per_agent - tokens_len)
                mask = [True] * tokens_len + padding
                tokens = tokens + padding
            else:
                if tokens_len > self._max_len_per_agent:
                    logging.warning(
                        f"Agent {agent_id} token length ({tokens_len}) exceeds max length per agent "
                        f"({self._max_len_per_agent}), truncating. "
                        "Consider increasing `max_token_len_per_agent` in your model config."
                    )
                tokens = tokens[:self._max_len_per_agent]
                mask = [True] * self._max_len_per_agent
            
            all_tokens.extend(tokens)
            all_masks.extend(mask)
        
        return np.asarray(all_tokens), np.asarray(all_masks)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        # Check for local tokenizer path first (for offline use)
        local_tokenizer_path = os.getenv("PALIGEMMA_TOKENIZER_PATH")
        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            path = pathlib.Path(local_tokenizer_path)
            logging.info(f"Using local tokenizer from: {path}")
        else:
            try:
                path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            except Exception as e:
                error_msg = (
                    f"Failed to download tokenizer. Error: {e}\n"
                    f"To use a local tokenizer, set environment variable:\n"
                    f"  export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model\n"
                    f"Or download manually using:\n"
                    f"  bash scripts/download_tokenizer.sh /path/to/download/dir"
                )
                logging.error(error_msg)
                raise ConnectionError(error_msg) from e
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        # Check for local tokenizer path first (for offline use)
        local_tokenizer_path = os.getenv("PALIGEMMA_TOKENIZER_PATH")
        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            path = pathlib.Path(local_tokenizer_path)
            logging.info(f"Using local tokenizer from: {path}")
        else:
            try:
                path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            except Exception as e:
                error_msg = (
                    f"Failed to download tokenizer. Error: {e}\n"
                    f"To use a local tokenizer, set environment variable:\n"
                    f"  export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model\n"
                    f"Or download manually using:\n"
                    f"  bash scripts/download_tokenizer.sh /path/to/download/dir"
                )
                logging.error(error_msg)
                raise ConnectionError(error_msg) from e
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        # Check for local tokenizer path first (for offline use)
        local_tokenizer_path = os.getenv("PALIGEMMA_TOKENIZER_PATH")
        if local_tokenizer_path and os.path.exists(local_tokenizer_path):
            path = pathlib.Path(local_tokenizer_path)
            logging.info(f"Using local tokenizer from: {path}")
        else:
            try:
                path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            except Exception as e:
                error_msg = (
                    f"Failed to download tokenizer. Error: {e}\n"
                    f"To use a local tokenizer, set environment variable:\n"
                    f"  export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model\n"
                    f"Or download manually using:\n"
                    f"  bash scripts/download_tokenizer.sh /path/to/download/dir"
                )
                logging.error(error_msg)
                raise ConnectionError(error_msg) from e
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
