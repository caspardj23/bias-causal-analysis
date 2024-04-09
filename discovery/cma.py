import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import logging

import transformer_lens
import transformer_lens.utils as utils

from configuration.cma import CMAConfig

from functools import partial
from jaxtyping import Float, Int

from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, ActivationCache
import transformers

# transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES = transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES + ["yhavinga/gpt2-medium-dutch"]

transformer_lens.loading.OFFICIAL_MODEL_NAMES = (
    transformer_lens.loading.OFFICIAL_MODEL_NAMES + ["yhavinga/gpt2-medium-dutch"]
)


transformer_lens.loading.OFFICIAL_MODEL_NAMES = (
    transformer_lens.loading.OFFICIAL_MODEL_NAMES + ["GroNLP/gpt2-small-dutch"]
)


LOGGER = logging.getLogger(__name__)


def attention_intervention_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    counterfactual_cache: ActivationCache,
    mask: torch.Tensor,
    tail_indices: torch.Tensor,
    cf_tail_indices: torch.Tensor,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """
    Args:
        value: The attention result.
        hook: The hook point.
        counterfactual_cache: The counterfactual cache.
        mask: The mask.
        end_indices: The end indices of the sequences.
    Returns:
        The intervened attention result.
    """
    b, p, h, d = value.shape
    tail_indices = (
        tail_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, d)
    )
    cf_tail_indices = (
        cf_tail_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, d)
    )
    counterfactual_value = counterfactual_cache[hook.name]

    v_select = torch.gather(value, 1, tail_indices)
    cf_select = torch.gather(counterfactual_value, 1, cf_tail_indices)
    mask = mask.repeat(b, 1)
    mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, d)

    intervention = (1 - mask) * v_select + mask * cf_select
    return torch.scatter(value, dim=1, index=tail_indices, src=intervention)


class CMA:
    def __init__(self, config: CMAConfig, device):
        self.config = config
        self.device = device

        """GPT2 Small Original"""
        # self.model = HookedTransformer.from_pretrained(config.model, device=device)
        # self.model.cfg.use_attn_result = True
        # self.she_token = self.model.tokenizer.encode(" she")[0]
        # self.he_token = self.model.tokenizer.encode(" he")[0]
        # print("GPT2-small d_model: ", self.model.cfg)

        """GPT2 Small Dutch GroNLP"""
        self.model = HookedTransformer.from_pretrained(
            "GroNLP/gpt2-small-dutch", device=device
        )
        self.model.cfg.use_attn_result = True
        self.she_token = self.model.tokenizer.encode(" zij")[0]
        self.he_token = self.model.tokenizer.encode(" hij")[0]

        self.model.cfg.tokenizer_prepends_bos = False
        self.model.cfg.d_vocab = 50257
        self.model.cfg.d_vocab_out = 50257
        print("GPT2-small-dutch Dutch GroNLP config: ", self.model.cfg)

        """GPT2 Medium Dutch Havinga"""
        # self.model = transformer_lens.HookedTransformer.from_pretrained(
        #     "yhavinga/gpt2-medium-dutch", device=device
        # )
        # self.model.cfg.use_attn_result = True
        # self.she_token = self.model.tokenizer.encode(" zij")[0]
        # self.he_token = self.model.tokenizer.encode(" hij")[0]
        # print("GPT2-small-dutch Dutch GroNLP config: ", self.model.cfg)

        # Check configs for differrent GPT2 models
        # print("GPT2-small-dutch GroNLP config: ", self.model.cfg)
        # print("GPT2-small-dutch GroNLP d_model: ", self.model.cfg.d_model)
        # print("GPT2-small-dutch GroNLP d_vocab: ", self.model.cfg.d_vocab)
        # print("GPT2-small-dutch GroNLP n_heads: ", self.model.cfg.n_heads)
        # gpt2nl = transformer_lens.HookedTransformer.from_pretrained(
        #     "yhavinga/gpt2-medium-dutch"
        # )
        # gpt2nl.cfg.use_attn_result = True
        # print("GPT2-medium-dutch Havinga config: ", gpt2nl.cfg)
        # print("GPT2-medium-dutch Havinga d_model: ", gpt2nl.cfg.d_model)
        # print("GPT2-medium-dutch Havinga d_vocab: ", gpt2nl.cfg.d_vocab)
        # print("GPT2-medium-dutch Havinga n_heads: ", gpt2nl.cfg.n_heads)
        # gpt2 = transformer_lens.HookedTransformer.from_pretrained(config.model)
        # gpt2.cfg.use_attn_result = True

        # print("GPT2-small d_model: ", gpt2.cfg)
        # print("GPT2-small d_model: ", gpt2.cfg.d_model)
        # print("GPT2-small d_vocab: ", gpt2.cfg.d_vocab)
        # print("GPT2-small n_heads: ", gpt2.cfg.n_heads)

    def indirect_effects(self, dataloader):
        self.model.eval()
        effects = np.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                mask = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))
                mask[layer, head] = 1
                effects[layer, head] = self.indirect_effect(dataloader, mask)
                print(
                    "Self-indirect effect -- effects[layer, head]: ",
                    effects[layer, head],
                )
        return effects

    def indirect_effect(self, dataloader, mask):
        """
        Args:
            dataloader: A dataloader.
            mask: A mask to select a set of heads
        Returns:
            The indirect effect.
        """
        mask = mask.to(self.device)
        LOGGER.info(mask)
        nie = []
        for batch in dataloader:
            print("batch: ", batch)
            originals, counterfactuals, y = batch
            y = y.to(self.device)
            o_logits, i_logits = self._intervene(originals, counterfactuals, mask)
            o_probs = torch.softmax(o_logits, dim=-1)
            i_probs = torch.softmax(i_logits, dim=-1)
            print("o_probs: ", o_probs)
            print("i_probs: ", i_probs)
            o_probs_he = o_probs[:, self.he_token].squeeze()
            o_probs_she = o_probs[:, self.she_token].squeeze()
            i_probs_he = i_probs[:, self.he_token].squeeze()
            i_probs_she = i_probs[:, self.she_token].squeeze()
            print("o_probs_he: ", o_probs_he)
            print("o_probs_she: ", o_probs_she)
            print("i_probs_he: ", i_probs_he)
            print("i_probs_she: ", i_probs_she)

            o_anti_probs = torch.where(y == 1, o_probs_she, o_probs_he)
            i_anti_probs = torch.where(y == 1, i_probs_she, i_probs_he)

            o_pro_probs = torch.where(y == 1, o_probs_he, o_probs_she)
            i_pro_probs = torch.where(y == 1, i_probs_he, i_probs_she)

            nie_batch = (i_anti_probs / i_pro_probs) / (o_anti_probs / o_pro_probs) - 1

            nie.extend(nie_batch.cpu().tolist())
        LOGGER.info(np.mean(nie))
        return np.mean(nie)

    def _intervene(self, originals, counterfactuals, mask):
        """
        Args:
            batch: A batch of original and counterfactual sequences.
            mask: A batch of masks.
        Returns:
            The logits of the original and counterfactual sequences.
        """
        cf_tokens = self.model.to_tokens(counterfactuals).to(self.device)
        (
            cf_logits,
            counterfactual_cache,
        ) = self.model.run_with_cache(cf_tokens)
        cf_tail_indices = self._tail_indices(
            cf_tokens, self.model.tokenizer.eos_token_id
        )

        o_tokens = self.model.to_tokens(originals).to(self.device)
        tail_indices = self._tail_indices(o_tokens, self.model.tokenizer.eos_token_id)
        o_logits = self.model(o_tokens, return_type="logits")
        i_logits = self.model.run_with_hooks(
            o_tokens,
            return_type="logits",
            fwd_hooks=[
                (
                    f"blocks.{i}.attn.hook_result",
                    partial(
                        attention_intervention_hook,
                        counterfactual_cache=counterfactual_cache,
                        mask=mask[i, :].squeeze(),
                        tail_indices=tail_indices,
                        cf_tail_indices=cf_tail_indices,
                    ),
                )
                for i in range(self.model.cfg.n_layers)
            ],
        )
        tail_indices = (
            tail_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, o_logits.shape[2])
        )
        o_logits = torch.gather(o_logits, 1, tail_indices).squeeze()
        i_logits = torch.gather(i_logits, 1, tail_indices).squeeze()
        return o_logits, i_logits

    def _tail_indices(self, tokens, eos_token_id):
        """
        Args:
            tokens: A batch of token sequences.
            eos_token_id: The id of the end-of-sequence token.
        Returns:
            The index of the last true token in each sequence.
        """
        _, length = tokens.shape
        eos_mask = torch.where(tokens == eos_token_id, 1.0, 0.0)
        tail_indices = torch.argmax(eos_mask[:, 1:], dim=1)
        tail_indices = torch.where(tail_indices == 0, length - 1, tail_indices)
        return tail_indices
