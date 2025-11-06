import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from einops import rearrange
from tqdm import tqdm


@torch.no_grad()
def compute_text_kl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text1: str,
    text2: str,
    continuation: str | None = None,
    model_name: str | None = None,
    apply_template: bool = True,
    n_samples: int = 10,
    continuation_len: int = 50,
    device: str | torch.device | None = None,
) -> tuple[float, float]:
    """
    Compute KL divergence between two prompts using a language model.

    KL(text1 || text2) measures how much the probability distribution induced by text1
    differs from that induced by text2 when predicting continuations.

    This follows the same approach as compute_dataset_kl:
    1. Generate continuations from text1 (or use provided continuation)
    2. Measure how well text1 predicts those continuations (neg_log_p1)
    3. Measure how well text2 predicts those continuations (neg_log_p2)
    4. KL = neg_log_p1 - neg_log_p2

    Args:
        model: The language model to use for computing probabilities
        tokenizer: The tokenizer for the model
        text1: First prompt (reference distribution) - the raw prompt text
        text2: Second prompt (comparison distribution) - the raw prompt text
        continuation: Optional fixed continuation text to evaluate on.
                     If None, generates continuations from text1.
        model_name: Model name or path (used to determine prompt template).
                   If None, uses model.config.name_or_path
        apply_template: If True, wrap texts with model-specific prompt templates.
                       If False, use raw texts without templates.
        n_samples: Number of continuation samples to generate (ignored if continuation is provided)
        continuation_len: Length of each continuation in tokens (ignored if continuation is provided)
        device: Device to run computation on (defaults to model's device)

    Returns:
        Tuple of (kl, std) where:
        - kl: Mean KL divergence across samples
        - std: Standard error of the mean

    Example:
        >>> from func_from_evil_twins import load_model_tokenizer
        >>> model, tokenizer = load_model_tokenizer("gpt2")
        >>> kl, std = compute_text_kl(
        ...     model, tokenizer,
        ...     "Write a story about cats.",
        ...     "Write a story about dogs.",
        ...     apply_template=True
        ... )
    """
    # Import build_prompt here to avoid circular dependency
    from func_from_evil_twins import build_prompt

    if device is None:
        device = model.device

    # Determine model name for template lookup
    if model_name is None:
        model_name = model.config.name_or_path

    # Build prompts with templates if requested
    if apply_template:
        wrapped_tokens1, prompt_slice1 = build_prompt(
            model_name, text1, tokenizer, validate_prompt=True
        )
        wrapped_tokens2, prompt_slice2 = build_prompt(
            model_name, text2, tokenizer, validate_prompt=True
        )

        tokens1 = wrapped_tokens1.to(device)
        tokens2 = wrapped_tokens2.to(device)
    else:
        # Use raw texts without templates
        tokens1 = tokenizer(text1, return_tensors="pt").input_ids.to(device)
        tokens2 = tokenizer(text2, return_tensors="pt").input_ids.to(device)
        prompt_slice1 = slice(0, tokens1.shape[1])
        prompt_slice2 = slice(0, tokens2.shape[1])

    # Get or generate continuations
    if continuation is not None:
        # Use provided continuation
        cont_tokens = tokenizer(continuation, return_tensors="pt").input_ids.to(device)
        # Use same continuation for all samples
        continuations = cont_tokens.repeat(1, 1)  # (1, cont_len)
    else:
        # Generate continuations from text1 (similar to DocDataset._gen_docs)
        continuations = torch.zeros((n_samples, continuation_len), dtype=torch.long, device=device)

        for i in tqdm(range(n_samples), desc="Generating continuations", leave=False):
            cur_prompt = tokens1.clone()
            for j in range(continuation_len):
                cur_logits = model(cur_prompt).logits
                cur_logits = cur_logits[:, -1, :]

                # Handle models with logit softcapping (Gemma)
                if hasattr(model.config, "final_logit_softcapping") and model.config.final_logit_softcapping is not None:
                    cur_logits /= model.config.final_logit_softcapping
                    cur_logits.tanh_()
                    cur_logits *= model.config.final_logit_softcapping

                # Prevent EOS during generation
                if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
                    cur_logits[..., model.config.eos_token_id] = -float("inf")

                cur_probs = F.softmax(cur_logits, dim=-1)
                cur_tok = torch.multinomial(cur_probs, 1)
                continuations[i, j] = cur_tok[0, 0]
                cur_prompt = torch.cat([cur_prompt, cur_tok], dim=-1)

    # Now compute KL: for each continuation, compare neg_log_p under text1 vs text2
    kls = []

    for i in tqdm(range(continuations.shape[0]), desc="Computing KL divergence", leave=False):
        cont = continuations[i:i+1]  # (1, cont_len)

        # Concatenate prompt1 + continuation
        seq1 = torch.cat([tokens1, cont], dim=-1)
        # Concatenate prompt2 + continuation
        seq2 = torch.cat([tokens2, cont], dim=-1)

        # Compute neg_log_p for text1 predicting the continuation
        logits1 = model(seq1).logits
        pred_slice1 = slice(tokens1.shape[1] - 1, seq1.shape[1] - 1)
        target_slice1 = slice(tokens1.shape[1], seq1.shape[1])

        neg_log_p1 = F.cross_entropy(
            rearrange(logits1[:, pred_slice1, :], "b k v -> b v k"),
            seq1[:, target_slice1],
            reduction="sum"
        ).item()

        # Compute neg_log_p for text2 predicting the continuation
        logits2 = model(seq2).logits
        pred_slice2 = slice(tokens2.shape[1] - 1, seq2.shape[1] - 1)
        target_slice2 = slice(tokens2.shape[1], seq2.shape[1])

        neg_log_p2 = F.cross_entropy(
            rearrange(logits2[:, pred_slice2, :], "b k v -> b v k"),
            seq2[:, target_slice2],
            reduction="sum"
        ).item()

        # KL for this continuation
        kl_i = neg_log_p1 - neg_log_p2
        kls.append(kl_i)

    # Compute mean and standard error
    kls_tensor = torch.tensor(kls)
    kl_mean = kls_tensor.mean().item()
    kl_std = kls_tensor.std().item() / (len(kls) ** 0.5)

    return kl_mean, kl_std
