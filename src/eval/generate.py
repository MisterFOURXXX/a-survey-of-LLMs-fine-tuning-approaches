"""Generation helpers for evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts,
    max_new_tokens: int = 128,
    batch_size: int = 4,
    max_input_length: int = 512,
):
    """Deterministic greedy generation for a list of prompt strings.

    Uses left-padding (correct for decoder-only generation) and slices off
    the exact number of *unpadded* input tokens per row.
    """
    if len(prompts) == 0:
        return []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    device = next(model.parameters()).device

    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Per-row true input length (ignores padding tokens)
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        for j, inp_len in enumerate(input_lens):
            new_ids = gen[j][inp_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append(text.strip())

    return outputs