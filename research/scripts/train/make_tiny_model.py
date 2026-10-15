"""Write a tiny random Qwen3-style causal LM + byte-level BPE tokenizer (every option code A-Z / a-z is one token) for
CPU pipeline checks: ``MODEL=<dir> DEVICE=cpu bash scripts/train/smoke_gpu.sh``. Same architecture family as the
tests' ``tiny`` fixture (tests/test_training_rl.py). Its outputs mean nothing; it only exercises the code paths.

  python scripts/train/make_tiny_model.py OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path


def make_tiny_model(out: str | Path, seed: int = 0) -> Path:
    import tokenizers
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<|im_end|>"],
                                             initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["STATE QUESTION OPTIONS Answer with the option label only. TASK HISTORY"] * 10, trainer)
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    ft.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                        "{% if add_generation_prompt %}assistant: {% endif %}")
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(ft), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
                      eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    ft.save_pretrained(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(make_tiny_model(sys.argv[1]))
