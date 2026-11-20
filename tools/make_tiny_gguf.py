#!/usr/bin/env python3
"""Write a tiny random-weight Llama GGUF (SentencePiece-style vocab) for smoke tests.

usage: PYTHONPATH=<llama.cpp>/gguf-py python3 make_tiny_gguf.py out.gguf [--layers 4 --embd 256]
"""
import argparse
import numpy as np
import gguf

WORDS = ("the of and to in is was that for on with as by at from system model data network energy "
         "river mountain history science music market computer language city token sequence layer "
         "attention cache prefill decode cluster node message window atomic counter").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=688)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0, help="weight RNG seed (same vocab for any seed)")
    ap.add_argument("--chat-template", action="store_true", help="embed a ChatML chat template")
    a = ap.parse_args()

    # vocab: specials + 256 byte fallbacks + word pieces
    toks, scores, types = [], [], []
    for t, ty in (("<unk>", gguf.TokenType.UNKNOWN), ("<s>", gguf.TokenType.CONTROL), ("</s>", gguf.TokenType.CONTROL)):
        toks.append(t); scores.append(0.0); types.append(ty)
    for b in range(256):
        toks.append(f"<0x{b:02X}>"); scores.append(0.0); types.append(gguf.TokenType.BYTE)
    pieces = ["▁" + w for w in WORDS] + list("abcdefghijklmnopqrstuvwxyz.") + ["▁"]
    for i, p in enumerate(pieces):
        toks.append(p); scores.append(-float(i)); types.append(gguf.TokenType.NORMAL)
    n_vocab = len(toks)

    w = gguf.GGUFWriter(a.out, "llama")
    w.add_name("tiny-random-llama")
    w.add_context_length(a.ctx)
    w.add_embedding_length(a.embd)
    w.add_block_count(a.layers)
    w.add_feed_forward_length(a.ff)
    w.add_head_count(a.heads)
    w.add_head_count_kv(a.kv_heads)
    w.add_rope_dimension_count(a.embd // a.heads)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(n_vocab)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)
    if a.chat_template:
        w.add_chat_template(
            "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    rng = np.random.default_rng(a.seed)
    hd = a.embd // a.heads
    def r(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)
    w.add_tensor("token_embd.weight", r(n_vocab, a.embd))
    for i in range(a.layers):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", np.ones(a.embd, np.float32))
        w.add_tensor(p + "attn_q.weight", r(a.embd, a.embd))
        w.add_tensor(p + "attn_k.weight", r(a.kv_heads * hd, a.embd))
        w.add_tensor(p + "attn_v.weight", r(a.kv_heads * hd, a.embd))
        w.add_tensor(p + "attn_output.weight", r(a.embd, a.embd))
        w.add_tensor(p + "ffn_norm.weight", np.ones(a.embd, np.float32))
        w.add_tensor(p + "ffn_gate.weight", r(a.ff, a.embd))
        w.add_tensor(p + "ffn_up.weight", r(a.ff, a.embd))
        w.add_tensor(p + "ffn_down.weight", r(a.embd, a.ff))
    w.add_tensor("output_norm.weight", np.ones(a.embd, np.float32))
    w.add_tensor("output.weight", r(n_vocab, a.embd))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: vocab={n_vocab} layers={a.layers} embd={a.embd}")


if __name__ == "__main__":
    main()
