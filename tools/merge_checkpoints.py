#!/usr/bin/env python
"""Merge needle checkpoints by weighted parameter averaging.

Needle finetuning writes full checkpoints (`{"params": <pytree of fp16>, "config": dict}`),
not adapters, so "merging" here is weight-space interpolation:

  merged = sum_i w_i * params_i        (weights renormalised to sum to 1)

Two common shapes:

  WiSE-FT / anti-forgetting soup — pull a finetuned model back toward base:
      merge_checkpoints.py out.pkl --base checkpoints/needle.pkl \
          --ft checkpoints/needle_finetuned_..._best.pkl --alpha 0.7
      (alpha = weight on the finetuned model; 1.0 == the finetuned model unchanged)

  Uniform / weighted soup over N runs:
      merge_checkpoints.py out.pkl --ckpt a.pkl --ckpt b.pkl:2 --ckpt c.pkl

Optionally scores the result on the SAME held-out test split finetune.py uses
(`_per_tool_split`, seed 42), so numbers are comparable with the BASE_EVAL /
FT_EVAL lines a finetune run prints.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="Output checkpoint path (.pkl)")
    ap.add_argument("--ckpt", action="append", default=[], metavar="PATH[:WEIGHT]",
                    help="Input checkpoint, optional :weight suffix (default 1.0). Repeatable.")
    ap.add_argument("--base", help="Base checkpoint (use with --ft/--alpha)")
    ap.add_argument("--ft", help="Finetuned checkpoint (use with --base/--alpha)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Weight on --ft; base gets (1-alpha). Default 0.7")
    ap.add_argument("--eval", dest="eval_jsonl",
                    help="Score merged model on the held-out test split of this needle-format JSONL")
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="Which held-out split to score on. Select alpha on 'val', "
                         "report the final number on 'test' (default: test)")
    ap.add_argument("--eval-inputs", action="store_true",
                    help="Also score each input checkpoint on the same test split")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU (use while a GPU training run holds the card)")
    ap.add_argument("--allow-config-diff", action="store_true",
                    help="Proceed when inputs disagree on a shared config value (keeps the first)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return ap.parse_args()


def resolve_inputs(args):
    """-> [(path, weight)] with weights renormalised to sum to 1."""
    if args.base or args.ft:
        if not (args.base and args.ft):
            sys.exit("--base and --ft must be given together")
        if args.ckpt:
            sys.exit("--ckpt cannot be combined with --base/--ft")
        if not 0.0 <= args.alpha <= 1.0:
            sys.exit(f"--alpha must be in [0, 1], got {args.alpha}")
        pairs = [(args.base, 1.0 - args.alpha), (args.ft, args.alpha)]
    else:
        if len(args.ckpt) < 2:
            sys.exit("need at least two --ckpt entries (or --base/--ft)")
        pairs = []
        for spec in args.ckpt:
            path, _, w = spec.rpartition(":")
            # a bare Windows-free path has no ':' -> rpartition puts it in `w`
            if not path:
                path, w = w, "1.0"
            try:
                weight = float(w)
            except ValueError:
                sys.exit(f"bad weight in --ckpt {spec!r}")
            pairs.append((path, weight))

    for path, _ in pairs:
        if not os.path.isfile(path):
            sys.exit(f"checkpoint not found: {path}")

    total = sum(w for _, w in pairs)
    if total <= 0:
        sys.exit(f"weights must sum to a positive number (got {total})")
    return [(p, w / total) for p, w in pairs]


def load_raw(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def check_compatible(loaded, allow_config_diff=False):
    """All checkpoints must share an identical param tree; configs are unioned.

    Checkpoints written by different needle versions carry different config KEYS
    (e.g. `enable_speech`/`n_mels` exist in the HF base but not in older finetunes).
    A key present in only one checkpoint is taken as-is; a key both define with
    conflicting values is a hard error unless --allow-config-diff.
    """
    import jax

    (ref_path, ref) = loaded[0]
    ref_leaves, ref_tree = jax.tree.flatten(ref["params"])

    config = dict(ref["config"])
    added = {}
    conflicts = {}

    for path, data in loaded[1:]:
        cfg = data["config"]
        for k, v in cfg.items():
            if k not in config:
                config[k] = v
                added[k] = (path, v)
            elif config[k] != v:
                conflicts[k] = (config[k], v, path)

        leaves, tree = jax.tree.flatten(data["params"])
        if tree != ref_tree:
            sys.exit(f"param tree structure differs between {ref_path} and {path}")
        for i, (a, b) in enumerate(zip(ref_leaves, leaves)):
            if a.shape != b.shape:
                sys.exit(f"shape mismatch at leaf {i}: {a.shape} ({ref_path}) vs {b.shape} ({path})")

    only_in_ref = {k for k in ref["config"]} - set.intersection(
        *[set(d["config"]) for _, d in loaded]) if len(loaded) > 1 else set()
    if only_in_ref or added:
        keys = sorted(only_in_ref | set(added))
        print(f"Config keys not shared by all inputs (kept, differing needle versions): {keys}")
    if conflicts:
        msg = "; ".join(f"{k}: {a!r} vs {b!r} (from {p})" for k, (a, b, p) in conflicts.items())
        if not allow_config_diff:
            sys.exit(f"conflicting config values — refusing to merge: {msg}\n"
                     f"(pass --allow-config-diff to keep the first checkpoint's values)")
        print(f"WARNING: conflicting config values, keeping first checkpoint's: {msg}")

    return config


def merge(pairs, allow_config_diff=False):
    import jax
    import numpy as np

    loaded = [(p, load_raw(p)) for p, _ in pairs]
    config = check_compatible(loaded, allow_config_diff)

    n_params = sum(x.size for x in jax.tree.leaves(loaded[0][1]["params"]))
    print(f"Merging {len(loaded)} checkpoints, {n_params:,} params each")
    for (path, _), (_, w) in zip(loaded, pairs):
        print(f"  {w:6.3f}  {path}")

    # Accumulate in float32 so fp16 rounding does not compound across inputs.
    acc = jax.tree.map(lambda x: np.asarray(x, dtype=np.float32) * pairs[0][1],
                       loaded[0][1]["params"])
    for (_, data), (_, w) in zip(loaded[1:], pairs[1:]):
        acc = jax.tree.map(lambda a, b: a + np.asarray(b, dtype=np.float32) * w,
                           acc, data["params"])

    merged = jax.tree.map(lambda x: np.asarray(x, dtype=np.float16), acc)

    # Drift from the first input — a sanity signal that the merge did something.
    ref = jax.tree.leaves(loaded[0][1]["params"])
    out = jax.tree.leaves(merged)
    num = sum(float(np.sum((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
              for a, b in zip(ref, out))
    den = sum(float(np.sum(np.asarray(a, np.float32) ** 2)) for a in ref)
    print(f"Relative L2 drift from {loaded[0][0]}: {(num / den) ** 0.5:.4%}")

    return merged, config


def score(path_or_params, config, test_examples, label):
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.run import load_checkpoint
    from needle.dataset.tokenizer import get_tokenizer
    from needle.training.finetune import _quick_tool_eval

    if isinstance(path_or_params, str):
        params, cfg = load_checkpoint(path_or_params)
    else:
        import jax.numpy as jnp
        import jax
        params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.bfloat16), path_or_params)
        from needle.model.run import TransformerConfig
        cfg = TransformerConfig(**config)

    m = _quick_tool_eval(SimpleAttentionNetwork(cfg), params, get_tokenizer(), test_examples)
    if not m:
        print(f"  {label}: no scorable examples")
        return {}
    print(f"  {label}: call_f1={m['call_f1']:.1%}  exact={m['exact_match']:.1%}  "
          f"name_f1={m.get('name_f1', float('nan')):.1%}")
    return m


def main():
    args = parse_args()
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    pairs = resolve_inputs(args)

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"refusing to overwrite {args.out} (pass --force)")

    merged, config = merge(pairs, args.allow_config_diff)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"params": merged, "config": config}, f)
    os.replace(tmp, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if args.eval_jsonl:
        from needle.training.finetune import _per_tool_split
        with open(args.eval_jsonl) as f:
            examples = [json.loads(line) for line in f if line.strip()]
        _, val_examples, test_examples = _per_tool_split(examples)
        test_examples = val_examples if args.split == "val" else test_examples
        scorable = [e for e in test_examples if e.get("answers", "").strip() not in ("", "[]")]
        print(f"\n{args.split} split: {len(test_examples)} examples "
              f"({len(scorable)} scorable; empty-answer negatives are skipped by needle's eval)")
        if args.eval_inputs:
            for path, _ in pairs:
                score(path, config, test_examples, os.path.basename(path))
        score(merged, config, test_examples, f"MERGED {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
