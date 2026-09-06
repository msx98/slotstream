"""Turn a `SLOTSTREAM_ROUTER_TRACE` stream into the npz `cachesim.py` reads.

The engine writes, per MoE call, `layer`/`tokens`/`topK` as little-endian int32
followed by `tokens * topK` int16 expert ids. Layers are visited in order within
a pass, so passes reassemble without a pass id.

Only single-token calls are kept by default, and that is the point rather than
convenience: a prefill pass of 256 tokens or more takes the sweep, which never
reads or writes the slot pool, so its routing is not cache traffic. Simulating
it would report a hit rate for accesses the pool never sees. `--all` keeps
everything for anyone measuring routing itself rather than the cache. (The DS4
prefill also never touches the pool, so the same default is right there.)

usage: trace_convert.py <trace.bin> <out.npz> [--all] [--name decode] [--layers N]

The layer count for the completeness check comes from the stream (a pass
visits every layer in order, so max(layer)+1) unless pinned with --layers.
"""
import sys
import numpy as np

N_LAYERS = 48


def read_stream(path):
    raw = np.fromfile(path, dtype=np.uint8)
    out = []
    off = 0
    while off < raw.size:
        layer, tokens, topk = raw[off:off + 12].view("<i4")
        off += 12
        n = int(tokens) * int(topk)
        ids = raw[off:off + n * 2].view("<i2").astype(np.int16)
        off += n * 2
        out.append((int(layer), int(tokens), int(topk), ids.reshape(int(tokens), int(topk))))
    return out


def main():
    global N_LAYERS
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_all = "--all" in sys.argv
    name = "decode"
    n_layers = None
    for a in sys.argv[1:]:
        if a.startswith("--name"):
            name = a.split("=", 1)[1] if "=" in a else name
        elif a.startswith("--layers"):
            n_layers = int(a.split("=", 1)[1])
    src, dst = args[0], args[1]

    calls = read_stream(src)
    kept = [c for c in calls if keep_all or c[1] == 1]
    if not kept:
        print("no calls matched; use --all to include multi-token passes")
        sys.exit(1)
    topk = kept[0][2]
    N_LAYERS = n_layers if n_layers is not None else max(c[0] for c in calls) + 1

    # Group into steps: a step is one visit to every layer, in layer order.
    steps, cur = [], {}
    for layer, tokens, _k, ids in kept:
        if layer in cur:
            steps.append(cur)
            cur = {}
        cur[layer] = ids
    if cur:
        steps.append(cur)

    complete = [s for s in steps if len(s) == N_LAYERS]
    dropped = len(steps) - len(complete)
    arr = np.zeros((len(complete), N_LAYERS, topk), dtype=np.int16)
    for i, s in enumerate(complete):
        for layer, ids in s.items():
            arr[i, layer] = ids[0] if ids.shape[0] == 1 else ids[0]

    np.savez_compressed(dst, **{name: arr})
    print(f"{len(calls)} calls -> {len(complete)} complete steps x {N_LAYERS} layers x {topk}"
          f"{f' ({dropped} partial step(s) dropped)' if dropped else ''} -> {dst}")


if __name__ == "__main__":
    main()
