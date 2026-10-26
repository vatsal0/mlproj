"""Synthetic retrieval tasks, generated on the fly on the training device.

Vocabulary layout (fixed across all runs so context length is the only thing that changes):
    [0, N_FILLER)                         filler tokens
    [N_FILLER, N_FILLER + N_KEYS)         keys
    [N_FILLER + N_KEYS, VOCAB_SIZE)       values

Both tasks end with a block of queries. Each query is a (key, value) pair, and the model is
scored on predicting the value from the key, which requires retrieving the matching pair from
the context (an induction-style lookup).

    mqar:   [k1 v1 k2 v2 ... kn vn | queries for all n keys]   n = T // 4
            more context = more items to track
    needle: [filler ... k1 v1 ... filler ... k16 v16 ... filler | queries for the 16 keys]
            more context = more filler, the retrieval problem itself stays fixed;
            at T = 64 (no filler) this is identical to mqar
"""
import torch

N_FILLER, N_KEYS, N_VALUES = 256, 1024, 512
KEY_OFF, VAL_OFF = N_FILLER, N_FILLER + N_KEYS
VOCAB_SIZE = N_FILLER + N_KEYS + N_VALUES
N_NEEDLES = 16


def num_queries(task, T, n_needles=N_NEEDLES):
    return T // 4 if task == "mqar" else n_needles


def _rand(shape, g, device):
    return torch.rand(shape, generator=g, device=device)


def _randint(high, shape, g, device):
    return torch.randint(high, shape, generator=g, device=device)


def _queries(keys, vals, g):
    """Shuffle the pairs into a query block. Returns query tokens and the permutation used."""
    B, n = keys.shape
    perm = _rand((B, n), g, keys.device).argsort(-1)
    q = torch.stack([keys.gather(1, perm), vals.gather(1, perm)], -1).view(B, 2 * n)
    return q, perm


def make_batch(task, B, T, g, device, n_needles=N_NEEDLES):
    """Returns a dict with
        x      (B, T)   token ids
        qpos   (B, n)   positions of query keys; the logit there should predict the value
        ans    (B, n)   correct value token for each query
        src    (B, n)   position of the matching value in the context (ground-truth attention target)
        depth  (B, n)   relative position of the matching pair in the context, in [0, 1)
    """
    assert T % 4 == 0
    n = num_queries(task, T, n_needles)
    keys = _rand((B, N_KEYS), g, device).argsort(-1)[:, :n] + KEY_OFF  # distinct keys per sequence
    vals = _randint(N_VALUES, (B, n), g, device) + VAL_OFF
    C = T - 2 * n  # context length before the query block

    if task == "mqar":
        ctx = torch.stack([keys, vals], -1).view(B, 2 * n)
        starts = torch.arange(n, device=device).expand(B, n) * 2
    elif task == "needle":
        n_fill = C - 2 * n
        assert n_fill >= 0, f"T={T} too short for {n} needles"
        ctx = _randint(N_FILLER, (B, C), g, device)
        # sorted offsets into the filler, shifted so the pairs never overlap
        u = _randint(n_fill + 1, (B, n), g, device).sort(-1).values
        starts = u + 2 * torch.arange(n, device=device)
        ctx.scatter_(1, starts, keys)
        ctx.scatter_(1, starts + 1, vals)
    else:
        raise ValueError(task)

    q, perm = _queries(keys, vals, g)
    x = torch.cat([ctx, q], 1)
    qpos = C + 2 * torch.arange(n, device=device).expand(B, n)
    src_start = starts.gather(1, perm)
    return dict(x=x, qpos=qpos, ans=vals.gather(1, perm), src=src_start + 1, depth=src_start.float() / C)
