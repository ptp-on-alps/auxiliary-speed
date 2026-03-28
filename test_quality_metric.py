"""Quick sanity check for the quantile coupling probability metric."""
import torch


def quantile_coupling_prob(p, q):
    cdf_p = p.cumsum(-1)
    cdf_q = q.cumsum(-1)
    zeros = torch.zeros(p.shape[0], 1)
    cdf_p_prev = torch.cat([zeros, cdf_p[:, :-1]], dim=-1)
    cdf_q_prev = torch.cat([zeros, cdf_q[:, :-1]], dim=-1)
    overlap = (torch.minimum(cdf_p, cdf_q) - torch.maximum(cdf_p_prev, cdf_q_prev)).clamp(min=0)
    return overlap.sum(-1)


def monte_carlo(p, q, n=1_000_000):
    """Empirical estimate via actual inverse-CDF sampling with the same u."""
    u = torch.rand(n)
    tok_p = (u.unsqueeze(1) > p.cumsum(-1)).sum(-1)
    tok_q = (u.unsqueeze(1) > q.cumsum(-1)).sum(-1)
    return (tok_p == tok_q).float().mean().item()


cases = [
    # (name, p, q, expected)
    ("identical",         [0.6, 0.4],       [0.6, 0.4],       1.0),
    ("two-token shift",   [0.6, 0.4],       [0.3, 0.7],        0.7),   # overlap: 0.3+0.4
    ("three-token",       [0.5, 0.3, 0.2],  [0.2, 0.3, 0.5],  0.4),   # 0.2+0+0.2
    ("disjoint",          [1.0, 0.0],       [0.0, 1.0],        0.0),
    ("one dominant",      [0.99, 0.01],     [0.98, 0.02],      0.99),  # ≈ min(0.99,0.98)+min(0.01,0.02)=0.99
]

print(f"{'Case':<22} {'expected':>10} {'analytic':>10} {'monte carlo':>12} {'ok?':>6}")
print("-" * 64)
all_ok = True
for name, pv, qv, expected in cases:
    p = torch.tensor([pv])
    q = torch.tensor([qv])
    analytic = quantile_coupling_prob(p, q).item()
    mc = monte_carlo(p, q)
    ok = abs(analytic - expected) < 1e-5 and abs(analytic - mc) < 0.003
    all_ok = all_ok and ok
    print(f"{name:<22} {expected:>10.4f} {analytic:>10.4f} {mc:>12.4f} {'yes' if ok else 'FAIL':>6}")

print()
print("All tests passed." if all_ok else "FAILURES detected.")
