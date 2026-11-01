"""Success of best-of-N vs N per policy group: python scripts/summarize_bon.py <bon out_dir> [out.json]."""
import glob, json, os, re, sys
import numpy as np

d = sys.argv[1]
rows = {}
for p in glob.glob(os.path.join(d, "*.json")):
    m = re.fullmatch(r"square_(ph|mh)_(\d+)demos_seed(\d+)_sde_bon(\d+)\.json", os.path.basename(p))
    if not m:
        continue
    ds, n, seed, N = m.groups()
    with open(p) as f:
        j = json.load(f)
    rows.setdefault((ds, int(n)), {}).setdefault(int(N), []).append((int(seed), j["success_rate"], j["mean_length"], j["episodes"]))
Ns = sorted({N for g in rows.values() for N in g})
print("success rate, mean ± std over policy seeds (n seeds); episodes per policy:",
      sorted({r[3] for g in rows.values() for v in g.values() for r in v}))
print("| group | " + " | ".join(f"N={N}" for N in Ns) + " |")
print("|---|" + "---|" * len(Ns))
out = {}
for key in sorted(rows, key=lambda k: (k[0] != "ph", k[1])):
    cells = []
    for N in Ns:
        v = rows[key].get(N, [])
        s = [r[1] for r in v]
        cells.append(f"{np.mean(s):.3f} ± {np.std(s):.3f} ({len(s)})" if s else "-")
        out[f"{key[0]}-{key[1]}/bon{N}"] = {"seeds": [r[0] for r in v], "success": s, "length": [r[2] for r in v]}
    print(f"| {key[0]}-{key[1]} | " + " | ".join(cells) + " |")
print("\nmean episode length")
for key in sorted(rows, key=lambda k: (k[0] != "ph", k[1])):
    print(f"{key[0]}-{key[1]}: " + "  ".join(f"N={N} {np.mean([r[2] for r in rows[key].get(N, [])]):.0f}" for N in Ns if rows[key].get(N)))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)
