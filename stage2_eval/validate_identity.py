"""Does the identity instrument work? Test it before trusting it.

    python -m stage2_eval.validate_identity                 # all 1000 LFW test pairs
    python -m stage2_eval.validate_identity --limit 40      # smoke test

LFW's standard test split: 500 same-person pairs and 500 different-person pairs of
real photos. For each recognizer: how far apart the two cosine distributions sit (AUC),
the threshold where false accepts equal false rejects (EER), and the true-accept rate
at 1% false accepts. Then: do the two recognizers agree? If they don't, a method
tuned against one will look good on one number and bad on the other.

Writes identity_calibration.json (thresholds used by score.py and later the on-device
identity guard) and identity_validation.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, roc_curve

from .identity import RECOGNIZERS, Recognizer, cosine


def stats(cos, same):
    ok = ~np.isnan(cos)
    c, y = cos[ok], same[ok]
    fpr, tpr, thr = roc_curve(y, c)
    eer_i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    far1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    return {
        "pairs_scored": int(ok.sum()), "pairs_no_face": int((~ok).sum()),
        "genuine_mean": float(c[y == 1].mean()), "genuine_std": float(c[y == 1].std()),
        "impostor_mean": float(c[y == 0].mean()), "impostor_std": float(c[y == 0].std()),
        "auc": float(roc_auc_score(y, c)),
        "eer": float((fpr[eer_i] + 1 - tpr[eer_i]) / 2), "threshold_eer": float(thr[eer_i]),
        "tar_at_far_1pct": float(far1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only N pairs (half same, half different)")
    ap.add_argument("--data", default="data/sklearn")
    ap.add_argument("--out", default="outputs/stage2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, data_home=args.data)
    pairs, same = lfw.pairs, lfw.target.astype(int)
    if args.limit:
        idx = np.r_[np.where(same == 1)[0][: args.limit // 2], np.where(same == 0)[0][: args.limit // 2]]
        pairs, same = pairs[idx], same[idx]
    if pairs.max() <= 1.0:
        pairs = pairs * 255
    pairs = pairs.astype(np.uint8)
    print(f"{len(pairs)} pairs ({same.sum()} same-person), images {pairs.shape[2]}x{pairs.shape[3]}")

    cos, report = {}, {}
    for role, pack in RECOGNIZERS.items():
        rec = Recognizer(pack)
        cos[role] = np.array([cosine(rec.embed(a), rec.embed(b)) for a, b in pairs])
        report[role] = {"pack": pack, **stats(cos[role], same)}
        r = report[role]
        print(f"{role:<10}{pack:<12} AUC {r['auc']:.4f}  EER {r['eer']:.3f} @ cos {r['threshold_eer']:.3f}  "
              f"TAR@FAR1% {r['tar_at_far_1pct']:.3f}  genuine {r['genuine_mean']:.2f}±{r['genuine_std']:.2f}  "
              f"impostor {r['impostor_mean']:.2f}±{r['impostor_std']:.2f}  no-face {r['pairs_no_face']}")

    both = ~np.isnan(cos["primary"]) & ~np.isnan(cos["secondary"])
    rho = spearmanr(cos["primary"][both], cos["secondary"][both]).statistic
    agree = np.mean((cos["primary"][both] >= report["primary"]["threshold_eer"]) ==
                    (cos["secondary"][both] >= report["secondary"]["threshold_eer"]))
    report["agreement"] = {"spearman": float(rho), "same_decision_rate": float(agree)}
    print(f"agreement: spearman {rho:.3f}, same accept/reject decision on {agree:.1%} of pairs")
    (out / "identity_calibration.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, role in zip(axes, RECOGNIZERS):
        c = cos[role]
        ax.hist(c[same == 1], bins=40, alpha=0.6, label="same person", color="#2e8b57")
        ax.hist(c[same == 0], bins=40, alpha=0.6, label="different people", color="#c0392b")
        ax.axvline(report[role]["threshold_eer"], color="k", ls="--", lw=1, label="EER threshold")
        ax.set_title(f"{role}: {RECOGNIZERS[role]}  AUC {report[role]['auc']:.3f}")
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
    axes[2].scatter(cos["primary"][both], cos["secondary"][both], s=5, c=same[both], cmap="RdYlGn")
    axes[2].set_xlabel("primary cosine")
    axes[2].set_ylabel("secondary cosine")
    axes[2].set_title(f"agreement  spearman {rho:.2f}")
    fig.tight_layout()
    fig.savefig(out / "identity_validation.png", dpi=110)
    print(f"-> {out}/identity_calibration.json, identity_validation.png")


if __name__ == "__main__":
    main()
