import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

os.makedirs("results/analysis", exist_ok=True)

files = {
    "Random 1-pixel": "results/random_100.csv",
    "DE 1-pixel": "results/de_100.csv",
    "DE 3-pixel": "results/de_3pixel_100.csv",
    "DE 5-pixel": "results/de_5pixel_100.csv",
}

dfs = {name: pd.read_csv(path) for name, path in files.items()}

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ---------------------------
# 1. OVERALL SUMMARY
# ---------------------------

def wilson(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / denom
    margin = (
        z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
        / denom
    )
    return centre - margin, centre + margin

summary = []

for name, df in dfs.items():
    n = len(df)
    successes = int(df["success"].sum())
    rate = successes / n
    low, high = wilson(successes, n)

    successful = df[df["success"]]

    summary.append({
        "attack": name,
        "n": n,
        "successes": successes,
        "ASR_percent": rate * 100,
        "CI95_low_percent": low * 100,
        "CI95_high_percent": high * 100,
        "mean_cdiff_all": df["cdiff"].mean(),
        "median_cdiff_all": df["cdiff"].median(),
        "mean_cdiff_successful": (
            successful["cdiff"].mean()
            if len(successful) else np.nan
        )
    })

summary = pd.DataFrame(summary)

print("\n================ OVERALL SUMMARY ================\n")
print(summary.to_string(index=False))

summary.to_csv(
    "results/analysis/summary_metrics.csv",
    index=False
)

# ---------------------------
# 2. PAIRED COMPARISON
# ---------------------------

def paired_comparison(a, b, a_name, b_name):
    m = a.merge(
        b,
        on="image",
        suffixes=("_a", "_b")
    )

    a_only = (
        (m["success_a"] == True) &
        (m["success_b"] == False)
    ).sum()

    b_only = (
        (m["success_a"] == False) &
        (m["success_b"] == True)
    ).sum()

    both = (
        (m["success_a"] == True) &
        (m["success_b"] == True)
    ).sum()

    neither = (
        (m["success_a"] == False) &
        (m["success_b"] == False)
    ).sum()

    discordant = a_only + b_only

    p = (
        binomtest(a_only, discordant, 0.5).pvalue
        if discordant else 1.0
    )

    print(f"\n================ {a_name} vs {b_name} ================\n")
    print("Matched images:", len(m))
    print(f"{a_name} only:", a_only)
    print(f"{b_name} only:", b_only)
    print("Both succeeded:", both)
    print("Neither succeeded:", neither)
    print("Exact McNemar p-value:", p)

    return {
        "comparison": f"{a_name} vs {b_name}",
        f"{a_name}_only": a_only,
        f"{b_name}_only": b_only,
        "both": both,
        "neither": neither,
        "p_value": p
    }

comparisons = []

comparisons.append(
    paired_comparison(
        dfs["DE 1-pixel"],
        dfs["Random 1-pixel"],
        "DE1",
        "Random1"
    )
)

comparisons.append(
    paired_comparison(
        dfs["DE 3-pixel"],
        dfs["DE 5-pixel"],
        "DE3",
        "DE5"
    )
)

pd.DataFrame(comparisons).to_csv(
    "results/analysis/pairwise_comparisons.csv",
    index=False
)

# ---------------------------
# 3. CLASS-LEVEL ANALYSIS
# ---------------------------

class_rows = []

for attack_name, df in dfs.items():

    for class_id in sorted(df["true"].unique()):
        d = df[df["true"] == class_id]

        class_rows.append({
            "attack": attack_name,
            "class_id": class_id,
            "class_name": classes[int(class_id)],
            "n": len(d),
            "successful": int(d["success"].sum()),
            "ASR_percent": d["success"].mean() * 100,
            "mean_cdiff": d["cdiff"].mean()
        })

class_stats = pd.DataFrame(class_rows)

class_stats.to_csv(
    "results/analysis/class_attack_success.csv",
    index=False
)

print("\n================ CLASS-LEVEL DE 1-PIXEL ================\n")

print(
    class_stats[
        class_stats["attack"] == "DE 1-pixel"
    ].sort_values(
        "ASR_percent",
        ascending=False
    ).to_string(index=False)
)

# ---------------------------
# 4. MOST AFFECTED IMAGES
# ---------------------------

for name, df in dfs.items():

    successes = (
        df[df["success"]]
        .sort_values("cdiff", ascending=False)
        .head(10)
    )

    print(f"\n================ TOP SUCCESSFUL: {name} ================\n")

    display_cols = [
        "image", "true", "predicted",
        "success", "cdiff"
    ]

    print(successes[display_cols].to_string(index=False))

# ---------------------------
# 5. ASR PLOT
# ---------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    summary["attack"],
    summary["ASR_percent"]
)

plt.ylabel("Attack Success Rate (%)")
plt.xlabel("Attack")
plt.title("Attack Success Rate Across Experimental Conditions")
plt.ylim(0, 100)
plt.xticks(rotation=15)

for i, value in enumerate(summary["ASR_percent"]):
    plt.text(
        i,
        value + 2,
        f"{value:.0f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "results/analysis/attack_success_rates.png",
    dpi=300
)
plt.close()

# ---------------------------
# 6. CONFIDENCE REDUCTION PLOT
# ---------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    summary["attack"],
    summary["mean_cdiff_all"]
)

plt.ylabel("Mean reduction in true-class probability")
plt.xlabel("Attack")
plt.title("Mean Confidence Reduction")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(
    "results/analysis/confidence_reduction.png",
    dpi=300
)
plt.close()

print("\n================ SAVED OUTPUTS ================\n")
print("results/analysis/summary_metrics.csv")
print("results/analysis/pairwise_comparisons.csv")
print("results/analysis/class_attack_success.csv")
print("results/analysis/attack_success_rates.png")
print("results/analysis/confidence_reduction.png")

print("\nCSV columns:")
for name, df in dfs.items():
    print(name, "->", list(df.columns))
