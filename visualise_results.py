import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.datasets import cifar10

import helper


os.makedirs("results/analysis/examples", exist_ok=True)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def parse_perturbation(value):
    if isinstance(value, np.ndarray):
        return value.astype(float)

    if isinstance(value, list):
        return np.asarray(value, dtype=float)

    value = str(value).strip()

    # Handle normal Python-list representation
    try:
        return np.asarray(ast.literal_eval(value), dtype=float)
    except Exception:
        pass

    # Fallback for NumPy-style strings
    value = value.replace("[", " ").replace("]", " ").replace(",", " ")
    return np.fromstring(value, sep=" ")


def draw_pixel_boxes(ax, perturbation):
    p = np.asarray(perturbation).astype(int).reshape(-1, 5)

    for row, col, r, g, b in p:
        # helper.perturb_image indexes image[row, col]
        rect = Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            linewidth=2
        )
        ax.add_patch(rect)


def save_comparison(df, image_id, attack_name, filename, x_test):
    row = df[df["image"] == image_id].iloc[0]

    perturbation = parse_perturbation(row["perturbation"])

    original = x_test[image_id]
    adversarial = helper.perturb_image(
        perturbation,
        original
    )[0]

    true_class = int(row["true"])
    predicted = int(row["predicted"])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(original)
    axes[0].set_title(
        f"Original\n{classes[true_class]}"
    )

    axes[1].imshow(original)
    draw_pixel_boxes(
        axes[1],
        perturbation
    )
    axes[1].set_title(
        f"Modified location(s)\n{int(row['pixels'])} pixel(s)"
    )

    axes[2].imshow(adversarial)
    axes[2].set_title(
        f"Adversarial\n"
        f"{classes[true_class]} → {classes[predicted]}"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{attack_name} | Image {image_id} | "
        f"Confidence reduction = {row['cdiff']:.3f}"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"{attack_name}: image={image_id}, "
        f"{classes[true_class]} -> {classes[predicted]}"
    )

    print(
        "Perturbation:",
        perturbation.astype(int).reshape(-1, 5)
    )


_, (x_test, _) = cifar10.load_data()

de1 = pd.read_csv("results/de_100.csv")
rnd = pd.read_csv("results/random_100.csv")
de3 = pd.read_csv("results/de_3pixel_100.csv")
de5 = pd.read_csv("results/de_5pixel_100.csv")


# --------------------------------------------------
# 1. Find strongest DE-only one-pixel example
# --------------------------------------------------

paired = de1.merge(
    rnd,
    on="image",
    suffixes=("_de", "_random")
)

de_only = paired[
    (paired["success_de"] == True) &
    (paired["success_random"] == False)
].copy()

de_only = de_only.sort_values(
    "cdiff_de",
    ascending=False
)

if len(de_only):
    image_id = int(de_only.iloc[0]["image"])

    save_comparison(
        de1,
        image_id,
        "DE 1-pixel — random failed",
        "results/analysis/examples/de1_random_failed.png",
        x_test
    )


# --------------------------------------------------
# 2. Strong DE 1-pixel example
# --------------------------------------------------

best_de1 = (
    de1[de1["success"]]
    .sort_values("cdiff", ascending=False)
    .iloc[0]
)

save_comparison(
    de1,
    int(best_de1["image"]),
    "DE 1-pixel",
    "results/analysis/examples/de1_best.png",
    x_test
)


# --------------------------------------------------
# 3. Strong DE 3-pixel example
# --------------------------------------------------

best_de3 = (
    de3[de3["success"]]
    .sort_values("cdiff", ascending=False)
    .iloc[0]
)

save_comparison(
    de3,
    int(best_de3["image"]),
    "DE 3-pixel",
    "results/analysis/examples/de3_best.png",
    x_test
)


# --------------------------------------------------
# 4. Strong DE 5-pixel example
# --------------------------------------------------

best_de5 = (
    de5[de5["success"]]
    .sort_values("cdiff", ascending=False)
    .iloc[0]
)

save_comparison(
    de5,
    int(best_de5["image"]),
    "DE 5-pixel",
    "results/analysis/examples/de5_best.png",
    x_test
)

print("\nFigures saved under results/analysis/examples/")