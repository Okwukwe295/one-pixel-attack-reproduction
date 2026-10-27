#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10

from networks.resnet import ResNet
import helper


def random_attack(model, x_test, y_test, img_id, rng,
                  queries=7600, batch_size=256):

    true_class = int(y_test[img_id, 0])

    # Prediction before attack
    prior_probs = model.predict(
        np.array([x_test[img_id]])
    )[0]

    # Generate random one-pixel perturbations:
    # (x, y, R, G, B)
    candidates = np.empty((queries, 5), dtype=np.int32)

    candidates[:, 0] = rng.integers(0, 32, size=queries)
    candidates[:, 1] = rng.integers(0, 32, size=queries)
    candidates[:, 2:5] = rng.integers(
        0, 256,
        size=(queries, 3)
    )

    # Apply every candidate to the same original image
    perturbed_images = helper.perturb_image(
        candidates,
        x_test[img_id]
    )

    # Evaluate all random perturbations
    probs = model.predict(
        perturbed_images
    )

    predictions = np.argmax(probs, axis=1)

    # Did any random perturbation cause misclassification?
    successful = predictions != true_class

    if np.any(successful):
        # Among successful attacks, retain the one that reduced
        # true-class confidence the most
        successful_indices = np.where(successful)[0]

        best_idx = successful_indices[
            np.argmin(
                probs[successful_indices, true_class]
            )
        ]
    else:
        # Otherwise retain the strongest failed attempt
        best_idx = np.argmin(
            probs[:, true_class]
        )

    predicted_class = int(predictions[best_idx])

    success = predicted_class != true_class

    cdiff = (
        float(prior_probs[true_class])
        - float(probs[best_idx, true_class])
    )

    return {
        "model": model.name,
        "pixels": 1,
        "image": int(img_id),
        "true": true_class,
        "predicted": predicted_class,
        "success": bool(success),
        "cdiff": cdiff,
        "queries": queries,
        "perturbation": candidates[best_idx].tolist()
    }


def main():

    parser = argparse.ArgumentParser(
        description="Random one-pixel attack baseline"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=7600
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--save",
        default="results/random_100.pkl"
    )

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load CIFAR-10
    _, (x_test, y_test) = cifar10.load_data()

    # Same ResNet used in DE experiment
    model = ResNet(load_weights=True)

    # Find correctly classified test images
    network_stats, correct_imgs = helper.evaluate_models(
        [model],
        x_test,
        y_test
    )

    print("\nNetwork statistics:")
    print(
        pd.DataFrame(
            network_stats,
            columns=["name", "accuracy", "param_count"]
        )
    )

    correct_df = pd.DataFrame(
        correct_imgs,
        columns=[
            "name",
            "img",
            "label",
            "confidence",
            "pred"
        ]
    )

    valid_imgs = correct_df[
        correct_df.name == model.name
    ].img.to_numpy()

    if args.samples > len(valid_imgs):
        raise ValueError(
            f"Requested {args.samples} images but only "
            f"{len(valid_imgs)} are correctly classified."
        )

    # IMPORTANT:
    # Same sampling rule + seed as attack.py
    img_samples = rng.choice(
        valid_imgs,
        size=args.samples,
        replace=False
    )

    print(
        f"\nSelected {args.samples} images "
        f"using seed {args.seed}"
    )

    print(
        f"Random search budget: "
        f"{args.queries} perturbations per image\n"
    )

    results = []

    for i, img_id in enumerate(img_samples):

        print(
            f"resnet - image {img_id} - "
            f"{i + 1} / {len(img_samples)}"
        )

        result = random_attack(
            model,
            x_test,
            y_test,
            img_id,
            rng,
            queries=args.queries,
            batch_size=args.batch_size
        )

        results.append(result)

    df = pd.DataFrame(results)

    print("\nResults:")
    print(
        df[
            [
                "model",
                "pixels",
                "image",
                "true",
                "predicted",
                "success",
                "cdiff"
            ]
        ]
    )

    success_rate = df["success"].mean() * 100

    print(
        f"\nRandom Attack Success Rate: "
        f"{success_rate:.2f}%"
    )

    # Save results
    df.to_pickle(args.save)

    csv_path = os.path.splitext(args.save)[0] + ".csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaving pickle to {args.save}")
    print(f"Saving CSV to {csv_path}")


if __name__ == "__main__":
    main()