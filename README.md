# One Pixel Attack reproduction

Independent Keras reproduction of the low-pixel, untargeted attack studied by Su et al. The target is the repository's pretrained CIFAR-10 ResNet, not a newly trained model or an ImageNet ResNet. This bundle includes the existing experimental code and the exact available checkpoint.

## Install and reproduce

Extract these files into the repository root. Install Conda (or Miniforge), then create the environment once:

```sh
conda env create -f environment.yml
```

From a clean checkout, the single reproduction command is:

```sh
conda run --no-capture-output -n one-pixel-reproduction python reproduce.py
```

This reruns all four attacks, validates image pairing, calculates all analysis tables, and generates all six requested figures. It does not reuse saved attacks. Existing named outputs are overwritten. Internet is needed on the first run to download CIFAR-10 through `tensorflow.keras.datasets.cifar10`; later runs use the Keras cache. Allow several hours or longer on CPU and several GB of available RAM; runtime has not been benchmarked. The random baseline materialises 7,600 perturbed images at a time.

## Files that must be committed

Commit this bundle's Python files, `environment.yml`, `model.sha256`, and `networks/models/resnet.h5`, including the existing network support files. The included `.gitignore` makes an exception for this exact checkpoint; if retaining your old ignore file, add `!networks/models/resnet.h5` after `*.h5`, or force-add this checkpoint. Verify the checkpoint is tracked before pushing. No manual download or training is then required after cloning. Preserve the repository's existing license and attribution.

## Fixed experiment

- CIFAR-10 test images, 32 by 32 RGB; eligibility is correct clean classification by the checkpoint.
- Sample 100 eligible test images without replacement using `numpy.random.default_rng(42).choice`. The same image IDs are required across all four conditions.
- Each DE condition runs in a separate process with NumPy seed 42: 1, 3, and 5 pixel tuples `(row, column, R, G, B)`, maxiter 75, nominal popsize 100.
- Keep the repository's bundled `differential_evolution.py`: `best1bin`, mutation dithering `(0.5, 1)`, recombination 1, `atol=-1`, no polishing, and early stopping when the best candidate misclassifies. This is not the original paper's exact DE configuration.
- Population multiplier is `max(1, 100 // (5 * pixels))`; actual populations are 100, 90, and 100. Bounds and integer conversion are those in `attack.py` and `helper.py`; repeated pixel locations can produce fewer distinct modified pixels.
- Random baseline: 7,600 integer one-pixel candidates per image, coordinates 0–31 and RGB 0–255. Its seed-42 generator first draws the shared sample, then draws candidates sequentially. Retain the successful candidate with lowest true-class probability, or the strongest failed attempt if none succeeds.
- ResNet input channels are standardised with means `[125.307, 122.95, 113.865]` and standard deviations `[62.9932, 62.0887, 66.7048]`, as implemented by the repository wrapper.

The 7,600 random candidates match the nominal one-pixel DE population budget `100 * (75 + 1)`. This is not an exact model-query match: DE can stop early, has callback/final predictions, and the three-pixel population is 90. Candidate count and model-call count are different quantities.

## Outputs

Raw attacks: `results/de_100`, `results/de_3pixel_100`, `results/de_5pixel_100`, and `results/random_100`, each as `.csv` and `.pkl`. Prefer CSV for inspection; load pickle files only from trusted sources.

Analysis in `results/analysis/`:

| File | Contents |
| --- | --- |
| `summary_metrics.csv` | Success counts, ASR, 95% Wilson intervals (z=1.96), mean/median confidence reduction over all images, and mean over successful attacks |
| `pairwise_comparisons.csv` | Paired DE1 vs random1 and DE3 vs DE5 discordant counts, both/neither counts, and exact two-sided binomial p-values (McNemar-style) |
| `class_attack_success.csv` | Class sample counts, successes, ASR and mean confidence reduction |
| `attack_success_rates.png` | ASR for the four conditions |
| `confidence_reduction.png` | Mean true-class probability reduction over all 100 images |
| `examples/de1_best.png` | Successful DE1 example with largest confidence reduction |
| `examples/de1_random_failed.png` | Largest DE1 confidence reduction among images where DE1 succeeds and random fails |
| `examples/de3_best.png` | Strongest successful DE3 example |
| `examples/de5_best.png` | Strongest successful DE5 example |
| `run_manifest.json` | Image IDs, settings, package versions, platform, checkpoint and source hashes |

Confidence reduction is clean true-class probability minus perturbed true-class probability, not the probability of the predicted adversarial class. ASR uses all 100 initially correct images as its denominator. Paired tests join by image ID, use only discordant outcomes, and return p=1 if none are discordant. Class-level samples are small and not balanced. The example script chooses examples from the new results; it does not hard-code historical image IDs. A missing qualifying example causes the pipeline to fail instead of claiming all figures were produced.

Each stage writes a log under `results/logs/`; the attack logs include the clean model evaluation. A failed stage stops the command with a nonzero exit status.

## Report comparison and limitations

Historical report values are clean test accuracy 92.31%, random1 ASR 26%, DE1 33%, DE3 78%, DE5 80%, and DE1 versus random1 exact paired p=0.015625. These are comparison values, never values inserted into regenerated results. The supplied `reference_results/` CSVs preserve the available historical per-image records for auditing, but the reproduction command never reads them.

The entrypoint pins the main package versions and uses CPU inference, single TensorFlow threads, deterministic operations and disabled oneDNN optimisations to reduce numerical variation. The historical execution platform was not recorded here, so bit-identical recovery of every historical attack outcome cannot be guaranteed: small inference changes can alter DE's search trajectory. Compare regenerated values and image IDs with the report and disclose differences. Do not change settings merely to force the expected percentages.

Validation of this deliverable: Python syntax, paired-image consistency and historical-result analysis were checked locally. A full fresh attack rerun and fresh Conda installation were not performed; consequently the environment and end-to-end execution remain unverified. The full command is provided to perform that verification on your machine.

Reference: Su, Vargas and Sakurai, “One Pixel Attack for Fooling Deep Neural Networks,” IEEE Transactions on Evolutionary Computation (2019). Model implementation credits in `networks/resnet.py` and the bundled optimizer's provenance are retained.
