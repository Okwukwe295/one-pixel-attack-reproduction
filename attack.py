#!/usr/bin/env python3
import os

import argparse
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10

# Target network for current reproduction/smoke test
from networks.resnet import ResNet

# Attack + helper functions
from differential_evolution import differential_evolution
import helper


class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x]

    def attack_all(self, models, samples=100, pixels=(1,), targeted=False, maxiter=75, popsize=400, verbose=False, seed=42):
        results = []

        rng = np.random.default_rng(seed)

        for model in models:
            model_results = []

            valid_imgs = self.correct_imgs[
                self.correct_imgs.name == model.name
            ].img.to_numpy()

            if samples > len(valid_imgs):
                raise ValueError(
                    f"Requested {samples} images but only "
                    f"{len(valid_imgs)} are correctly classified."
                )

            img_samples = rng.choice(
                valid_imgs,
                size=samples,
                replace=False
            )

            print(f"Selected {samples} images using seed {seed}")

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):

                    print(
                        model.name,
                        '- image',
                        img,
                        '-',
                        i + 1,
                        '/',
                        len(img_samples)
                    )

                    targets = [None] if not targeted else range(10)

                    for target in targets:

                        if targeted:
                            print(
                                'Attacking with target',
                                self.class_names[target]
                            )

                            if target == self.y_test[img, 0]:
                                continue

                        result = self.attack(
                            img,
                            model,
                            target,
                            pixel_count,
                            maxiter=maxiter,
                            popsize=popsize,
                            verbose=verbose
                        )

                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)

        return results


if __name__ == '__main__':

    model_defs = {
        'resnet': ResNet,
    }

    parser = argparse.ArgumentParser(
        description='One Pixel Attack reproduction on CIFAR-10'
    )

    parser.add_argument(
        '--model',
        nargs='+',
        choices=model_defs.keys(),
        default=['resnet']
    )

    parser.add_argument(
        '--pixels',
        nargs='+',
        default=[1],
        type=int
    )

    parser.add_argument(
        '--maxiter',
        default=75,
        type=int
    )

    parser.add_argument(
        '--popsize',
        default=400,
        type=int
    )

    parser.add_argument(
        '--samples',
        default=100,
        type=int
    )

    parser.add_argument(
        '--seed',
        default=42,
        type=int
    )

    parser.add_argument(
        '--targeted',
        action='store_true'
    )

    parser.add_argument(
        '--save',
        default='results/results.pkl'
    )

    parser.add_argument(
        '--verbose',
        action='store_true'
    )

    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)

    # Ensure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('networks/results', exist_ok=True)

    # CIFAR-10 test data
    _, test = cifar10.load_data()

    class_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    # Load pretrained model
    models = [
        model_defs[m](load_weights=True)
        for m in args.model
    ]

    attacker = PixelAttacker(
        models,
        test,
        class_names
    )

    print('\nNetwork statistics:')
    print(attacker.network_stats)

    print('\nStarting attack')

    results = attacker.attack_all(
        models,
        samples=args.samples,
        pixels=args.pixels,
        targeted=args.targeted,
        maxiter=args.maxiter,
        popsize=args.popsize,
        verbose=args.verbose,
        seed=args.seed
    )

    columns = [
        'model',
        'pixels',
        'image',
        'true',
        'predicted',
        'success',
        'cdiff',
        'prior_probs',
        'predicted_probs',
        'perturbation'
    ]

    results_table = pd.DataFrame(
        results,
        columns=columns
    )

    print('\nResults:')
    print(
        results_table[
            [
                'model',
                'pixels',
                'image',
                'true',
                'predicted',
                'success',
                'cdiff'
            ]
        ]
    )

    if len(results_table) > 0:
        success_rate = results_table['success'].mean()

        print(
            f'\nAttack Success Rate: '
            f'{success_rate * 100:.2f}%'
        )

    print('\nSaving pickle to', args.save)

    with open(args.save, 'wb') as file:
        pickle.dump(results, file)

    csv_path = args.save.replace('.pkl', '.csv')

    results_table.to_csv(
        csv_path,
        index=False
    )

    print('Saving CSV to', csv_path)