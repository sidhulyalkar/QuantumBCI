"""Run the reusable density benchmark on a tiny frozen-embedding example."""

from __future__ import annotations

import numpy as np

from quantumbci.benchmarking import IndexSplit, benchmark_density_embeddings


def make_example(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    windows = []
    labels = []
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for index in range(80):
        label = index % 2
        phase = 0.13 * index
        a = np.sin(t + phase)
        sign = 1.0 if label else -1.0
        window = np.stack([a, sign * a, np.cos(2 * t), np.sin(3 * t)], axis=1)
        window += rng.normal(0.0, 0.02, size=window.shape)
        windows.append(window)
        labels.append(label)
    return np.stack(windows), np.asarray(labels)


def main() -> None:
    embeddings, labels = make_example()
    split = IndexSplit(np.arange(60), np.arange(60, 80), name="fixed-example")
    result = benchmark_density_embeddings(embeddings, labels, split)
    for key, value in result.to_mapping().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
