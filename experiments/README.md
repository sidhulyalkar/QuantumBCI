# QuantumBCI experiment registry

This directory contains machine-readable experiment manifests. The narrative protocol for each study lives in `docs/experiments/`.

The registry is intentionally small and staged:

| ID | Study | Claim ceiling | Status |
| --- | --- | --- | --- |
| E001 | Density geometry on neurOS + EEG representations | quantum-inspired | implementation-ready around neurOS evidence authority |
| E002 | Lindblad vs classical latent dynamics | quantum-inspired | implementation-ready after E001 embedding cache |
| E003 | Context/order effects | quantum-inspired | retrospective harness ready; prospective data requires preregistration/ethics |
| E004 | Quantum resource sandbox | quantum algorithm | gated on E001/E002 producing a useful observable |
| E005 | Physical quantum mechanism screen | physical quantum | protocol only; no automatic promotion |

A manifest is a scientific contract, not just a job launcher. It specifies the claim class, data/split assumptions, stage DAG, primary metrics, artifacts, and promotion gates before results are inspected.

## Shared neurOS evidence authority

E001 deliberately reuses neurOS rather than creating a second neural-data/runtime stack. Primary longitudinal lanes should consume neurOS `GroupedEvaluationData`, chronological `EvaluationPartition` objects, and `NestedCalibrationSplit` authority. QuantumBCI then changes the representation/mechanism while the sample identities remain frozen.

The intended final run identity binds all of:

- QuantumBCI manifest and source revision;
- upstream/raw dataset checksum;
- neurOS partition fingerprint;
- neurOS calibration/evaluation fingerprint when applicable;
- exact neurOS source revision and package versions.

See `docs/NEUROS_INTEGRATION.md`. The reciprocal neurOS discoverability work is tracked in `sidhulyalkar/neurOS-v1#29`.

Raw participant data, credentials, model checkpoints with restrictive licenses, and subject-identifying metadata must never be committed here.
