# QuantumBCI experiment registry

This directory contains machine-readable experiment manifests. The narrative protocol for each study lives in `docs/experiments/`.

The registry is intentionally small and staged:

| ID | Study | Claim ceiling | Status |
| --- | --- | --- | --- |
| E001 | Density geometry on EEG representations | quantum-inspired | implementation-ready |
| E002 | Lindblad vs classical latent dynamics | quantum-inspired | implementation-ready after E001 embedding cache |
| E003 | Context/order effects | quantum-inspired | retrospective harness ready; prospective data requires preregistration/ethics |
| E004 | Quantum resource sandbox | quantum algorithm | gated on E001/E002 producing a useful observable |
| E005 | Physical quantum mechanism screen | physical quantum | protocol only; no automatic promotion |

A manifest is a scientific contract, not just a job launcher. It specifies the claim class, data/split assumptions, stage DAG, primary metrics, artifacts, and promotion gates before results are inspected.

Raw participant data, credentials, model checkpoints with restrictive licenses, and subject-identifying metadata must never be committed here.
