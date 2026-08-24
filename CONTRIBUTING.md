# Contributing to QuantumBCI

QuantumBCI welcomes mechanisms, controls, datasets, model adapters, experiment recipes, and reproducibility improvements. The project is deliberately stricter about scientific language than a normal numerical library because a correct piece of quantum mathematics can still support an incorrect biological claim.

## Before opening a PR

Run:

```bash
pip install -e '.[dev]'
pytest -q
quantumbci doctor
quantumbci smoke
```

Changes to the public CLI, recipes, run bundles, exports, or neurOS bridge should include an installed-surface or cross-repository test where appropriate.

## Claim-class rule

Every new mechanism belongs to one strongest allowed claim class:

- `classical_control`
- `quantum_inspired`
- `quantum_algorithm`
- `physical_quantum`

A mathematical density matrix, non-commuting observable, contextual fit, Lindblad parameterization, QFT circuit, or quantum kernel is not by itself evidence that neural tissue implements a physical quantum process.

New mechanism families should include or extend a `MechanismCard` with:

1. a falsifiable hypothesis;
2. measurable observables;
3. strongest classical alternatives;
4. explicit falsifiers;
5. the strongest claim class the implementation may support.

Physical-quantum claims additionally require an independently measurable substrate, timescale, operational witness, differentiating perturbation, detection floor, classical mimic controls, and replication design.

## Evidence and split discipline

Do not add convenience code that silently creates a train/test split for promoted work. Reuse immutable evidence authority from neurOS, MOABB, a preregistered registry, or caller-provided indices.

Participant/session leakage, calibration/evaluation overlap, altered dataset fingerprints, and non-finite numerical outputs are infrastructure failures. A mechanism that loses to its control is a valid negative scientific result.

## Public recipes

Recipe v1 is intentionally narrow: frozen `examples × tokens × features` embeddings, labels, explicit train indices, explicit test indices, and a quantum-inspired density benchmark. The machine-readable contract lives at:

`quantumbci/schemas/recipe-v1.schema.json`

Changes to recipe semantics require a schema-version decision. Do not silently reinterpret an existing v1 field.

## Reproducibility

A public result should ideally identify:

- raw or upstream dataset/version;
- preprocessing authority;
- model/checkpoint revision;
- frozen input hashes;
- immutable split/calibration/evaluation identity;
- QuantumBCI source revision;
- matched controls and interventions;
- appropriate inference unit;
- exported evidence object.

RO-Crate and BIDS-aware exports must remain conservative about what standard they actually conform to. SHA-256 ledgers provide integrity checking, not authorship signatures.

## Dependency boundaries

Keep the base package dependency-light. Optional integrations should remain lazy where possible.

The intended neurOS boundary is:

- neurOS owns neural runtime, replay, provenance, and evidence authority;
- QuantumBCI owns quantum-structured mechanism hypotheses and falsification controls;
- `neuros-mechint` owns shared causal/intervention evidence machinery.

Do not introduce a reverse runtime dependency from neurOS core to QuantumBCI.
