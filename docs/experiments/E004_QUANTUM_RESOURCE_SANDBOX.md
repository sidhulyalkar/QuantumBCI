# E004: Quantum algorithm resource sandbox

**Claim ceiling:** quantum algorithm. No physical-neural quantum claim.

This lane exists to stop QuantumBCI from jumping from an interesting equation directly to hardware. A QPU experiment begins only after E001/E002 promotes a concrete observable that is valuable on classical data.

## Candidate mappings

1. **QFT observable:** amplitude-encode a deliberately compressed signal and estimate a small number of Fourier-basis probabilities. Classical FFT/Goertzel is the mandatory reference.
2. **Quantum kernel:** map a low-dimensional promoted latent state to a feature map and compare against RBF, polynomial, Matérn and random-feature kernels under matched train/test splits.
3. **QLSA observable:** only for a sparse/well-conditioned/Hermitian-or-embedded linear system where the desired output is a small observable. Never reconstruct a full inverse/solution and call it an HHL advantage.

## Four-step execution pattern

Follow the current IBM/Qiskit pattern conceptually:

1. map the promoted problem to circuits/operators;
2. optimize/transpile for a target backend;
3. execute on simulator, noisy simulator, then hardware only if eligible;
4. analyze observable error and complete resource ledger.

## Mandatory resource ledger

- input dimension and compression cost;
- state preparation circuit/algorithm and cost assumptions;
- width, depth, two-qubit gates;
- transpiled width/depth on the target snapshot;
- shot count and confidence interval;
- error mitigation overhead;
- queue/execution time where reported;
- classical preprocessing/postprocessing time;
- strongest classical algorithm and error tolerance;
- total wall-clock and monetary cost when available.

## Hardware gate

No fixed qubit/depth number is hard-coded because backend capabilities change. The gate consumes the current backend calibration snapshot and requires the noisy simulation to predict usable error at a declared shot budget. If not, the experiment terminates successfully as `not_eligible_for_qpu`.

## Advantage language

A quantum circuit producing the correct answer is not an advantage. Any speedup/efficiency statement must include loading, sampling and readout and compare end-to-end against the strongest classical algorithm for the same observable/error tolerance.

## References

- Current IBM Quantum development/Qiskit pattern: https://quantum.cloud.ibm.com/docs/en/guides/optimize-for-hardware
- Historical HHL docs explicitly note oracle/readout caveats and are not current API: https://quantum.cloud.ibm.com/docs/api/qiskit/0.40/qiskit.algorithms.linear_solvers.HHL
