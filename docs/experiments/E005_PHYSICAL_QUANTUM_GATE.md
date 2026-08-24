# E005: Physical quantum neural-mechanism gate

**Claim ceiling:** physical quantum, but nothing is promoted into this class by model fit alone.

The purpose of E005 is to decide whether a proposed biological quantum mechanism is precise enough to justify a real experiment or collaboration.

## Candidate families, not endorsed mechanisms

Examples worth screening include spin-dependent radical-pair/redox processes, proton/electron tunnelling in a specific molecular substrate, or quantum-sensing measurements of nanoscale biological fields. Evidence that such phenomena occur elsewhere in biology does not imply functional relevance to EEG/cognition.

## Required mechanism card

A candidate must specify all of the following before code or data collection:

1. **Degree of freedom:** spin, proton coordinate, electronic excitation, etc.
2. **Physical substrate:** named molecule/complex/channel/compartment.
3. **Preparation/source:** how the relevant non-classical state/process arises biologically.
4. **Coupling:** a quantitative route from the microscopic process to a measurable neural/biological variable.
5. **Timescale:** predicted coherence/tunnelling/spin-dynamic timescale under physiological temperature/noise.
6. **Observable/witness:** an operational measurement, not a classifier score.
7. **Perturbation:** a manipulation for which the quantum and strongest classical model make different predictions.
8. **Detection floor:** expected effect size relative to instrument resolution/background.
9. **Classical mimic:** stochastic, nonlinear, electromagnetic, chemical, thermal or instrumentation explanation that must be defeated.
10. **Replication design:** independent replication condition/lab/cohort where feasible.

## Evidence ladder

`literature plausibility -> quantitative simulation -> instrument feasibility -> in-vitro/ex-vivo pilot -> preregistered perturbation -> independent replication -> only then neural functional relevance`

Human EEG should be downstream evidence, not the first detector of a microscopic quantum mechanism.

## Automatic rejection conditions

- substrate is unspecified;
- expected timescale is incompatible with the proposed measurement;
- predicted signal is below the detection floor;
- the perturbation changes many classical biological variables in the same direction;
- only model accuracy distinguishes hypotheses;
- a macroscopic quantum label is inferred from non-commutative statistics alone.

## Research context

The 2026 PNAS perspective *What is quantum biology?* emphasizes suitable quantum-scale probes, quantum-to-macroscale amplification, and the possibility of classical biological machinery mimicking quantum behavior. QuantumBCI adopts those constraints as its default standard: https://pubmed.ncbi.nlm.nih.gov/41860951/
