# Code quality and static-debt policy

QuantumBCI treats software-quality debt similarly to scientific uncertainty: tolerated exceptions should be explicit, bounded, and made easier to remove than to expand.

## Static-debt ratchet

`tests/test_static_debt_budget.py` turns the current audit into an executable contract.

It currently enforces that production Python code:

- contains no bare `except:` handlers;
- does not catch `BaseException`;
- contains exactly one deliberately tolerated broad `except Exception`, at the optional neurOS registry boundary;
- allows `type: ignore[...]` only in the explicitly listed current numerical/trajectory modules;
- requires every typing suppression to name its error code;
- fails when an allowlist entry becomes stale, so removing debt requires shrinking the budget rather than leaving permanent exemptions behind.

This is intentionally a **ratchet**, not a declaration that the current codebase is fully typed. The goal is to make the existing debt monotonically easier to see and reduce.

## Why one broad exception remains

`quantumbci.integrations.neuros.NeurOSFoundationEncoder.from_registry` translates errors from an optional external neurOS adapter registry into `NeurOSUnavailableError`. QuantumBCI cannot depend on neurOS's complete runtime exception hierarchy without defeating the optional dependency boundary.

The broad catch is therefore tolerated only at that translation seam. If neurOS exposes a stable typed adapter-error protocol, QuantumBCI should narrow the catch and the static-debt budget must be reduced in the same change.

## Typing debt

The current `type: ignore[...]` files are concentrated in trajectory-role and NumPy-heavy dynamics code rather than spread throughout the package. This makes an incremental typing program practical.

The preferred removal order is:

1. replace stringly typed evidence-role parameters with shared `Literal`/protocol types;
2. introduce typed array/protocol boundaries where NumPy's inferred type is the only blocker;
3. type artifact/policy/public-API boundaries;
4. enable a static type checker on that qualified subset;
5. expand the typed surface only when the existing subset remains clean.

Do not silence a type checker merely to achieve a green badge. A suppression must identify a specific checker diagnostic and should live as close as possible to the external or numerical typing seam that requires it.

## Relationship to scientific correctness

Static analysis does not replace the scientific contract matrix. Conversely, scientific tests do not justify unchecked exception handling or invisible typing escapes. Both layers should fail closed for the kinds of errors they are designed to catch.
