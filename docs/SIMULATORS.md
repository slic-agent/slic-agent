# Simulator backends

`CIRCUIT_SIM_BACKEND` selects the simulator: `ngspice` (default, free) or `spectre`. Spectre is the more stable of the two;
use it when you have a licence. The released task sets were simulated with ngspice, and an episode re-simulates its start
with the selected backend.

## ngspice (default)

`NGSPICE_BIN` names the binary (default: `ngspice` on `PATH`; `environment.yml` installs ngspice 41). The AC sweep follows
the netlist's Spectre analysis line, so both backends measure on the same points.

## Spectre

Set `CIRCUIT_SIM_BACKEND=spectre` and either

- `CIRCUITOP_SPECTRE_BACKEND=local` with `spectre` on `PATH` (`scripts/configure_spectre_backend.sh` shows a module-based
  setup), or
- `CIRCUITOP_SPECTRE_BACKEND=http` with `CIRCUITOP_SPECTRE_API_URL` (and optionally `CIRCUITOP_SPECTRE_API_TOKEN`) for a
  remote simulation service.

## Measurements

Each test bench runs a DC operating point and an AC sweep from 1 Hz to 500 GHz (100 points per decade).

- Gain: the magnitude at 1 kHz.
- Bandwidth: the −3 dB frequency relative to the first sweep point (`bandwidth_kind` = `f3db`; none if there is none).
- Phase margin: the unwrapped phase at the first 0 dB crossing minus the critical phase (1 Hz phase − 180°); negative means
  unstable in unity feedback.
- In-band flatness: the peak within the bandwidth minus the low-frequency gain.

## Technology

`env.circuitop.tech_node` (65nm in every config) selects the device model card under `circuit/dataset/`; the 65 nm card
ships with the repository, and `slic data --cards` downloads the 22, 32, 45, 90 and 130 nm cards (public PTM cards in
Spectre syntax) used by the technology-node task sets. `env.circuitop.tech_model` points at any other `.scs` card.
