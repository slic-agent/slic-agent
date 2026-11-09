# Task sets

The task sets are on the Hugging Face organisation [`slic-agent`](https://huggingface.co/slic-agent) (`slic data`). All
are 65 nm bulk CMOS (the open predictive card `circuit/dataset/65nm_bulk.scs`) except the `nodes` family; 41 topologies,
34 in distribution and 7 held out.

**Where the files go.** `slic data` writes the task sets to `data/<family>/` and the device model cards (`.scs`) to
`circuit/dataset/` (next to the 65 nm card); the configs use these repository-relative paths. When downloading without
`slic` (`hf download slic-agent/slic-data --repo-type dataset --local-dir data`), copy `data/ptm-cards/*.scs` to
`circuit/dataset/`.

## Row format

Each row has two columns, `data_source` and `extra_info`:

| field | content |
|---|---|
| `extra_info.task_id`, `split`, `index`, `goal_level` | task identity; `goal_level` is the goal's difficulty tier |
| `extra_info.initial_state_class` | `invalid` when simulating the start fails, its gain is ≤ 0 dB or not finite, or its in-band flatness is missing or above 6 dB; otherwise `valid` |
| `extra_info.env_kwargs.topology` | topology name |
| `extra_info.env_kwargs.target` | the goal: `gain_min` (dB), `power_max` (µW), `bandwidth_min` (MHz), `phase_margin_min_deg`, `phase_margin_max_deg` |
| `extra_info.env_kwargs.task_data.structure` | the circuit as `netlist` (simulated; shown to the model with `representation: netlist`) and `xml` (shown with the default `representation: xml`) |
| `extra_info.env_kwargs.task_data.admissible_actions` | the action space (tunable devices and parameters, action format, step limits) as `xml` (devices by handle, `@10`) and `netlist` (by device id, `X0.MM0`), for `env.circuitop.representation` |
| `extra_info.env_kwargs.task_data.initial_state` | starting `params` and their simulated `metrics` (`gain_dB`, `power_uW`, `bandwidth_MHz`, `bandwidth_kind`, `inband_flatness_db`, `phase_margin_deg`, `phase_margin_status`; docs/SIMULATORS.md, "Measurements"), which the environment's reset reproduces |
| `extra_info.env_kwargs.task_data.metadata` | `complexity`, `signal_mode`, `circuit_kind` |

The files of one family share one schema, so the per-topology maps (`initial_state.params`, the action spaces' per-device
parameter lists) also carry the other topologies' keys with null values; ignore null entries (the environment does).

**Browse columns.** The released files also carry flat copies of the most-used fields as top-level columns for browsing and
filtering (the Hugging Face viewer filters top-level columns only): `task_id`, `topology`, `circuit_kind`, `held_out`,
`goal_level`, `start_class` (= `initial_state_class`), the five targets and `start_gain_dB`, `start_power_uW`,
`start_bandwidth_MHz`, `start_phase_margin_deg`, `start_inband_flatness_db`. The code reads `extra_info` only.

The environment writes every prompt itself from these fields and the reward preset. The Hub dataset also has
`topologies.csv`, `topologies/<topology>/{schematic.svg,netlist.scs}` and a `stats.json` per family.

## `standard` (`slic data --family standard`)

| file                 | rows | content                                                                  |
|----------------------|-----:|--------------------------------------------------------------------------|
| `train.parquet`      | 6442 | RL training pool, 34 in-distribution topologies                          |
| `validation.parquet` |  408 | the first half of `test_iid` (12 tasks per topology), evaluated during training for checkpoint selection; a subset because simulating all 832 tasks at every validation is slow |
| `test_iid.parquet`   |  832 | in-distribution test set, 24 tasks per topology (40 for `single_ended_telescopic_pmos_LG_pmos_biasn_LV`) |
| `test_ood.parquet`   |  280 | 7 held-out topologies, 40 tasks each                                     |

## `hard` (`slic data --family hard`)

Not used in the paper: the same topologies and goal distribution, with one of the four specifications starting far from
its target.

| file                 | rows | content                                         |
|----------------------|-----:|-------------------------------------------------|
| `train.parquet`      | 4352 | training pool (128 per topology)                |
| `validation.parquet` |  102 | training-time validation (3 per topology)       |
| `test_iid.parquet`   |  436 | in-distribution topologies (12 per topology, 40 for the telescopic one) |
| `test_ood.parquet`   |  280 | held-out topologies (40 per topology)           |

## `nodes` (`slic data --family nodes`, cards: `slic data --cards`)

The standard `validation.parquet` and `test_ood.parquet` rebuilt for 22, 32, 45, 90 and 130 nm: same topologies, goals and
task ids, the starting design re-simulated under the node's card (sizes below the node's floors raised to the floor):
`validation_<node>.parquet` (408) and `test_ood_<node>.parquet` (280). An evaluation must select the matching card
(`tech_node:` in its config, e.g. `configs/eval/node_ood.yaml`).

## Success

Four-spec success: gain ≥ `gain_min`, power ≤ `power_max`, bandwidth ≥ `bandwidth_min` and `phase_margin_min_deg` ≤ PM ≤
`phase_margin_max_deg`, each within the preset's relative tolerance, on the post-action state. Pass@k uses the unbiased
estimator over n samples per task.

## Building your own tasks

`slic tasks` turns your own circuits (and optionally your own technology card) into a task parquet in the row format
above, built and simulated by the same code the environment runs, so `slic train` / `slic eval` take it unchanged.

```yaml
# my_tasks.yaml
tech_node: 65nm                    # geometry box (and card, unless tech_model is set): 22nm 32nm 45nm 65nm 90nm 130nm
# tech_model: pdk/my_card.scs      # your own Spectre-syntax card with BSIM4 models nch / pch in a `tt` section
output: data/my_tasks/train.parquet
split: train                       # default for tasks without one (default train)
goal_level: custom                 # default label (default custom)
tasks:
  - topology: common_source_amplifier_with_current_source_load
    netlist: |                               # or netlist_file: circuits/cs_amp.scs (relative to this file)
      subckt cs_currentsource_load_ibias vb vdd vin vout vss
          M0 (vout vin vss vss) nch w=M0_w l=M0_l
          M1 (vout vb vdd vdd) pch w=M1_w l=M1_l
      ends cs_currentsource_load_ibias
      subckt currentsource_pmos_vb_generator vb vdd vss
          M2 (vb vb vdd vdd) pch w=M2_w l=M2_l
          I0 (vb vss) isource dc=iref type=dc
      ends currentsource_pmos_vb_generator
      X0 (vb vdd vb_in vout 0) cs_currentsource_load_ibias
      X1 (vb vdd 0) currentsource_pmos_vb_generator
    params: {M0_w: 20u, M0_l: 500n, M1_w: 10u, M1_l: 1u, M2_w: 10u, M2_l: 1u, iref: 5u, vdd: 1.2}
    target: {gain_min: 30, power_max: 100, bandwidth_min: 1.0, phase_margin_min_deg: 30, phase_margin_max_deg: 90}
    task_id: cs_amp_0                        # optional, default <topology>__<index:05d>
```

```bash
slic tasks --spec my_tasks.yaml --workers 8      # --check: validate only (no simulation); --out overrides `output:`
```

What a task must contain:

| key | content |
|---|---|
| `topology` | a name; it selects the test bench (below) and becomes `env_kwargs.topology` |
| `netlist` / `netlist_file` | the circuit only: `subckt ... ends` blocks plus their top-level `X` instances, as in `structure.netlist` of a released row; device sizes (`w`, `l`, `r`, `c`, `dc`) are parameter names; no `parameters`, `include`, sources, loads or analyses (the environment adds them) |
| `params` | the starting value of every parameter the netlist uses, plus `vdd` (SI strings such as `1.2u` are kept as written; numbers are stored as SI strings with 12 significant digits) |
| `target` | `gain_min` (dB), `power_max` (µW), `bandwidth_min` (MHz), `phase_margin_min_deg`, `phase_margin_max_deg` |
| optional | `task_id`, `split`, `goal_level`, `index`, `metadata` (`circuit_kind`: `simple`, `complex_single_ended`, `complex_fully_differential`; inferred from the topology name otherwise) |

The environment wraps the circuit in one of three fixed test benches, so the circuit has to expose the nets they use:
a **simple** circuit connects its input to `vb_in` (`vin` for a topology named `common_gate_amplifier_...`), its output
to `vout`, the supply to `vdd` and ground to `0`; a **complex** OTA (topology name containing `single_ended` or
`fully_differential`) has a main sub-circuit with pins `Vinn`, `Vinp`, `Vout...` (two for fully differential), `vdd`,
`vss`, and bias generators as further `X` instances on nets named `LG_*`. The released rows and the netlists under
`topologies/` of the Hugging Face dataset show both styles. A design outside the node's geometry box, or malformed input,
stops with a message naming the task and the field.

To use the result, copy a config and point it at your files, at the node the tasks were built for:

```yaml
# training (a copy of configs/train/slic_random.yaml): replaces dataset_dir:
train_file: data/my_tasks/train.parquet
val_file: data/my_tasks/validation.parquet
overrides: [env.circuitop.tech_node=65nm]

# evaluation (a copy of configs/eval/standard_iid.yaml)
tasks: data/my_tasks/test.parquet
protocol:
  tech_node: 65nm
```

With a device model card of your own, add `env: {CIRCUITOP_TECH_MODEL: /path/to/card.scs}` to the config.
