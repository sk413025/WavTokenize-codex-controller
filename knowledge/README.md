# Knowledge Layer

This directory is the durable memory of the Codex controller.

- `experiments/index.json`: normalized findings by experiment family
- `failures/index.json`: recurring failure signatures and suggested fixes
- `policies/controller_defaults.json`: controller policy defaults
- `best_runs.json`: promoted best-known runs by family

Controller runs may snapshot and update these files under policy, but volatile run state stays in `controller_runs/`.
