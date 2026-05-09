#!/bin/zsh

set -euo pipefail

# Pipeline outputs are organized per-run under `runs/<RUN_NAME>/...`.
# Override RUN_NAME if you want a stable directory name:
#   RUN_NAME=my_experiment ./run_pipeline.sh
RUN_NAME="${RUN_NAME:-run_kelvin_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/${RUN_NAME}"

MODEL_DIR="${RUN_DIR}/model"
TIMESTEP_DIR="${RUN_DIR}/timesteps"
YAML_DIR="${RUN_DIR}/yaml"
META_DIR="${RUN_DIR}/meta"

mkdir -p "${MODEL_DIR}" "${TIMESTEP_DIR}" "${YAML_DIR}" "${META_DIR}"

# Save a copy of the input lattice definition for reproducibility.
cp lattice.yaml "${META_DIR}/lattice.yaml"

# 1) Generate a lattice FE model from the cylinder-based `lattice.yaml`.
uv run python scripts/generate_lattice_from_yaml.py \
  --yaml lattice.yaml --out "${MODEL_DIR}/out.json" --nx 4 --ny 4 --nz 4 --subdivide 8

# 2) Add indentation boundary conditions / indenter patch to the generated model.
uv run python scripts/apply_indent_boundary_conditions.py \
  --in "${MODEL_DIR}/out.json" --out "${MODEL_DIR}/out.json" \
  --patch-cells-x 2 --patch-cells-y 2 --patch-placement center \
  --indent-uz -0.8 --indenter-uxuy-zero

# 3) Run the simulation and write timestep outputs under `runs/<RUN_NAME>/timesteps/`.
uv run python -m src.main "${MODEL_DIR}/out.json" \
  --ramp-steps 20 \
  --output-prefix "${TIMESTEP_DIR}/${RUN_NAME}" \
  --output-every 2

# 4) Convert timestep JSONs to the renderer's YAML object-collection format.
# r = sqrt(A/pi) tracks the default circular strut (r=0.025 in YAML units), including normalization scale.
uv run python scripts/json_to_yaml.py "${TIMESTEP_DIR}/${RUN_NAME}_t*.json" \
  --radius-from-area \
  --outdir "${YAML_DIR}" \
  --overwrite

echo "Run outputs written to: ${RUN_DIR}"