#!/usr/bin/env bash
set -euo pipefail

# Train multiple torch LAN models (for example angle and ddm) in one command.
# Uses uv + torchtrain and writes outputs into the chosen networks path.

CONFIG_TEMPLATE="src/lanfactory/cli/config_network_training_lan.yaml"
TRAINING_DATA_FOLDER=""
TRAINING_DATA_BASE=""
NETWORKS_PATH_BASE="data/torch_models"
MODELS="angle,ddm"
NETWORK_IDS="0,1,2"
DL_WORKERS="1"
LOG_LEVEL="INFO"
DRY_RUN="0"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_torch_models_batch.sh [options]

Options:
  --config-template PATH       YAML template with a MODEL field.
                               Default: src/lanfactory/cli/config_network_training_lan.yaml
  --training-data-folder PATH  Shared training data folder used for all models.
  --training-data-base PATH    Base folder where each model has its own subfolder.
                               Example: <base>/angle and <base>/ddm
  --networks-path-base PATH    Base output path for trained models.
                               Default: data/torch_models
  --models CSV                 Comma-separated model names.
                               Default: angle,ddm
  --network-ids CSV            Comma-separated network IDs.
                               Default: 0,1,2
  --dl-workers N               DataLoader workers passed to torchtrain.
                               Default: 1
  --log-level LEVEL            Logging level for torchtrain.
                               Default: INFO
  --dry-run                    Validate commands without training.
  --help                       Show this help message.

Examples:
  scripts/train_torch_models_batch.sh \
    --training-data-base data/data \
    --networks-path-base data/torch_models \
    --models angle,ddm \
    --network-ids 0,1,2,3

  scripts/train_torch_models_batch.sh \
    --training-data-folder data/data/angle \
    --models angle \
    --network-ids 0,1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-template)
            CONFIG_TEMPLATE="$2"
            shift 2
            ;;
        --training-data-folder)
            TRAINING_DATA_FOLDER="$2"
            shift 2
            ;;
        --training-data-base)
            TRAINING_DATA_BASE="$2"
            shift 2
            ;;
        --networks-path-base)
            NETWORKS_PATH_BASE="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --network-ids)
            NETWORK_IDS="$2"
            shift 2
            ;;
        --dl-workers)
            DL_WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
    echo "Config template not found: $CONFIG_TEMPLATE" >&2
    exit 1
fi

if [[ -z "$TRAINING_DATA_FOLDER" && -z "$TRAINING_DATA_BASE" ]]; then
    echo "Provide either --training-data-folder or --training-data-base" >&2
    exit 1
fi

IFS=',' read -r -a MODELS_ARR <<< "$MODELS"
IFS=',' read -r -a IDS_ARR <<< "$NETWORK_IDS"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

make_model_config() {
    local template="$1"
    local model="$2"
    local out_file="$3"

    # Replace MODEL: ... in YAML while preserving all other lines.
    awk -v model="$model" '
        BEGIN { replaced = 0 }
        /^[[:space:]]*MODEL[[:space:]]*:/ && replaced == 0 {
            print "MODEL: \"" model "\""
            replaced = 1
            next
        }
        { print }
        END {
            if (replaced == 0) {
                print "MODEL: \"" model "\""
            }
        }
    ' "$template" > "$out_file"
}

resolve_model_training_folder() {
    local candidate_folder="$1"
    local model="$2"

    if [[ ! -d "$candidate_folder" ]]; then
        echo ""
        return
    fi

    # Preferred: pickle shards directly in the provided folder.
    if find "$candidate_folder" -maxdepth 1 -type f -name '*.pickle' | grep -q .; then
        echo "$candidate_folder"
        return
    fi

    # Common layout: one subfolder per model.
    if [[ -d "$candidate_folder/$model" ]] && find "$candidate_folder/$model" -maxdepth 1 -type f -name '*.pickle' | grep -q .; then
        echo "$candidate_folder/$model"
        return
    fi

    # Fallback: if exactly one immediate subfolder contains pickle shards, use it.
    local subdir_count
    subdir_count="$(find "$candidate_folder" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    if [[ "$subdir_count" == "1" ]]; then
        local only_subdir
        only_subdir="$(find "$candidate_folder" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
        if find "$only_subdir" -maxdepth 1 -type f -name '*.pickle' | grep -q .; then
            echo "$only_subdir"
            return
        fi
    fi

    echo "$candidate_folder"
}

validate_training_folder() {
    local folder="$1"
    local model="$2"

    local n_pickles
    n_pickles="$(find "$folder" -maxdepth 1 -type f -name '*.pickle' | wc -l | tr -d ' ')"

    if [[ "$n_pickles" == "0" ]]; then
        echo "No .pickle files found for model '$model' in: $folder" >&2
        echo "Expected either:" >&2
        echo "  1) <folder>/*.pickle" >&2
        echo "  2) <folder>/$model/*.pickle" >&2
        exit 1
    fi

    local validation_out
    if ! validation_out="$((uv run python - "$folder" <<'PY'
import glob
import os
import pickle
import sys

folder = sys.argv[1]
files = sorted(glob.glob(os.path.join(folder, "*.pickle")))
if not files:
    print("ERR_NO_PICKLES")
    raise SystemExit(2)

with open(files[0], "rb") as f:
    obj = pickle.load(f)

if not isinstance(obj, dict):
    print("ERR_NOT_DICT")
    raise SystemExit(3)

keys = set(obj.keys())
required = {"lan_data", "lan_labels"}
if not required.issubset(keys):
    print("ERR_BAD_KEYS")
    print(",".join(sorted(keys)))
    raise SystemExit(4)

print("OK_KEYS")
PY
))"; then
        if grep -q "ERR_BAD_KEYS" <<<"$validation_out"; then
            echo "Training data format mismatch for model '$model' in: $folder" >&2
            echo "Expected pickle keys: lan_data and lan_labels" >&2
            echo "Found keys:" >&2
            echo "$(tail -n 1 <<<"$validation_out")" >&2
            exit 1
        fi

        if grep -q "ERR_NOT_DICT" <<<"$validation_out"; then
            echo "Unexpected pickle format in: $folder" >&2
            echo "Expected each shard to be a dict with lan_data/lan_labels." >&2
            exit 1
        fi

        echo "Failed to validate training folder '$folder'." >&2
        echo "$validation_out" >&2
        exit 1
    fi
}

for model in "${MODELS_ARR[@]}"; do
    model="${model//[[:space:]]/}"
    if [[ -z "$model" ]]; then
        continue
    fi

    if [[ -n "$TRAINING_DATA_FOLDER" ]]; then
        model_training_folder="$(resolve_model_training_folder "$TRAINING_DATA_FOLDER" "$model")"
    else
        model_training_folder="$(resolve_model_training_folder "$TRAINING_DATA_BASE/$model" "$model")"
    fi

    if [[ ! -d "$model_training_folder" ]]; then
        echo "Training data folder not found for model '$model': $model_training_folder" >&2
        exit 1
    fi

    validate_training_folder "$model_training_folder" "$model"

    model_cfg="$TMP_DIR/config_${model}.yaml"
    make_model_config "$CONFIG_TEMPLATE" "$model" "$model_cfg"

    for net_id in "${IDS_ARR[@]}"; do
        net_id="${net_id//[[:space:]]/}"
        if [[ -z "$net_id" ]]; then
            continue
        fi

        cmd=(
            uv run torchtrain
            --config-path "$model_cfg"
            --training-data-folder "$model_training_folder"
            --networks-path-base "$NETWORKS_PATH_BASE"
            --network-id "$net_id"
            --dl-workers "$DL_WORKERS"
            --log-level "$LOG_LEVEL"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            cmd+=(--dry-run)
        fi

        echo ""
        echo "=== Training model=$model network_id=$net_id ==="
        echo "Training data: $model_training_folder"
        "${cmd[@]}"
    done
done

echo ""
echo "Batch training complete."
