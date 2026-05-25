#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rl_root="$repo_root/RL"

export PYTHONPATH="$rl_root/src${PYTHONPATH:+:$PYTHONPATH}"

configs_dir="$rl_root/configs_to_test"
active_config="$rl_root/configs/config_ppo.yaml"
logs_dir="$rl_root/logs"
train_logs="$logs_dir/ppo_train"

rm -rf "$train_logs" "$logs_dir"/ppo_train_*

shopt -s nullglob
configs=("$configs_dir"/*.yaml)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
	echo "No config files found in $configs_dir"
	exit 1
fi

iter=1
for cfg in "${configs[@]}"; do
	cp "$cfg" "$active_config"

	(python3 "$rl_root/scripts/ppo_train.py")

	if [[ -d "$train_logs" ]]; then
		mv "$train_logs" "$logs_dir/ppo_train_${iter}"
	else
		echo "Expected $train_logs to exist after training"
	fi

	iter=$((iter + 1))
done
