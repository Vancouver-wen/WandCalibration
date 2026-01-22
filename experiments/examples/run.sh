export PYTHONPATH="${PYTHONPATH}:../.."

mkdir -p logs

LOG_FILE="./logs/$(date +'%Y%m%d_%H%M%S').log"

python ../../main.py --config ./config.yaml 2>&1 | tee "$LOG_FILE"