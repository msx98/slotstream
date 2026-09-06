#!/bin/bash

# Runs the model in DS4 mode on the query "what is the capital of france"
# Should respond with "paris"
# Logs are in output.log

LOG_DIR=".logs"
CURRENT_TIME=$(date "+%Y%m%d%H%M%S")
mkdir -p "$LOG_DIR"
OUTPUT_PATH="${LOG_DIR}/${CURRENT_TIME}.log"
PID_PATH="${LOG_DIR}/${CURRENT_TIME}.pid"

export SLOTSTREAM_ROUTER_TRACE=1
export SLOTSTREAM_SWEEP_TRACE=1
export SLOTSTREAM_MEM_TRACE=1
export SLOTSTREAM_DS4_TRACE=1
export SLOTSTREAM_DS4_PROF=1

if [ -e ".logs/output.pid" ]; then kill -9 $(cat .logs/output.pid); rm .logs/output.pid; fi;
pkill -9 -f slotstream

nohup .build/debug/slotstream run \
    --model /opt/common/models/text/antirez/deepseek-v4-gguf \
    --pool-floor-gb 0.3 --memory-gb 12 \
    --max-tokens 4 \
    --prompt "What is the capital of France? Respond with exactly one word:" \
    > "${OUTPUT_PATH}" 2>&1 & 1>/dev/null 2>/dev/null

# Capture the exact PID of the background pipeline
pid=$!
echo $pid > "${PID_PATH}"

if [ -e ".logs/output.log" ]; then rm .logs/output.log; fi; ln -s "$OUTPUT_PATH" .logs/output.log
ln -s "$PID_PATH" .logs/output.pid

echo "PID: ${pid}. Logs saved to: ${OUTPUT_PATH}"
