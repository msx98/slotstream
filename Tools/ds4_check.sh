#!/bin/bash
# DeepSeek-V4-Flash (DS4) gate. Weights-free by default: the math self-tests
# and the tokenizer round-trip read the GGUF header only and run on any
# machine. The 1-request server smoke at the end is the only part that loads
# the 156 GB checkpoint, and it only fires when the weights are present, no
# other model process is running, and the machine can hold the DS4 floor with
# headroom to spare.
set -u
cd "$(dirname "$0")/.."
BIN=.build/release/slotstream
PASS=0; FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then echo "PASS  $1"; PASS=$((PASS+1)); else echo "FAIL  $1"; FAIL=$((FAIL+1)); fi }

SPID=""
cleanup() {
  if [ -n "$SPID" ]; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "== build =="
make build >/dev/null

# The model dir, first hit wins (the /opt tree keeps the reference GGUF).
MODEL_DIR=""
for d in /opt/common/models/text/antirez/deepseek-v4-gguf \
         models/deepseek-v4-gguf \
         "$HOME/.slotstream/models/deepseek-v4-gguf"; do
  if [ -d "$d" ] && "$BIN" ds4-check --model "$d" >/dev/null 2>&1; then MODEL_DIR="$d"; break; fi
done

echo "== DS4 math self-tests + tokenizer round trip (GGUF header only) =="
if [ -n "$MODEL_DIR" ]; then
  check "ds4-check on the real dir (DS4SelfTest.runAll + DS4TokenizerSmoke)" \
        "$BIN ds4-check --model $MODEL_DIR"
else
  check "ds4-check (any deepseek4 GGUF on this machine)" "$BIN ds4-check"
fi

echo "== DS4 config validation + planner (header only) =="
if [ -n "$MODEL_DIR" ]; then
  check "doctor validates the real DS4 GGUF header + geometry" \
        "$BIN doctor --model $MODEL_DIR --memory-gb 15.5"
else
  echo "SKIP  doctor on a DS4 dir (no deepseek4 GGUF found)"
fi

# ---- the smoke below loads the 8.8 GB trunk + 3.4 GB floor pool. The DS4
# floor is its trunk, not the Qwen 8.1-10 GB convention: the planner refuses
# anything under 15.5 GB (11.0 fixed footprint + 3.4 GB floor cache + margin).
# Never raise it here without a measured process-RSS peak first.
SMOKE_MEMORY=15.5
if [ -z "$MODEL_DIR" ]; then
  echo "SKIP  DS4 server smoke (no deepseek4 GGUF found)"
  echo
  echo "passed $PASS, failed $FAIL"
  [ $FAIL -eq 0 ]
  exit $?
fi

# One model process at a time, machine-wide (the lock is per-user).
MODEL_LOCK="/tmp/slotstream-model-$(id -u).lock"
if [ -e "$MODEL_LOCK" ] && lsof -t "$MODEL_LOCK" >/dev/null 2>&1; then
  echo "SKIP  DS4 server smoke (another Slotstream model process holds the lock)"
  echo
  echo "passed $PASS, failed $FAIL"
  [ $FAIL -eq 0 ]
  exit $?
fi

AVAIL_GB=$("$BIN" doctor --model "$MODEL_DIR" --json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("device_available_gb", 0))' \
  2>/dev/null || echo 0)
NEED_GB=$(awk "BEGIN{print $SMOKE_MEMORY + 2}")
if [ "$(awk "BEGIN{print ($AVAIL_GB < $NEED_GB)}")" = "1" ]; then
  echo "SKIP  DS4 server smoke (only ${AVAIL_GB} GB reclaimable, needs ${NEED_GB})"
  echo
  echo "passed $PASS, failed $FAIL"
  [ $FAIL -eq 0 ]
  exit $?
fi

echo "== DS4 server smoke (1 greedy request at --memory-gb $SMOKE_MEMORY) =="
PORT=11469
"$BIN" serve --model "$MODEL_DIR" --memory-gb $SMOKE_MEMORY --port $PORT \
  >/tmp/ssv_ds4_serve.log 2>&1 &
SPID=$!
UP=0
for _ in $(seq 1 240); do
  if printf 'GET /api/version HTTP/1.0\r\n\r\n' | nc -w 2 127.0.0.1 $PORT 2>/dev/null | grep -q 200; then
    UP=1; break
  fi
  sleep 2
done
check "serve answers /api/version over a raw socket" "[ $UP = 1 ]"

BODY='{"model":"ds4","messages":[{"role":"user","content":"Say hi"}],"stream":false,"options":{"temperature":0,"num_predict":12}}'
RESP=$(printf 'POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' \
  "$(printf %s "$BODY" | wc -c | tr -d ' ')" "$BODY" | nc -w 240 127.0.0.1 $PORT 2>/dev/null)
printf '%s' "$RESP" | sed 's/^[^{]*//' > /tmp/ssv_ds4_chat.json
check "greedy /api/chat returns a 12-token message" \
  "python3 -c 'import json; d=json.loads(open(\"/tmp/ssv_ds4_chat.json\").read().strip().splitlines()[-1]); m=d[\"message\"][\"content\"]; assert len(m)>0 and d[\"done_reason\"] in (\"stop\",\"length\"), d'"

kill "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
SPID=""

echo
echo "passed $PASS, failed $FAIL"
[ $FAIL -eq 0 ]
