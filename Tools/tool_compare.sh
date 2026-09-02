#!/usr/bin/env bash
# Compare tool-call streaming responses between slotstream and an external
# OpenAI-compatible endpoint. Default comparison: slotstream vs openrouter
# running the same model call against the same iOS simulator tool schema.
#
# Why this script exists
# ----------------------
# Slotstream's Server.swift parses tool calls out of the model's streamed text
# (Qwen template emits <tool_call>...</tool_call> XML). That parser has been
# observed to drop arguments to {} for some calls — see the calm-mountain
# subagent transcript (152/166 tool calls errored with empty input). To
# distinguish parser bug from model behavior, we send the *same* prompt to:
#
#   - openrouter (well-tested reference, free minimax-m3 model)
#   - slotstream at 127.0.0.1:8000 (qwen3.8-flash-next:4bit)
#
# …and diff. If openrouter always produces a valid call and slotstream
# sometimes produces {}/schema-error, the parser is the problem.
#
# Defaults
# --------
# - Tool set: ios-simulator_* only (the same MCP-registered set opencode uses).
#   Specifically ui_tap, screenshot, ui_describe_all, launch_app — chosen
#   because ui_tap (x:int,y:int,udid:string) is the most-called in the
#   failing transcript and stresses both integer and string typing.
# - Prompt: forces ui_tap with all three parameters, on a coordinate that
#   represents "tap the Search tab". If the parser drops any argument the
#   opencode schema validator would catch it (SchemaError Missing key
#   at ["x"] / ["y"] / ["udid"]).
#
# Usage
# -----
#   Tools/tool_compare.sh                            # default task, both
#   Tools/tool_compare.sh --local                    # slotstream only
#   Tools/tool_compare.sh --openrouter               # openrouter only
#   Tools/tool_compare.sh --prompt "..."             # override prompt
#   Tools/tool_compare.sh --tools tools.json         # custom tool schemas
#   Tools/tool_compare.sh --model openrouterid/x     # override openrouter model
#   Tools/tool_compare.sh --out /tmp/diff            # output dir
#   Tools/tool_compare.sh -h                         # help
#
# Outputs (in $OUT):
#   <endpoint>_raw.txt        verbatim SSE bytes
#   <endpoint>_summary.json   parsed view: content, tool_calls, finish_reason
#   diff.txt                  side-by-side comparison (both mode only)
#
# Exit codes:
#   0   both succeeded (or only one side ran)
#   1   usage error
#   2   openrouter failed
#   3   slotstream failed
#
# Memory safety: this script does NOT load any model. It only fires one HTTP
# request against an already-running slotstream server and one against
# openrouter. Safe to run alongside other slotstream processes.

set -euo pipefail

# --- defaults ----------------------------------------------------------------
ENDPOINT="both"
SLOTSTREAM_URL="${SLOTSTREAM_URL:-http://127.0.0.1:8000}"
OPENROUTER_MODEL="minimax/minimax-m3:free"
MAX_TOKENS=1024
TEMPERATURE=0
OUT="/tmp/slotstream-toolcompare/$(date +%Y%m%d-%H%M%S)"
TIMEOUT_S=120
USER_PROMPT=""
SYSTEM_PROMPT=""
TOOLS_FILE=""

usage() { sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
die() { echo "tool_compare: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) ENDPOINT="local"; shift ;;
    --openrouter) ENDPOINT="openrouter"; shift ;;
    --both) ENDPOINT="both"; shift ;;
    --url) SLOTSTREAM_URL="$2"; shift 2 ;;
    --model) OPENROUTER_MODEL="$2"; shift 2 ;;
    --prompt) USER_PROMPT="$2"; shift 2 ;;
    --system) SYSTEM_PROMPT="$2"; shift 2 ;;
    --tools) TOOLS_FILE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

mkdir -p "$OUT"

# --- tool schemas ------------------------------------------------------------
# Schemas match what opencode registers with the ios-simulator MCP server.
# Only ios-simulator_* tools — no bash, no read, no edit, no general shell.
# Prompt + tool combo forces ui_tap with all three required parameters.
DEFAULT_TOOLS='[
  {
    "type": "function",
    "function": {
      "name": "ios-simulator_get_booted_sim_id",
      "description": "Get the UDID of the currently booted iOS simulator. No parameters.",
      "parameters": {"type": "object", "properties": {}, "required": []}
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ios-simulator_screenshot",
      "description": "Takes a screenshot of the booted iOS simulator and saves it to output_path.",
      "parameters": {
        "type": "object",
        "properties": {
          "output_path": {"type": "string", "description": "Absolute path for the PNG."},
          "udid": {"type": "string", "description": "Simulator UDID. Omit to use the booted simulator."}
        },
        "required": ["output_path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ios-simulator_ui_describe_all",
      "description": "Returns accessibility information for every element on the simulator screen.",
      "parameters": {
        "type": "object",
        "properties": {
          "udid": {"type": "string", "description": "Simulator UDID. Omit to use the booted simulator."}
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ios-simulator_ui_tap",
      "description": "Tap at the given screen coordinates on the booted simulator.",
      "parameters": {
        "type": "object",
        "properties": {
          "x": {"type": "integer", "description": "X coordinate in points."},
          "y": {"type": "integer", "description": "Y coordinate in points."},
          "udid": {"type": "string", "description": "Simulator UDID. Omit to use the booted simulator."}
        },
        "required": ["x", "y", "udid"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ios-simulator_launch_app",
      "description": "Launch an app on the simulator by bundle identifier.",
      "parameters": {
        "type": "object",
        "properties": {
          "bundle_id": {"type": "string", "description": "The bundle identifier of the app to launch, e.g. com.apple.mobilesafari."},
          "udid": {"type": "string", "description": "Simulator UDID. Omit to use the booted simulator."}
        },
        "required": ["bundle_id"]
      }
    }
  }
]'

if [[ -n "$TOOLS_FILE" ]]; then
  [[ -r "$TOOLS_FILE" ]] || die "tools file not readable: $TOOLS_FILE"
  TOOLS_JSON=$(cat "$TOOLS_FILE")
else
  TOOLS_JSON="$DEFAULT_TOOLS"
fi

# --- default prompt ----------------------------------------------------------
# The prompt explicitly requires ui_tap with x, y, and udid — three parameters
# of two types (int, string). Any parser bug that drops parameters will show
# up as an empty arguments object or a schema-error in opencode's validator.
DEFAULT_SYSTEM='You are an iOS simulator assistant. When you call a tool you must pass every required parameter explicitly. Do not omit any parameter.'

DEFAULT_PROMPT='The iOS simulator with UDID 4AFC7324-A9D2-4E80-9A5E-BF8C9B6F1A2D is currently booted. Use ios-simulator_ui_tap to tap the Search tab at coordinates x=140, y=820 on that simulator. You MUST pass all three required parameters (x=140, y=820, udid="4AFC7324-A9D2-4E80-9A5E-BF8C9B6F1A2D") in a single call to ios-simulator_ui_tap. Do not call any other tool. Do not ask any clarifying questions.'

[[ -z "$USER_PROMPT" ]] && USER_PROMPT="$DEFAULT_PROMPT"
[[ -z "$SYSTEM_PROMPT" ]] && SYSTEM_PROMPT="$DEFAULT_SYSTEM"

# --- payload builder ---------------------------------------------------------
# We use Python so the JSON is well-formed regardless of shell quoting. The
# prompt/system/tools come from heredoc-safe substitutions below.

build_payload() {
  local model="$1"
  USER_PROMPT="$USER_PROMPT" SYSTEM_PROMPT="$SYSTEM_PROMPT" \
    TOOLS_JSON="$TOOLS_JSON" MODEL="$model" \
    MAX_TOKENS="$MAX_TOKENS" TEMPERATURE="$TEMPERATURE" \
  python3 <<'PY'
import json, os
messages = []
if os.environ.get('SYSTEM_PROMPT'):
    messages.append({'role': 'system', 'content': os.environ['SYSTEM_PROMPT']})
messages.append({'role': 'user', 'content': os.environ['USER_PROMPT']})
payload = {
    'model': os.environ['MODEL'],
    'stream': True,
    'messages': messages,
    'tools': json.loads(os.environ['TOOLS_JSON']),
    'tool_choice': 'auto',
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': float(os.environ['TEMPERATURE']),
}
print(json.dumps(payload))
PY
}

# --- send one request --------------------------------------------------------

run_one() {
  local name="$1" url="$2" model="$3" out_raw="$4" out_summary="$5"
  local payload
  payload=$(build_payload "$model")

  echo "→ $name  $url"
  echo "  model:  $model"
  echo "  prompt: $(echo "$USER_PROMPT" | head -c 100)…"

  local t0 t1
  t0=$(date +%s)

  # Capture raw bytes AND a one-line status by writing a temp file curl can't
  # clobber. We do two curls (one for status, one for body) so we get timing
  # plus the raw SSE without curl clobbering the body with the format string.
  local code
  if [[ "$name" == "openrouter" ]]; then
    local key
    key=$(python3 -c "
import re
print(re.search(r'\"apiKey\":\s*\"([^\"]+)\"', open('/Users/sub/.config/opencode/opencode.jsonc').read()).group(1))
")
    code=$(curl -sN -m "$TIMEOUT_S" \
      -H "Authorization: Bearer $key" \
      -H "Content-Type: application/json" \
      -H "Accept: text/event-stream" \
      -o "$out_raw" \
      -w '%{http_code}' \
      -d "$payload" \
      "$url" 2>/dev/null || echo "000")
  else
    code=$(curl -sN -m "$TIMEOUT_S" \
      -H "Content-Type: application/json" \
      -H "Accept: text/event-stream" \
      -o "$out_raw" \
      -w '%{http_code}' \
      -d "$payload" \
      "$url" 2>/dev/null || echo "000")
  fi
  t1=$(date +%s)
  local size
  size=$(wc -c < "$out_raw" | tr -d ' ')

  echo "  status: $code   bytes: $size   wall: $((t1-t0))s   raw: $out_raw"

  if [[ "$code" != "200" ]] || [[ "$size" -lt 10 ]]; then
    echo "  request failed or returned empty body" >&2
    return 1
  fi
}

# --- parser summary ----------------------------------------------------------
# Re-stream the raw SSE bytes into a flat structure: full content text, list
# of (index,id,name,arguments_raw,arguments_parsed_or_error).

summarize() {
  local name="$1" raw="$2" summary="$3"
  python3 - "$raw" "$summary" "$name" <<'PY'
import json, sys
from pathlib import Path
raw_path, summary_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
text = Path(raw_path).read_text(errors='replace')
records = []
for line in text.splitlines():
    if not line.startswith('data:'): continue
    body = line[5:].strip()
    if body == '[DONE]':
        records.append({'__type__': 'done'})
        continue
    try:
        records.append(json.loads(body))
    except Exception:
        records.append({'__type__': 'unparseable', 'raw': body[:200]})
content = ''
tool_calls = {}  # idx -> {id, type, name, arguments_raw}
finish_reason = None
usage = None
unparseable_count = 0
for r in records:
    if r.get('__type__') == 'done':
        continue
    if r.get('__type__') == 'unparseable':
        unparseable_count += 1
        continue
    ch = (r.get('choices') or [{}])[0]
    delta = ch.get('delta', {})
    if 'content' in delta and delta['content']:
        content += delta['content']
    if 'tool_calls' in delta and delta['tool_calls']:
        for tc in delta['tool_calls']:
            idx = tc.get('index', 0)
            tc_e = tool_calls.setdefault(idx, {
                'id': tc.get('id'),
                'type': tc.get('type'),
                'name': None,
                'arguments_raw': '',
            })
            if 'id' in tc and tc['id'] is not None:
                tc_e['id'] = tc['id']
            if 'type' in tc and tc['type'] is not None:
                tc_e['type'] = tc['type']
            fn = tc.get('function') or {}
            if 'name' in fn and fn['name'] is not None:
                tc_e['name'] = fn['name']
            if 'arguments' in fn and fn['arguments'] is not None:
                tc_e['arguments_raw'] += fn['arguments']
    if 'finish_reason' in ch and ch['finish_reason']:
        finish_reason = ch['finish_reason']
    if 'usage' in r:
        usage = r['usage']

# Try to parse arguments as JSON for each tool call.
parsed_calls = []
for idx in sorted(tool_calls):
    tc = tool_calls[idx]
    args_str = tc['arguments_raw']
    parsed = None
    parse_error = None
    if args_str.strip():
        try:
            parsed = json.loads(args_str)
        except Exception as e:
            parse_error = str(e)
    elif tc['name']:
        # Empty args but a name was set — this is the parser-bug signature.
        parse_error = 'arguments string was empty'
    parsed_calls.append({
        'index': idx,
        'id': tc['id'],
        'type': tc['type'],
        'name': tc['name'],
        'arguments_raw': args_str,
        'arguments': parsed,
        'arguments_parse_error': parse_error,
    })

summary = {
    'endpoint': name,
    'raw_bytes': len(text),
    'unparseable_chunks': unparseable_count,
    'finish_reason': finish_reason,
    'content_text': content,
    'content_chars': len(content),
    'tool_calls': parsed_calls,
    'tool_call_count': len(parsed_calls),
    'usage': usage,
}
Path(summary_path).write_text(json.dumps(summary, indent=2))
# Print a one-screen summary
print(f"  finish_reason: {finish_reason}")
print(f"  content chars: {len(content)}  preview={content[:100]!r}")
print(f"  tool_calls: {len(parsed_calls)}")
for tc in parsed_calls:
    sigil = 'ok'
    if tc['arguments'] is None:
        if tc['arguments_parse_error'] == 'arguments string was empty':
            sigil = 'EMPTY args string (parser dropped the params)'
        elif tc['arguments_parse_error']:
            sigil = f'parse err: {tc["arguments_parse_error"][:60]}'
    # arguments == {} is fine for tools that genuinely have no required params;
    # that's a schema-property, not a parser signal. Don't flag.
    if tc['arguments'] and isinstance(tc['arguments'], dict):
        coerced = [f"{k}={v!r}" for k, v in tc['arguments'].items()
                   if isinstance(v, str) and v.lstrip('-').isdigit()]
        if coerced:
            sigil = f'⚠️  TYPE-COERCION (string instead of int): {", ".join(coerced)}'
    print(f"    [{tc['index']}] {tc['name']}  id={tc['id']!r}  args={tc['arguments_raw'][:120]!r}  parsed={tc['arguments']}  {sigil}")
if usage:
    print(f"  usage: {usage}")
PY
}

# --- diff --------------------------------------------------------------------

diff_summaries() {
  python3 - "$OUT/openrouter_summary.json" "$OUT/slotstream_summary.json" <<'PY'
import json, sys
from pathlib import Path
o = json.load(open(sys.argv[1]))
l = json.load(open(sys.argv[2]))
def sig(tc):
    return (tc['name'], tc['arguments_raw'])
o_calls = [sig(tc) for tc in o['tool_calls']]
l_calls = [sig(tc) for tc in l['tool_calls']]
print()
print('=' * 78)
print('SIDE-BY-SIDE (openrouter  vs  slotstream)')
print('=' * 78)
print(f"content chars:    openrouter={o['content_chars']:<6}  slotstream={l['content_chars']}")
print(f"finish_reason:    openrouter={o['finish_reason']!r:<22}  slotstream={l['finish_reason']!r}")
print(f"tool_call count:  openrouter={o['tool_call_count']:<6}  slotstream={l['tool_call_count']}")
print(f"unparseable SSE:  openrouter={o['unparseable_chunks']:<6}  slotstream={l['unparseable_chunks']}")
print(f"raw bytes:        openrouter={o['raw_bytes']:<6}  slotstream={l['raw_bytes']}")
print(f"usage:            openrouter={o.get('usage')}")
print(f"                  slotstream={l.get('usage')}")
print()
print('openrouter tool_calls:')
for tc in o['tool_calls']:
    print(f"  [{tc['index']}] {tc['name']}  id={tc['id']!r}  args={tc['arguments_raw'][:120]!r}")
print('slotstream tool_calls:')
for tc in l['tool_calls']:
    print(f"  [{tc['index']}] {tc['name']}  id={tc['id']!r}  args={tc['arguments_raw'][:120]!r}")
# Highlight differences
print()
print('verdict:')
def has_empty(tc):
    return tc['arguments'] is None and tc['arguments_parse_error'] == 'arguments string was empty'
o_empty = [tc for tc in o['tool_calls'] if has_empty(tc)]
l_empty = [tc for tc in l['tool_calls'] if has_empty(tc)]
o_sparse = [tc for tc in o['tool_calls'] if tc['arguments'] == {}]
l_sparse = [tc for tc in l['tool_calls'] if tc['arguments'] == {}]
if not o_empty and not l_empty and not o_sparse and not l_sparse:
    print("  both backends produced non-empty arguments.")
elif o_sparse or l_sparse:
    if o_sparse and not l_sparse:
        print(f"  openrouter returned {{}} for {len(o_sparse)} call(s); slotstream non-empty")
    elif l_sparse and not o_sparse:
        print(f"  slotstream returned {{}} for {len(l_sparse)} call(s); openrouter non-empty")
        for tc in l_sparse:
            print(f"      → {tc['name']} args={tc['arguments_raw'][:120]!r}")
    else:
        print(f"  both backends returned {{}} for {len(o_sparse)}/{len(l_sparse)} call(s)")
if o_empty or l_empty:
    print()
    print("EMPTY-ARGS failures:")
    if o_empty:
        print(f"  openrouter ({len(o_empty)}):")
        for tc in o_empty:
            print(f"      → {tc['name']} args={tc['arguments_raw'][:120]!r}")
    if l_empty:
        print(f"  slotstream ({len(l_empty)}):")
        for tc in l_empty:
            print(f"      → {tc['name']} args={tc['arguments_raw'][:120]!r}")

# Type-coercion diff: which arguments came back as strings instead of numbers?
def type_mismatches(tc):
    if tc['arguments'] is None:
        return []
    bad = []
    for k, v in tc['arguments'].items():
        if isinstance(v, str) and v.lstrip('-').isdigit():
            bad.append(f"{k}={v!r} (str, should be int)")
    return bad
o_coerced = [(tc['name'], tc['arguments']) for tc in o['tool_calls'] if type_mismatches(tc)]
l_coerced = [(tc['name'], tc['arguments']) for tc in l['tool_calls'] if type_mismatches(tc)]
if o_coerced or l_coerced:
    print()
    print("TYPE-COERCION: arguments emitted as strings that look numeric")
    for name, args in o_coerced:
        bad = type_mismatches({'arguments': args})
        print(f"  openrouter {name}: {bad}")
    for name, args in l_coerced:
        bad = type_mismatches({'arguments': args})
        print(f"  slotstream {name}: {bad}")
PY
}

# --- main --------------------------------------------------------------------

echo "output: $OUT"
echo "mode:   $ENDPOINT"
echo "model:  $OPENROUTER_MODEL (openrouter) vs qwen3.8-flash-next:4bit (slotstream)"
echo

rc=0

if [[ "$ENDPOINT" == "openrouter" || "$ENDPOINT" == "both" ]]; then
  if run_one "openrouter" "https://openrouter.ai/api/v1/chat/completions" \
             "$OPENROUTER_MODEL" "$OUT/openrouter_raw.txt" "$OUT/openrouter_summary.json"; then
    summarize "openrouter" "$OUT/openrouter_raw.txt" "$OUT/openrouter_summary.json"
  else
    rc=2
  fi
  echo
fi

if [[ "$ENDPOINT" == "local" || "$ENDPOINT" == "both" ]]; then
  if run_one "slotstream" "${SLOTSTREAM_URL}/v1/chat/completions" \
             "qwen3.8-flash-next:4bit" "$OUT/slotstream_raw.txt" "$OUT/slotstream_summary.json"; then
    summarize "slotstream" "$OUT/slotstream_raw.txt" "$OUT/slotstream_summary.json"
  else
    [[ $rc -eq 0 ]] && rc=3
  fi
  echo
fi

if [[ "$ENDPOINT" == "both" ]]; then
  diff_summaries | tee "$OUT/diff.txt"
fi

exit $rc