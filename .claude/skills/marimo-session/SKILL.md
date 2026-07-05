---
name: marimo-session
description: Manage detached marimo read-only notebook server processes. Subcommands: create <path> (launch a detached server), ls (list running sessions), stop <pid_or_path> (stop one session), stop-all (stop all sessions). Use when the user asks to "run a marimo notebook", "start a marimo server", "list marimo sessions", "stop marimo", or "stop all marimo".
argument-hint: "<create <path> | ls | stop <pid_or_path> | stop-all>"
---

Manage detached marimo read-only notebook server processes.

The helper script lives at `${PWD}/.claude/skills/marimo-session/scripts/sessions.py` and tracks session state in `~/.marimo-sessions.json`.

Read `$ARGUMENTS[0]` to determine the subcommand. Work through the matching section below.

---

## Subcommand: `ls`

List all currently running marimo sessions.

### Step 1 — Query Sessions

```bash
python3 ${PWD}/.claude/skills/marimo-session/scripts/sessions.py list
```

### Step 2 — Report

If the output is empty, report: "No marimo sessions are currently running."

Otherwise display a table with columns: PID | Port | Notebook | URL

---

## Subcommand: `create`

Launch a detached read-only marimo server for a notebook file.

`$ARGUMENTS[1]` is the path to the notebook file.

### Step 1 — Resolve Path

```bash
realpath "$ARGUMENTS[1]"
```

If the file does not exist, report an error and stop.

### Step 2 — Pick a Free Port

Find a free port in the range 7000–8999:

```bash
python3 -c "
import socket, random
for _ in range(100):
    p = random.randint(7000, 8999)
    try:
        s = socket.socket(); s.bind(('', p)); s.close(); print(p); break
    except OSError:
        pass
"
```

Store the result as `PORT`.

### Step 3 — Generate a Token

```bash
openssl rand -hex 16
```

Store the result as `TOKEN`.

### Step 4 — Launch Detached Server

```bash
nohup env/bin/marimo run "$ARGUMENTS[1]" \
  --host 0.0.0.0 \
  --port $PORT \
  --token-password $TOKEN \
  --headless \
  > /tmp/marimo-$PORT.log 2>&1 &
echo $!
```

Store the printed value as `PID`.

Wait 2 seconds for the server to start, then verify the process is alive:

```bash
kill -0 $PID 2>/dev/null && echo alive || echo dead
```

If it prints `dead`, show the last 20 lines of `/tmp/marimo-$PORT.log` and report failure.

### Step 5 — Register Session

```bash
NOTEBOOK_PATH=$(realpath "$ARGUMENTS[1]")
python3 ${PWD}/.claude/skills/marimo-session/scripts/sessions.py add $PID $PORT $TOKEN "$NOTEBOOK_PATH" 0.0.0.0
```

### Step 6 — Report

Report to the user:

- **Notebook**: the resolved path
- **PID**: the process ID
- **Port**: the port number
- **URL**: `http://0.0.0.0:$PORT?access_token=$TOKEN`
- **Log**: `/tmp/marimo-$PORT.log`
- Note that `0.0.0.0` should be replaced with the machine's actual IP or hostname when accessing from another machine.

---

## Subcommand: `stop`

Stop a single marimo session by PID or notebook path.

`$ARGUMENTS[1]` is either a numeric PID or a file path (full or partial).

### Step 1 — Determine Type

If `$ARGUMENTS[1]` is numeric, use the PID path. Otherwise use the path path.

**PID path:**

```bash
python3 ${PWD}/.claude/skills/marimo-session/scripts/sessions.py remove-pid $ARGUMENTS[1]
```

If output contains `removed`, kill the process:

```bash
kill $ARGUMENTS[1] 2>/dev/null || true
```

**Path path:**

```bash
python3 ${PWD}/.claude/skills/marimo-session/scripts/sessions.py remove-path "$ARGUMENTS[1]"
```

For each line of output that starts with `removed`, extract the PID (second field) and kill it:

```bash
# For each removed PID from remove-path output:
kill <PID> 2>/dev/null || true
```

### Step 2 — Report

Report which sessions were stopped (PID + path), or "No matching session found." if nothing was removed.

---

## Subcommand: `stop-all`

Stop all running marimo sessions.

### Step 1 — Collect and Remove All Sessions

```bash
python3 ${PWD}/.claude/skills/marimo-session/scripts/sessions.py clear
```

Save the output lines. Each line is: `stopped <PID> <path>`

### Step 2 — Kill All Registered Processes

For each PID from the output above:

```bash
kill <PID> 2>/dev/null || true
```

Then sweep for any stragglers:

```bash
pkill -f "marimo run" 2>/dev/null || true
```

### Step 3 — Report

Report the list of stopped sessions (PID + path), or "No sessions were running." if none.

---

## Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "marimo-session"
```
