#!/usr/bin/env python3
"""Manage marimo session state at ~/.marimo-sessions.json"""
import json
import os
import sys

STATE_FILE = os.path.expanduser("~/.marimo-sessions.json")


def load() -> list[dict]:
    if not os.path.exists(STATE_FILE):
        return []
    with open(STATE_FILE) as f:
        return json.load(f)


def save(sessions: list[dict]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def clean(sessions: list[dict]) -> list[dict]:
    return [s for s in sessions if is_alive(s["pid"])]


def add(pid: int, port: int, token: str, path: str, host: str) -> None:
    sessions = clean(load())
    url = f"http://{host}:{port}?access_token={token}"
    sessions.append({"pid": pid, "port": port, "token": token, "path": path, "url": url})
    save(sessions)


def remove_by_pid(pid: int) -> dict | None:
    sessions = load()
    match = next((s for s in sessions if s["pid"] == pid), None)
    save([s for s in sessions if s["pid"] != pid])
    return match


def remove_by_path(path: str) -> list[dict]:
    sessions = load()
    abs_path = os.path.abspath(path)
    matches = [s for s in sessions if s["path"] == abs_path or abs_path in s["path"] or s["path"] in abs_path]
    save([s for s in sessions if s not in matches])
    return matches


def list_sessions() -> list[dict]:
    sessions = clean(load())
    save(sessions)
    return sessions


def clear_all() -> list[dict]:
    sessions = clean(load())
    save([])
    return sessions


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        for s in list_sessions():
            print(f"{s['pid']}\t{s['port']}\t{s['path']}\t{s['url']}")

    elif cmd == "add":
        # add <pid> <port> <token> <path> <host>
        add(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])

    elif cmd == "remove-pid":
        m = remove_by_pid(int(sys.argv[2]))
        if m:
            print(f"removed\t{m['pid']}\t{m['path']}")

    elif cmd == "remove-path":
        for m in remove_by_path(sys.argv[2]):
            print(f"removed\t{m['pid']}\t{m['path']}")

    elif cmd == "clear":
        for s in clear_all():
            print(f"stopped\t{s['pid']}\t{s['path']}")
