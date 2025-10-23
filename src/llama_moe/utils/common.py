from typing import Any, Mapping, Optional, Tuple

def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{int(s)} {u}" if u == "B" else f"{s:.2f} {u}"
        s /= 1024.0

def pretty(val: Any, max_len: int = 200) -> str:
    s = val if isinstance(val, str) else repr(val)
    return s if len(s) <= max_len else s[:max_len] + f"... (len={len(s)})"