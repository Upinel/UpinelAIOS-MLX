#!/usr/bin/env python3
"""
Real-time dashboard for the UpinelAIOS server.

  ./status.sh                 live dashboard, 1s refresh, Ctrl-C to exit
  ./status.sh --once          one-shot summary (scripts, logs)
  ./status.sh --json          machine-readable snapshot
  ./status.sh --interval 2    slower refresh
  ./status.sh --power         add ANE/GPU power via sudo powermetrics

Where each number comes from, and what it honestly means:

  CPU     host_statistics(HOST_CPU_LOAD_INFO) tick deltas. Exact, no sudo.
  GPU     IOKit IOAccelerator "Device Utilization %" via ioreg. This is the
          GPU's own hardware busy counter, readable without sudo.
  ANE     Apple does not expose the Neural Engine without powermetrics, which
          needs root. MLX is GPU-only in any case, so the ANE is genuinely
          idle for this workload -- it is shown as "--" rather than faked.
          With --power (or passwordless sudo) the real figure is read.
  RAM     vm_stat pages, sysctl for swap. Wired memory is what matters here:
          it is the MLX allocation and macOS cannot reclaim it.
  SERVER  MTPLX /health, /admin/sessions and /metrics.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import deque

# ── terminal ─────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
# True when we can draw in place with cursor control. We deliberately do
# NOT switch to the alternate screen buffer: the frame is sized to fit the
# window exactly, so leaving the previous shell output in scrollback is a
# feature, not pollution.
TTY_MODE = sys.stdout.isatty()


def c(code):
    return code if USE_COLOR else ""


RESET, BOLD, DIM = c("\033[0m"), c("\033[1m"), c("\033[2m")
RED, GREEN, YELLOW, BLUE, CYAN, MAGENTA = (
    c("\033[31m"), c("\033[32m"), c("\033[33m"),
    c("\033[34m"), c("\033[36m"), c("\033[35m"))
HOME, CLR_EOL, CLR_EOS = "\033[H", "\033[K", "\033[J"
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"

# Window/tab title. OSC 0 sets icon + title, OSC 2 the window title; Terminal
# and iTerm2 both honour them. Re-asserted every frame so nothing else can
# claim the heading.
WINDOW_TITLE = "UpinelAIOS Status"
SET_TITLE = f"\033]0;{WINDOW_TITLE}\007\033]2;{WINDOW_TITLE}\007"
CLEAR_TITLE = "\033]0;\007\033]2;\007"


def term_size():
    sz = shutil.get_terminal_size((100, 30))
    return sz.columns, sz.lines


# ── keyboard input ───────────────────────────────────────────────────────────
class Keys:
    """
    Single-keypress input in cbreak mode.

    cbreak rather than raw: it turns off line buffering so a key arrives
    immediately, while leaving ISIG alone so Ctrl-C still raises SIGINT and the
    dashboard can shut down cleanly. Terminal attributes are always restored,
    including on an exception, because leaving a shell in cbreak mode is a
    genuinely unpleasant state to hand back to someone.
    """

    def __init__(self, enabled):
        self.enabled = bool(enabled) and sys.stdin.isatty()
        self._fd = None
        self._saved = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self.enabled = False
        return self

    def __exit__(self, *exc):
        self.restore()
        return False

    def restore(self):
        if self._fd is None or self._saved is None:
            return
        try:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        except Exception:
            pass
        self._fd = None
        self._saved = None

    def poll(self, timeout):
        """Wait up to `timeout` seconds for one key. Returns str or None."""
        if not self.enabled:
            time.sleep(max(0.0, timeout))
            return None
        try:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        try:
            ch = sys.stdin.read(1)
        except (OSError, ValueError):
            return None
        if ch == "\x1b":
            # Report Escape immediately rather than trying to classify the
            # sequence. Waiting to see whether more bytes follow an ESC is a
            # coin flip against terminal buffering, and getting it wrong means
            # the cancel key silently does nothing - which is how an accidental
            # model switch got committed during testing. Drain anything already
            # queued (an arrow key's trailing bytes) and treat it as Escape:
            # cancel is only meaningful while a toggle is pending, so a stray
            # arrow key cancelling a pending action is harmless.
            try:
                import select
                while select.select([sys.stdin], [], [], 0.005)[0]:
                    if not sys.stdin.read(1):
                        break
            except (OSError, ValueError):
                pass
            return "ESC"
        return ch


# ── pending-action state ─────────────────────────────────────────────────────
# A toggle does not apply on the keystroke that chose it. It arms a countdown
# and applies when the countdown expires, so pressing the key again cycles on
# without committing, and Enter commits early. That is what makes "press t until
# you see the level you want" work.
SETTLE_SECONDS = 2.0


class Pending:
    def __init__(self):
        self.kind = None          # "thinking" | "model"
        self.value = None         # level, or repo id
        self.deadline = 0.0
        self.from_value = None    # what it will change away from

    def arm(self, kind, value, from_value=None, seconds=SETTLE_SECONDS):
        self.kind = kind
        self.value = value
        self.from_value = from_value
        self.deadline = time.time() + seconds

    def clear(self):
        self.kind = self.value = self.from_value = None
        self.deadline = 0.0

    @property
    def active(self):
        return self.kind is not None

    def remaining(self):
        return max(0.0, self.deadline - time.time())

    def expired(self):
        return self.active and self.remaining() <= 0


# ── sampling helpers ─────────────────────────────────────────────────────────
def run(cmd, timeout=5):
    """
    Run a probe command, return stdout or '' on any failure.

    start_new_session=True is load-bearing, not hygiene. Without it each probe
    shares our session and controlling terminal, and Terminal.app titles the
    window from whatever process group is in front of the tty - so spawning
    lsof, ioreg and pmset once a second made the window heading flicker between
    "lsof", "ioreg", "python3" and back. setsid() detaches them completely, so
    the terminal never sees them at all.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, start_new_session=True)
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def vm_stat():
    """Parse vm_stat into a dict of page counts."""
    out = run(["vm_stat"])
    stats = {}
    for line in out.splitlines():
        m = re.match(r'"?([^":]+)"?:\s+(\d+)', line.strip())
        if m:
            stats[m.group(1).strip()] = int(m.group(2))
    return stats


class CPUSampler:
    """CPU utilisation from Mach host_statistics tick counters."""

    HOST_CPU_LOAD_INFO = 3
    CPU_STATE_MAX = 4

    def __init__(self):
        self.ok = False
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            self.libc = libc
            libc.mach_host_self.restype = ctypes.c_uint
            libc.host_statistics.argtypes = [
                ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint)]
            self.host = libc.mach_host_self()
            self.ok = True
        except Exception:
            return
        # Mach reports cumulative ticks, so the first delta is always zero and
        # a single-sample reading shows "--". Take a short baseline now so even
        # the first rendered frame carries a real percentage.
        self.prev = self._read()
        if self.prev is not None:
            time.sleep(0.15)
            second = self._read()
            if second is not None:
                self.prev = second

    def _read(self):
        buf = (ctypes.c_uint32 * (self.CPU_STATE_MAX))()
        count = ctypes.c_uint(self.CPU_STATE_MAX)
        rc = self.libc.host_statistics(
            self.host, self.HOST_CPU_LOAD_INFO,
            ctypes.byref(buf), ctypes.byref(count))
        if rc != 0:
            return None
        return list(buf)

    def sample(self):
        """Return (total_pct, user_pct, sys_pct) since the previous call."""
        if not self.ok:
            return None
        cur = self._read()
        if cur is None or self.prev is None:
            self.prev = cur
            return None
        deltas = [cur[i] - self.prev[i] for i in range(self.CPU_STATE_MAX)]
        self.prev = cur
        total = sum(deltas)
        if total <= 0:
            return None
        user = deltas[0] + deltas[3]      # USER + NICE
        sysc = deltas[1]                  # SYSTEM
        busy = user + sysc
        return (100.0 * busy / total, 100.0 * user / total, 100.0 * sysc / total)


class GPUSampler:
    """
    GPU utilisation from IOKit. The accelerator publishes its own busy counter
    as "Device Utilization %"; ioreg can read it without privileges.
    """

    PATTERNS = {
        "device": re.compile(r'"Device Utilization %"=(\d+)'),
        "renderer": re.compile(r'"Renderer Utilization %"=(\d+)'),
        "tiler": re.compile(r'"Tiler Utilization %"=(\d+)'),
        "vram_alloc": re.compile(r'"Allocated PB Size"=(\d+)'),
    }

    def __init__(self):
        self.available = self._probe()

    def _probe(self):
        return bool(run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                        timeout=4).strip())

    def sample(self):
        out = run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                  timeout=4)
        if not out:
            return None
        result = {}
        for key, pat in self.PATTERNS.items():
            m = pat.search(out)
            if m:
                result[key] = int(m.group(1))
        return result or None


class PowerSampler:
    """
    Optional ANE/GPU power via powermetrics. Needs root; we only use it when
    sudo is already passwordless, so the dashboard never prompts mid-run.
    """

    def __init__(self):
        self.available = False
        try:
            probe = subprocess.run(["sudo", "-n", "true"], capture_output=True,
                                   timeout=3, start_new_session=True)
            self.available = (probe.returncode == 0)
        except (subprocess.SubprocessError, OSError):
            self.available = False

    def sample(self):
        if not self.available:
            return None
        out = run(["sudo", "-n", "powermetrics", "--samplers", "gpu_power,ane_power",
                   "-n", "1", "-i", "200"], timeout=8)
        if not out:
            return None
        res = {}
        for key, pat in (
            ("gpu_busy", r"GPU HW active frequency.*?(\d+) MHz"),
            ("gpu_residency", r"GPU Active Residency\s+([\d.]+)%"),
            ("ane_power", r"ANE Power:\s+([\d.]+) mW"),
            ("gpu_power", r"GPU Power:\s+([\d.]+) mW"),
        ):
            m = re.search(pat, out)
            if m:
                res[key] = float(m.group(1))
        return res or None


def swap_usage():
    out = run(["sysctl", "-n", "vm.swapusage"])
    m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M", out)
    if not m:
        return None
    return {"total_gb": float(m.group(1)) / 1024,
            "used_gb": float(m.group(2)) / 1024,
            "free_gb": float(m.group(3)) / 1024}


def proc_stats(pid):
    out = run(["ps", "-o", "rss=,%cpu=,etime=", "-p", str(pid)], timeout=4)
    parts = out.split()
    if len(parts) < 3:
        return None
    return {"rss_gb": int(parts[0]) / 1024 / 1024,
            "cpu_pct": float(parts[1]),
            "elapsed": parts[2]}


def thermal_state():
    out = run(["pmset", "-g", "therm"], timeout=4)
    if "No thermal warning level has been recorded" in out:
        return "nominal"
    m = re.search(r"CPU_Scheduler_Limit\s*=\s*(\d+)", out)
    if m and int(m.group(1)) < 100:
        return f"throttled ({m.group(1)}%)"
    return "elevated"


# ── server sampling ──────────────────────────────────────────────────────────
class ServerSampler:
    """
    MTPLX publishes one full telemetry snapshot on
    /v1/mtplx/metrics/stream. We take the first event and hang up, which is
    simpler and more robust than holding an SSE connection open, and gives
    the same data as the menu-bar dashboard: live in-flight progress, the
    process's own memory breakdown, and lifetime token counters.

    Note the OpenAI API lives under /v1 but the operational endpoints are at
    the server root, and this one is at /v1/mtplx/... -- hence both fields.
    """

    def __init__(self, base, api_key, model_dir=None, pid_file=None,
                 port=None, lan=False):
        self.base = base.rstrip("/")          # .../v1  (OpenAI surface)
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.key = api_key
        self.model_dir = model_dir
        self.pid_file = pid_file
        self.port = port
        self.lan = lan

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def snapshot(self, timeout=4):
        """Read exactly one telemetry event and close."""
        req = urllib.request.Request(
            self.base + "/mtplx/metrics/stream",
            headers={"Accept": "text/event-stream", **self._headers()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                deadline = time.time() + timeout
                while time.time() < deadline:
                    raw = r.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace")
                    if line.startswith("data: "):
                        return json.loads(line[6:])
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, OSError, TimeoutError):
            return None
        return None

    def live_settings(self):
        """Current live decode policy (reasoning mode, depth, ...)."""
        try:
            req = urllib.request.Request(self.root + "/v1/mtplx/settings",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=4) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return None

    def set_thinking(self, level):
        """
        Change thinking live. The model stays loaded; only the decode policy
        moves, so this is instant and safe mid-generation.

        The restart-persistent default is THINKING in env.conf; this changes the
        running server only.
        """
        if level == "off":
            payload = {"reasoning": "off", "enable_thinking": False}
        else:
            effort = {"minimal": "low", "low": "low",
                      "medium": "medium", "high": "xhigh"}.get(level, "low")
            payload = {"reasoning": "on", "enable_thinking": True,
                       "reasoning_effort": effort}
        try:
            req = urllib.request.Request(
                self.root + "/v1/mtplx/settings",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", **self._headers()})
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, OSError):
            return None

    def server_pid(self):
        if self.pid_file and os.path.exists(self.pid_file):
            try:
                return int(open(self.pid_file).read().strip())
            except (OSError, ValueError):
                return None
        return None

    def log_totals(self, path, cache):
        """
        Sum prompt/completion tokens from the server log.

        MTPLX's own lifetime counters reset when the process restarts; the log
        is appended across restarts, so this is the number that answers "how
        much has this machine generated in total". The read is incremental:
        `cache` carries the byte offset so each refresh only parses new lines.
        """
        if not path or not os.path.exists(path):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        offset = cache.get("offset", 0)
        if size < offset:               # log rotated
            offset = 0
            cache.update(prompt=0, completion=0, requests=0)
        totals = {
            "prompt": cache.get("prompt", 0),
            "completion": cache.get("completion", 0),
            "requests": cache.get("requests", 0),
        }
        try:
            with open(path, "r", errors="replace") as fh:
                fh.seek(offset)
                for line in fh:
                    if '"event": "mtplx_openai_generation"' not in line:
                        continue
                    mp = re.search(r'"prompt_tokens": (\d+)', line)
                    mc = re.search(r'"completion_tokens": (\d+)', line)
                    if mp and mc:
                        totals["prompt"] += int(mp.group(1))
                        totals["completion"] += int(mc.group(1))
                        totals["requests"] += 1
                cache["offset"] = fh.tell()
        except OSError:
            return None
        cache.update(prompt=totals["prompt"], completion=totals["completion"],
                     requests=totals["requests"], offset=cache.get("offset", 0))
        return totals

    def clients(self):
        """
        TCP peers currently connected to the server port.

        MTPLX does not record the client address on a request, so this is read
        from the socket table: it answers "who is connected", not "which
        request came from where".

        lsof prints one row per socket endpoint, so a single connection appears
        twice - once owned by the server and once by the client. We keep only
        the client side and label it, so the panel reports who is talking to us
        rather than naming our own server process back at the user.
        """
        if not self.port:
            return []
        port = int(self.port)
        out = run(["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:ESTABLISHED"],
                  timeout=5)
        if not out:
            return []

        server_pid = self.server_pid()
        peers = {}
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                pid = int(parts[1])
            except (ValueError, IndexError):
                continue
            name = parts[8]
            # Skip the server's own end of the socket; keep the remote end.
            if pid == server_pid:
                continue
            if "->" not in name:
                continue
            local, remote = name.split("->", 1)
            peer_ip = remote.rsplit(":", 1)[0]
            if not peer_ip:
                continue
            entry = peers.setdefault(peer_ip, {"conns": 0, "procs": set()})
            entry["conns"] += 1
            # lsof escapes spaces and other characters in COMMAND as \x20 etc.
            entry["procs"].add(re.sub(r"\\x([0-9a-fA-F]{2})",
                                      lambda m: chr(int(m.group(1), 16)),
                                      parts[0]))

        result = []
        for ip, entry in sorted(peers.items()):
            loopback = ip.startswith("127.") or ip == "::1"
            result.append({
                "ip": ip,
                "conns": entry["conns"],
                "procs": sorted(entry["procs"]),
                "local": loopback,
            })
        return result


# ── rendering ────────────────────────────────────────────────────────────────
def bar(pct, width=16, warn=75, crit=90):
    """A horizontal meter. Returns text; caller adds colour."""
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(round(pct * width / 100.0))
    return "\u2588" * filled + "\u2591" * (width - filled)


def color_for(pct, warn=75, crit=90):
    if pct >= crit:
        return RED
    if pct >= warn:
        return YELLOW
    return GREEN


SPARK = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values, width=40):
    vals = list(values)[-width:]
    if not vals:
        return DIM + "\u00b7" * width + RESET
    hi = max(vals)
    lo = min(vals)
    if hi == lo:
        # Flat series: draw a steady line rather than dividing by zero.
        return DIM + "\u2585" * len(vals) + "\u00b7" * (width - len(vals)) + RESET
    span = (hi - lo) or 1.0
    out = []
    for v in vals:
        idx = int(round((v - lo) / span * 8))
        out.append(SPARK[max(1, min(8, idx))])
    pad = "\u00b7" * (width - len(out))
    return pad + "".join(out)


CHART_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def line_chart(values, width, height=4):
    """
    Draw a filled area chart `height` rows tall, and return
    (rows, axis_lo, axis_hi).

    The y-axis is zoomed to the data's own range rather than anchored at zero.
    A throughput chart anchored at zero is a flat wall of blocks: 14-23 t/s
    against a 0-23 axis is all in the top eighth, so the shape disappears. The
    returned bounds are printed next to the chart so the zoom is explicit
    rather than implied.

    Each column is a bar of eighths, so four rows still resolve 32 levels.
    """
    vals = [float(v) for v in values][-width:]
    if not vals:
        return [" " * width for _ in range(height)], 0.0, 0.0
    lo_raw, hi_raw = min(vals), max(vals)
    if hi_raw <= lo_raw:
        # Flat series: centre it in the band rather than dividing by zero.
        lo, hi = lo_raw * 0.9, hi_raw * 1.1 + 1e-6
    else:
        span = hi_raw - lo_raw
        lo = max(0.0, lo_raw - span * 0.35)
        hi = hi_raw + span * 0.15

    vals = [0.0] * (width - len(vals)) + vals
    span = (hi - lo) or 1.0

    rows = [[" "] * width for _ in range(height)]
    for x, v in enumerate(vals):
        eighths = int(round(max(0.0, (v - lo)) / span * height * 8))
        full, part = divmod(eighths, 8)
        for r in range(full):
            y = height - 1 - r
            if 0 <= y < height:
                rows[y][x] = "\u2588"
        if part and full < height:
            y = height - 1 - full
            if 0 <= y < height:
                rows[y][x] = CHART_BLOCKS[part - 1]
    return ["".join(r) for r in rows], lo, hi


def num(value, default=0.0):
    """
    Coerce a telemetry value to float.

    MTPLX publishes `null` for a field it has not measured yet - most often
    decode_tok_s while a long prompt is still prefilling - and JSON `null`
    becomes Python None. `d.get(key, 0)` does NOT protect against that: the key
    is present, so the default is never used and the None flows into a format
    specifier and raises. Everything numeric that comes from telemetry goes
    through here.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def lp_num(value, default=0):
    """Integer coercion for a telemetry count."""
    try:
        return int(num(value, default))
    except (TypeError, ValueError):
        return default


def fmt_num(value, spec=".1f", suffix="", default="--"):
    """Format a possibly-None telemetry number, falling back to `default`."""
    if value is None:
        return default
    try:
        return format(float(value), spec) + suffix
    except (TypeError, ValueError, ValueError):
        return default


def human_gb(gb):
    gb = num(gb)
    if gb >= 100:
        return f"{gb:.0f}G"
    if gb >= 10:
        return f"{gb:.1f}G"
    return f"{gb:.2f}G"


def col_widths(left_lines, right_lines, total_width):
    """Left/right column widths for a two-column block."""
    gutter = 3
    lw = max((len(strip_ansi(s)) for s in left_lines), default=0) + gutter
    lw = max(24, min(lw, (total_width * 62) // 100))
    return lw, total_width - lw


def two_col(left_lines, right_lines, total_width):
    """Lay two lists of strings side by side."""
    lw, rw = col_widths(left_lines, right_lines, total_width)
    rows = max(len(left_lines), len(right_lines))
    out = []
    for i in range(rows):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        pad = " " * max(0, lw - len(strip_ansi(l)))
        # Truncate the right column so we never wrap and corrupt the frame.
        r = truncate(r, rw)
        out.append(l + pad + r)
    return out


ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def truncate(s, width):
    if len(strip_ansi(s)) <= width:
        return s
    # Colour-safe truncation: drop from the plain-text end.
    out, visible = [], 0
    i = 0
    while i < len(s):
        m = ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        if visible >= width - 1:
            out.append("\u2026")
            break
        out.append(s[i])
        visible += 1
        i += 1
    return "".join(out) + RESET


class Dashboard:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.cpu = CPUSampler()
        self.gpu = GPUSampler()
        self.power = PowerSampler() if args.power else None
        self.server = ServerSampler(cfg["base"], cfg["api_key"],
                                    cfg.get("model_dir"), cfg.get("pid_file"),
                                    port=cfg.get("port"))
        self.hist_cpu = deque(maxlen=80)
        self.hist_gpu = deque(maxlen=80)
        self.hist_tps = deque(maxlen=80)
        self.t0 = time.time()
        self.last_good = 0.0
        self._log_cache = {}
        self.log_totals = None
        # Cached probes. lsof and pmset are not free, and neither the client
        # list nor the thermal state changes meaningfully at 1 Hz - spawning
        # them every second just churns the process table for no new data.
        self._clients_cache = (0, [])
        self._thermal_cache = (0, "unknown")
        self._frame = 0
        self._error_count = 0
        self._last_error = ""
        self.pending = Pending()
        self.status_note = ""        # one-line feedback under the panels
        self.status_note_until = 0.0
        self.levels = ["off", "minimal", "low", "medium", "high"]
        self._cached_thinking = None
        self._thinking_checked = 0.0

    # ── model discovery ──────────────────────────────────────────────────────
    def downloaded_models(self):
        """
        Models present on disk, newest-relevant first.

        Directory names are owner--name, which is how fetch-model.sh lays them
        out, so the repo id is recoverable without reading any manifest.
        """
        root = self.cfg.get("models_dir")
        if not root or not os.path.isdir(root):
            return []
        found = []
        try:
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if not os.path.isdir(path) or "--" not in name:
                    continue
                if not any(f.endswith(".safetensors") for f in os.listdir(path)):
                    continue
                found.append({"repo": name.replace("--", "/", 1), "dir": path})
        except OSError:
            return []
        return found

    def short_model(self, repo):
        """A label that fits in a footer: the tail of the repo id."""
        tail = repo.split("/")[-1]
        for marker in ("Uncensored-HauhauCS-Aggressive-MTPLX-",
                       "Uncensored-MTPLX-", "-MTPLX-Optimized-Speed", "-MTPLX"):
            if marker in tail:
                head = tail.split(marker)[0].rstrip("-_.")
                if head:
                    return head if len(head) <= 22 else head[:21] + "…"
        return tail if len(tail) <= 26 else tail[:25] + "…"

    # ── live actions ─────────────────────────────────────────────────────────
    def note(self, text, seconds=4.0):
        self.status_note = text
        self.status_note_until = time.time() + seconds

    def current_thinking(self):
        """The server's live setting, which may differ from env.conf."""
        try:
            data = json.loads(self.server.live_settings() or "{}")
        except (json.JSONDecodeError, TypeError):
            return None
        mode = data.get("reasoning")
        if mode == "off":
            return "off"
        return data.get("reasoning_effort") or "minimal"

    def apply_thinking(self, level):
        payload = self.server.set_thinking(level)
        if payload is None:
            self.note(f"{RED}Could not change thinking - see run/server.log{RESET}", 6)
            return
        # Update the footer immediately rather than waiting for the next
        # periodic refresh, which only runs every few seconds.
        self._cached_thinking = level
        self._thinking_checked = time.time()
        self.note(f"{GREEN}thinking -> {level}{RESET}  "
                  f"{DIM}(live only; set THINKING in env.conf to persist){RESET}", 6)

    def apply_model(self, repo):
        """
        Persist the choice and restart. The restart is detached, so the
        dashboard survives it and simply shows "not running" until the new
        model finishes loading.
        """
        env_file = self.cfg.get("env_file")
        repo_dir = self.cfg.get("repo_dir")
        if not env_file or not repo_dir:
            self.note(f"{RED}Cannot switch models: env.conf path unknown{RESET}", 6)
            return
        try:
            import re as _re
            src = open(env_file).read()
            src, n = _re.subn(r'^MODEL=.*$', f'MODEL="{repo}"', src,
                              count=1, flags=_re.M)
            if n != 1:
                raise OSError("MODEL= not found in env.conf")
            open(env_file, "w").write(src)
        except OSError as exc:
            self.note(f"{RED}Could not write env.conf: {exc}{RESET}", 8)
            return
        try:
            subprocess.Popen([os.path.join(repo_dir, "restart.sh")],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             start_new_session=True)
        except OSError as exc:
            self.note(f"{RED}Could not restart: {exc}{RESET}", 8)
            return
        self.note(f"{GREEN}switching to {self.short_model(repo)}{RESET}  "
                  f"{DIM}loading, this takes 30-90s{RESET}", 12)

    # ── key handling ─────────────────────────────────────────────────────────
    def handle_key(self, key):
        if key is None:
            return
        # Everything is compared in lower case. Keys.poll() returns the literal
        # "ESC" for an escape byte, so Escape must be matched as "esc" here -
        # comparing against "ESC" after lowering is a silent no-op, and that
        # bug let an accidental model switch commit twice during testing.
        key = key.lower()

        if key == "t":
            cur = self.pending.value if self.pending.kind == "thinking" \
                else (self.current_thinking() or "minimal")
            if cur not in self.levels:
                cur = "minimal"
            nxt = self.levels[(self.levels.index(cur) + 1) % len(self.levels)]
            self.pending.arm("thinking", nxt, from_value=cur)
        elif key == "m":
            models = self.downloaded_models()
            if len(models) < 2:
                self.note(f"{YELLOW}Only one model downloaded. "
                          f"Try ./model_download.sh{RESET}", 5)
                return
            repos = [m["repo"] for m in models]
            cur = self.pending.value if self.pending.kind == "model" \
                else self.cfg.get("model_repo")
            idx = repos.index(cur) if cur in repos else -1
            self.pending.arm("model", repos[(idx + 1) % len(repos)], from_value=cur)
        elif key in ("\r", "\n"):
            if self.pending.active:
                self._commit()
        elif key == "esc":
            if self.pending.active:
                self.pending.clear()
                self.note(f"{DIM}cancelled{RESET}", 3)
        elif key == "q":
            raise KeyboardInterrupt

    def _commit(self):
        kind, value = self.pending.kind, self.pending.value
        self.pending.clear()
        if kind == "thinking":
            self.apply_thinking(value)
        elif kind == "model":
            self.apply_model(value)

    def tick(self):
        """Apply a pending action whose settle time has elapsed."""
        if self.pending.expired():
            self._commit()

    # ── one sample ───────────────────────────────────────────────────────────
    def collect(self):
        s = {"t": time.time()}

        cpu = self.cpu.sample()
        s["cpu"] = cpu[0] if cpu else None
        s["cpu_user"] = cpu[1] if cpu else None
        s["cpu_sys"] = cpu[2] if cpu else None

        g = self.gpu.sample()
        s["gpu"] = g.get("device") if g else None
        s["gpu_renderer"] = g.get("renderer") if g else None
        s["gpu_tiler"] = g.get("tiler") if g else None

        if self.power:
            p = self.power.sample()
            s["ane"] = p.get("ane_power") if p else None
            s["gpu_power"] = p.get("gpu_power") if p else None
            s["gpu_residency"] = p.get("gpu_residency") if p else None
        else:
            s["ane"] = None

        vs = vm_stat()
        total_bytes = os.sysconf("SC_PHYS_PAGES") * PAGE_SIZE
        s["ram_total_gb"] = total_bytes / 1024 ** 3

        def pages(key):
            return vs.get(key, 0) * PAGE_SIZE / 1024 ** 3

        # vm_stat pages overlap in ways that are easy to double count. Use the
        # conventional macOS breakdown:
        #   free  = truly available (free + speculative)
        #   cached= inactive + file-backed; reclaimable on demand
        #   used  = wired + active + compressed; not available without swapping
        s["ram_wired_gb"] = pages("Pages wired down")
        s["ram_compressed_gb"] = pages("Pages occupied by compressor")
        s["ram_free_gb"] = pages("Pages free") + pages("Pages speculative")
        s["ram_cached_gb"] = pages("Pages inactive") + pages("File-backed pages")
        s["ram_active_gb"] = pages("Pages active")
        s["ram_used_gb"] = (s["ram_wired_gb"] + pages("Pages active")
                            + s["ram_compressed_gb"])
        # Never let rounding push the columns past the total.
        over = (s["ram_used_gb"] + s["ram_cached_gb"] + s["ram_free_gb"]
                - s["ram_total_gb"])
        if over > 0:
            s["ram_cached_gb"] = max(0.0, s["ram_cached_gb"] - over)
        s["ram_pressure_pct"] = (100.0 * s["ram_wired_gb"] / s["ram_total_gb"]
                                 if s["ram_total_gb"] else 0)

        s["swap"] = swap_usage()
        # Thermal state moves on the scale of tens of seconds.
        if self._frame % 10 == 0:
            self._thermal_cache = (self._frame, thermal_state())
        s["thermal"] = self._thermal_cache[1]

        pid = self.server.server_pid()
        s["proc"] = proc_stats(pid) if pid else None

        # ── server ──
        snap = self.server.snapshot()
        s["snap"] = snap
        s["server_up"] = snap is not None
        # The connection list is worth refreshing often enough to notice a new
        # client, but not every frame.
        if snap is not None and self._frame % 3 == 0:
            self._clients_cache = (self._frame, self.server.clients())
        s["clients"] = self._clients_cache[1] if snap is not None else []
        s["log_totals"] = self.server.log_totals(self.cfg.get("log_file"),
                                                 self._log_cache)
        s["in_flight"] = []
        s["sessions"] = []
        s["lifetime"] = {}
        s["srv_mem"] = {}
        s["live_tps"] = None
        s["live_history"] = []
        s["rolling"] = {}
        s["scheduler"] = {}
        s["sessions_n"] = 0

        if snap:
            s["in_flight"] = snap.get("in_flight") or []
            s["sessions"] = (snap.get("sessions") or {}).get("sessions") or []
            s["sessions_n"] = (snap.get("sessions") or {}).get("count") or len(s["sessions"])
            s["lifetime"] = snap.get("lifetime") or {}
            s["srv_mem"] = snap.get("mem") or {}
            s["rolling"] = snap.get("rolling") or {}
            s["scheduler"] = snap.get("scheduler") or {}
            s["memory_pressure_level"] = snap.get("memory_pressure_level")
            s["context_window"] = snap.get("context_window")
            hist = s["rolling"].get("live_history") or []
            s["live_history"] = hist
            # The most recent live sample is the current generation rate.
            for h in reversed(hist):
                if isinstance(h, dict) and h.get("tok_s"):
                    s["live_tps"] = float(h["tok_s"])
                    break
            if s["live_tps"] is None:
                lat = snap.get("latest") or {}
                s["live_tps"] = lat.get("decode_tok_s")
        if s["cpu"] is not None:
            self.hist_cpu.append(s["cpu"])
        if s["gpu"] is not None:
            self.hist_gpu.append(float(s["gpu"]))
        return s

    # ── render ───────────────────────────────────────────────────────────────
    # The layout is built as ordered blocks and emitted only while they fit the
    # terminal. A dashboard that renders more rows than the window has scrolls
    # its own header off the top, which is worse than showing fewer panels.
    def _block_header(self, s, width):
        cfg = self.cfg
        up = time.time() - self.t0
        hh, rem = divmod(int(up), 3600)
        mm, ss = divmod(rem, 60)

        L = []
        title = f"{BOLD}UpinelAIOS{RESET}"
        status = (f"{GREEN}\u25cf serving{RESET}" if s["server_up"]
                  else f"{RED}\u25cf not running{RESET}")
        clock = f"up {hh:02d}:{mm:02d}:{ss:02d}"
        pad = max(1, width - len(strip_ansi(title)) - len(strip_ansi(status))
                  - len(clock) - 4)
        L.append(f"{title}{' ' * pad}{status}  {DIM}{clock}{RESET}")
        L.append(f"{DIM}  Upinel's One-Click AI Agent Server OS for Mac{RESET}")
        L.append(DIM + "\u2500" * width + RESET)

        key = cfg.get("api_key") or "(none - loopback only)"
        L.append(f"  {DIM}model {RESET}{BOLD}{cfg['model_repo']}{RESET}")
        L.append(f"  {DIM}served as {RESET}{CYAN}{cfg.get('served_name','')}{RESET}"
                 f"{DIM}   ctx {cfg.get('context','?')}   KV {cfg.get('kv','?')}"
                 f"   MTP d{cfg.get('depth','?')}   {cfg.get('profile','?')}{RESET}")
        L.append(f"  {DIM}api key {RESET}{YELLOW}{key}{RESET}"
                 f"{DIM}   {cfg.get('lan_url', cfg['base'])}{RESET}")
        L.append("")
        return L

    def _live_thinking_label(self):
        """What thinking is actually set to right now, not what env.conf says."""
        if self.pending.kind == "thinking":
            return self.pending.value
        live = self._cached_thinking
        if live:
            return live
        return self.cfg.get("thinking", "?")

    def _header_lines(self):
        """Header line count, used to size everything else."""
        return 7

    def _block_host(self, s, width):
        def gauge(label, pct, text, barw=16):
            if pct is None:
                return (f"  {BOLD}{label:<5}{RESET} "
                        f"{DIM}{'\u2591' * barw}   {'--':>7}{RESET}")
            return (f"  {BOLD}{label:<5}{RESET} "
                    f"{color_for(pct)}{bar(pct, barw)}{RESET} {text:>7}")

        ram_pct = 100.0 * s["ram_used_gb"] / (s["ram_total_gb"] or 1)
        swap_pct = None
        if s.get("swap") and s["swap"]["total_gb"] > 0:
            swap_pct = 100.0 * s["swap"]["used_gb"] / s["swap"]["total_gb"]

        left = [f"  {BOLD}HOST{RESET}"]
        left.append(gauge("CPU", s["cpu"],
                          f"{s['cpu']:.1f}%" if s["cpu"] is not None else "--"))
        left.append(gauge("GPU", s["gpu"],
                          f"{s['gpu']:.0f}%" if s["gpu"] is not None else "--"))
        if s.get("ane") is not None:
            left.append(f"  {BOLD}ANE  {RESET} {GREEN}{bar(min(100, s['ane'] / 20), 16)}"
                        f"{RESET} {s['ane']:>5.0f}mW")
        else:
            left.append(f"  {BOLD}ANE  {RESET} {DIM}{'\u2591' * 16}   n/a{RESET}")
        left.append(gauge("RAM", ram_pct, human_gb(s["ram_used_gb"])))
        left.append(gauge("SWAP", swap_pct,
                          human_gb(s["swap"]["used_gb"]) if s.get("swap") else "--"))
        detail = []
        if s["cpu_user"] is not None:
            detail.append(f"cpu {s['cpu_user']:.0f}u/{s['cpu_sys']:.0f}s")
        detail.append(s["thermal"])
        if s.get("memory_pressure_level") is not None:
            detail.append(f"pressure L{s['memory_pressure_level']}")
        detail.append("gpu=system")
        left.append(f"  {DIM}{'  '.join(detail)}{RESET}")

        mem = s.get("srv_mem") or {}
        right = [f"  {BOLD}ENDPOINT PROCESS{RESET}"]
        if mem and mem.get("ok"):
            def gb(key):
                return num(mem.get(key)) / 1024 ** 3

            total_gb = gb("active_memory_bytes")
            span = max(total_gb, 1.0)
            for label, key in (("weights", "model_weights_bytes"),
                               ("session bank", "session_bank_bytes"),
                               ("generation", "generation_working_bytes"),
                               ("kv cache", "cache_memory_bytes")):
                v = gb(key)
                right.append(f"  {label:<13}{human_gb(v):>7} "
                             f"{DIM}{bar(100 * v / span, 10)}{RESET}")
            right.append(f"  {'\u2500' * 13}{'\u2500' * 18}")
            right.append(f"  {'total':<13}{human_gb(total_gb):>7}  "
                         f"{DIM}peak {human_gb(gb('peak_memory_bytes'))}{RESET}")
            right.append(f"  {'host free':<13}{human_gb(s['ram_free_gb']):>7}  "
                         f"{DIM}(+{human_gb(s['ram_cached_gb'])} cached){RESET}")
        else:
            right.append(f"  {DIM}no telemetry{RESET}")
        return two_col(left, right, width) + [""]

    def _block_activity(self, s, width, compact=False):
        sched = s.get("scheduler") or {}
        act = [f"  {BOLD}CONCURRENT ACTIVITY{RESET}"]
        if compact:
            now = s.get("live_tps")
            act.append(f"  in-flight {YELLOW}{len(s['in_flight'])}{RESET}"
                       f"  sessions {s.get('sessions_n', 0)}"
                       f"  {sched.get('active_lane') or '-'}"
                       + (f"  {GREEN}{now:.1f} t/s{RESET}" if now else ""))
        else:
            act.append(f"  in-flight {YELLOW}{len(s['in_flight'])}{RESET}"
                       f"   sessions {s.get('sessions_n', 0)}"
                       f"   lane {sched.get('active_lane') or '-'}"
                       f"   policy {sched.get('scheduler_policy') or '-'}")
            for req in s["in_flight"][:2]:
                lp = req.get("last_progress") or {}
                rid = (req.get("request_id") or "")[:14]
                # Guard and format the SAME field. The original code tested
                # completion_tokens but formatted decode_tok_s, so a request in
                # prefill - tokens counted, decode rate still null - crashed the
                # whole dashboard with a TypeError.
                progress = ""
                if lp.get("completion_tokens") is not None:
                    progress = (f"  {GREEN}{lp['completion_tokens']} tok"
                                f" @ {fmt_num(lp.get('decode_tok_s'), '.1f', ' t/s')}"
                                f"{RESET}")
                act.append(f"  {CYAN}{rid}\u2026{RESET} "
                           f"{fmt_num(req.get('age_s'), '>5.0f', 's')}"
                           f" {DIM}{(req.get('session_id') or '')[:12]}{RESET}"
                           f" {lp_num(req.get('prompt_tokens'))} ctx"
                           + progress)
                prev = (req.get("prompt_preview") or "").replace("\n", " ")
                if prev:
                    act.append(f"    {DIM}\u201c{truncate(prev, width // 2 - 8)}{RESET}")
        if not s["in_flight"] and s["server_up"] and not compact:
            act.append(f"  {DIM}idle{RESET}")

        cli = [f"  {BOLD}CLIENTS{RESET}  {DIM}connected peers{RESET}"]
        if s["clients"]:
            for cl in s["clients"][:4]:
                where = "this Mac" if cl.get("local") else cl["ip"]
                who = ",".join(cl["procs"])[:16] or "?"
                cli.append(f"  {where:<16} {cl['conns']} conn  {DIM}{who}{RESET}")
        elif s["server_up"]:
            cli.append(f"  {DIM}none connected{RESET}")
        else:
            cli.append(f"  {DIM}-{RESET}")
        return two_col(act, cli, width) + [""]

    def _block_rate(self, s, width, chart_rows_n, max_rows):
        """
        Live token-rate chart beside the cumulative counters.

        Both columns are sized to max_rows: the counter column is usually taller
        than the chart, and letting it dictate the block height is what pushed
        the whole panel out of a 24-row terminal.
        """
        hist = [num(h.get("tok_s")) for h in (s.get("live_history") or [])
                if isinstance(h, dict)]
        lw, _ = col_widths(
            [f"  {BOLD}TOKEN RATE{RESET}  {DIM}live{RESET}"
             f"  {DIM}min 00.0  avg 00.0  max 00.0{RESET}"], [], width)
        chart_w = max(12, lw - 9)
        # Budget: header + chart + scale line + trailing blank = chart_h + 3.
        chart_h = max(1, min(chart_rows_n, max_rows - 3))
        rows, axis_lo, axis_hi = line_chart(hist, chart_w, height=chart_h)

        life = s.get("lifetime") or {}
        logt = s.get("log_totals") or {}
        counters = [f"  {BOLD}TOKENS GENERATED{RESET}"]
        detail = []
        if logt:
            detail.append(("output", f"{logt['completion']:,}"))
            detail.append(("input", f"{logt['prompt']:,}"))
            detail.append(("total", f"{logt['completion'] + logt['prompt']:,}"))
            detail.append(("requests", f"{logt['requests']:,}"))
        if life:
            detail.append(("restart",
                           f"{int(num(life.get('completion_tokens_total'))):,}"))
        # Always keep at least 'output'; add rows only while they fit. The cap
        # leaves room for the blank row that separates this block from the next.
        counter_cap = max(1, max_rows - 1)
        for label, value in detail:
            if len(counters) < counter_cap:
                counters.append(f"  {label:<9}{value:>12}")

        now = s.get("live_tps")
        head = f"  {BOLD}TOKEN RATE{RESET}  {DIM}live{RESET}"
        if hist:
            head += (f"  {DIM}min {min(hist):.1f}  avg {sum(hist)/len(hist):.1f}"
                     f"  max {max(hist):.1f}{RESET}")
        lines = [head]
        for i, row in enumerate(rows):
            label = f"{axis_hi:>6.1f} " if i == 0 else " " * 7
            lines.append(f"  {DIM}{label}{RESET}{GREEN}{row}{RESET}")
        if hist:
            lines.append(f"  {DIM}{axis_lo:>6.1f} {RESET}"
                         + (f"{CYAN}now {now:.1f} t/s{RESET}" if now
                            else f"{DIM}idle{RESET}"))
        else:
            lines.append(f"  {DIM}{' ' * 7}waiting for traffic{RESET}")
        return two_col(lines, counters, width) + [""]

    def _block_rate_summary(self, s):
        """One-line token panel for terminals too short for the chart."""
        logt = s.get("log_totals") or {}
        now = s.get("live_tps")
        bits = [f"  {BOLD}TOKENS{RESET}"]
        if now:
            bits.append(f"{GREEN}{now:.1f} t/s{RESET}")
        if logt:
            bits.append(f"{DIM}out {logt['completion']:,}"
                        f"   in {logt['prompt']:,}"
                        f"   {logt['requests']:,} reqs{RESET}")
        return [" ".join(bits), ""]

    def _block_spark(self, width):
        sw = max(20, width - 16)
        return [f"  {BOLD}cpu {RESET}{sparkline(self.hist_cpu, sw)}",
                f"  {BOLD}gpu {RESET}{sparkline(self.hist_gpu, sw)}",
                ""]

    def render(self, s):
        width, height = term_size()
        width = max(60, min(width, 200))
        L = []

        header = self._block_header(s, width)
        host = self._block_host(s, width)
        footer = None   # built after the blocks are placed

        # Essential blocks first, then optional ones in priority order, each
        # admitted only if the remaining rows can hold it.
        L.extend(header)
        L.extend(host)

        # The token rate is the panel people watch, so it outranks activity and
        # gets whatever rows are left after the essential blocks.
        # Reserve two rows for the footer block (separator + footer line).
        RESERVED = 2
        remaining = height - len(L) - RESERVED
        rate = None
        if remaining >= 4:
            for chart_rows_n in (5, 4, 3, 2, 1):
                candidate = self._block_rate(s, width, chart_rows_n, remaining)
                if len(candidate) <= remaining:
                    rate = candidate
                    break
                # Last resort: a one-line summary so the panel still appears.
            if rate is None:
                summary = self._block_rate_summary(s)
                if len(summary) <= remaining:
                    rate = summary
        if rate:
            L.extend(rate)
            remaining -= len(rate)

        activity_shown = False
        for compact in (False, True):
            if remaining < 3:
                break
            act = self._block_activity(s, width, compact=compact)
            if len(act) <= remaining:
                L.extend(act)
                remaining -= len(act)
                activity_shown = True
                break

        if remaining >= len(self._block_spark(width)):
            L.extend(self._block_spark(width))

        # One blank row between the last panel and the footer, no more.
        while L and not strip_ansi(L[-1]).strip():
            L.pop()
        L.append("")

        # A pending toggle owns the footer: it is the thing needing a decision,
        # and the countdown has to be visible or the delay feels like a hang.
        if self.pending.active:
            left_txt = self.short_model(self.pending.from_value or "?") \
                if self.pending.kind == "model" else (self.pending.from_value or "?")
            right_txt = self.short_model(self.pending.value) \
                if self.pending.kind == "model" else self.pending.value
            label = "model" if self.pending.kind == "model" else "thinking"
            secs = self.pending.remaining()
            hint = (f"{YELLOW}{label}: {left_txt} \u2192 {right_txt}{RESET}"
                    f"   {BOLD}applying in {secs:.1f}s{RESET}"
                    f"   {DIM}[same key] next  [Enter] now  [Esc] cancel{RESET}")
        elif self.status_note and time.time() < self.status_note_until:
            hint = self.status_note
        else:
            if self.keys.enabled:
                # Show each toggle with its current value, so the footer is a
                # readout as well as a keymap - you can see what t and m are
                # sitting on without pressing them.
                hint = (f"{BOLD}t{RESET}{DIM} thinking{RESET}"
                        f"{CYAN}={self._live_thinking_label()}{RESET}"
                        f"   {BOLD}m{RESET}{DIM} model{RESET}"
                        f"{CYAN}={self.short_model(self.cfg.get('model_repo',''))}{RESET}"
                        f"   {DIM}q quit{RESET}")
            else:
                hint = (f"refresh {self.args.interval}s \u00b7 Ctrl-C to exit"
                        + ("" if self.power and self.power.available
                           else f" \u00b7 --power for ANE")
                        + RESET)
        # If the activity panel did not fit, its essentials ride along in the
        # footer rather than vanishing: who is connected and how much is running.
        if not activity_shown and s["server_up"]:
            bits = [f"{len(s['in_flight'])} in-flight",
                    f"{s.get('sessions_n', 0)} sessions"]
            if s["clients"]:
                who = ", ".join(
                    (c["ip"] if not c.get("local") else "this Mac")
                    + (f" x{c['conns']}" if c["conns"] > 1 else "")
                    for c in s["clients"][:2])
                bits.append(who)
            hint = " \u00b7 ".join(bits) + "   " + f"{DIM}{hint}{RESET}"
        footer = [f"  {DIM}\u2500\u2500{RESET} {hint}"]

        while len(L) < height - len(footer) - 1:
            L.append("")
        L.extend(footer)

        # Final safety pass: clamp every line to the terminal width. The model
        # id and the API key are printed in full, and an unclamped line would
        # wrap and corrupt every row below it.
        L = [truncate(line, width) if len(strip_ansi(line)) > width else line
             for line in L][:height - 1]

        if not TTY_MODE:
            body = "\n".join(strip_ansi(x) if not USE_COLOR else x for x in L)
            return body + "\n" + "\u2500" * min(width, 78) + "\n"

        return SET_TITLE + HOME + "\n".join(x + CLR_EOL for x in L) + CLR_EOS

    def _log_fault(self, detail):
        """Append a render fault to run/dashboard.err, trimmed to one file."""
        path = self.cfg.get("error_log")
        if not path:
            return
        try:
            if os.path.exists(path) and os.path.getsize(path) > 262144:
                os.replace(path, path + ".1")
            with open(path, "a") as fh:
                fh.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"frame {self._frame} ---\n{detail}")
        except OSError:
            pass

    def _fault_frame(self, s):
        """
        Minimal but useful frame when rendering failed.

        The system numbers are collected before rendering, so they are still
        good; only the panel layout is suspect. Showing them means the dashboard
        remains worth watching even while a panel is broken.
        """
        width, _ = term_size()
        width = max(60, min(width, 200))
        L = [
            f"{BOLD}{WINDOW_TITLE}{RESET}"
            f"{' ' * max(1, width - len(WINDOW_TITLE) - 26)}"
            f"{RED}\u25cf display error{RESET}",
            DIM + "\u2500" * width + RESET,
            f"  {YELLOW}The dashboard hit a rendering fault and degraded rather "
            f"than exiting.{RESET}",
            f"  {DIM}{self._last_error.strip().splitlines()[-1][:width - 6]}{RESET}",
            "",
        ]
        if s.get("cpu") is not None:
            L.append(f"  {'cpu':<12}{s['cpu']:.1f}%")
        if s.get("gpu") is not None:
            L.append(f"  {'gpu':<12}{s['gpu']:.0f}%")
        if s.get("ram_total_gb"):
            L.append(f"  {'ram used':<12}{human_gb(s.get('ram_used_gb'))}"
                     f" / {s['ram_total_gb']:.0f} GB")
        if s.get("live_tps"):
            L.append(f"  {'decode':<12}{fmt_num(s['live_tps'], '.1f', ' t/s')}")
        L += [
            "",
            f"  {DIM}in-flight {len(s.get('in_flight') or [])}"
            f"   sessions {s.get('sessions_n', 0)}"
            f"   faults this run: {self._error_count}{RESET}",
            f"  {DIM}detail logged to "
            f"{self.cfg.get('error_log', '(none)')}{RESET}",
            "",
            f"  {DIM}Ctrl-C to exit{RESET}",
        ]
        L = [truncate(x, width) if len(strip_ansi(x)) > width else x for x in L]
        if not TTY_MODE:
            return "\n".join(strip_ansi(x) for x in L) + "\n"
        return SET_TITLE + HOME + "\n".join(x + CLR_EOL for x in L) + CLR_EOS

    def run(self):
        if TTY_MODE:
            sys.stdout.write(SET_TITLE + HIDE_CURSOR)
            sys.stdout.flush()
        self.keys = Keys(self.args.enable_keys and TTY_MODE)
        try:
            with self.keys:
                self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.keys.restore()
            if TTY_MODE:
                # Hand the heading back rather than leaving ours behind.
                sys.stdout.write(CLEAR_TITLE + SHOW_CURSOR + "\n")
                sys.stdout.flush()

    def _loop(self):
        while True:
            started = time.time()
            s = self.collect()
            self.tick()
            try:
                frame = self.render(s)
            except Exception:                                  # noqa: BLE001
                # This is a long-running display. A fault in one panel must not
                # kill the tool and dump a traceback over the frame - that is
                # exactly how the decode_tok_s bug announced itself. Show a
                # degraded frame, keep running, and leave the detail in a log.
                self._error_count += 1
                self._last_error = traceback.format_exc()
                self._log_fault(self._last_error)
                frame = self._fault_frame(s)
            sys.stdout.write(frame)
            sys.stdout.flush()
            if self.args.iterations and self.args.iterations > 0:
                self.args.iterations -= 1
                if self.args.iterations == 0:
                    break
            # Wait out the refresh interval, but wake immediately on a key or
            # when a pending action's countdown is about to expire.
            wait = max(0.05, self.args.interval - (time.time() - started))
            if self.pending.active:
                wait = min(wait, max(0.02, self.pending.remaining()))
            key = self.keys.poll(wait)
            if key is not None:
                self.handle_key(key)


# ── one-shot report (previously status.sh) ───────────────────────────────────
def print_once(cfg, snap):
    s = snap
    print(f"\n{BOLD}UpinelAIOS{RESET} {DIM}- "
          f"Upinel's One-Click AI Agent Server OS for Mac{RESET}\n")

    print(f"{BOLD}Model{RESET}")
    print(f"  {'name':<18} {cfg['model_repo']}")
    if cfg.get("model") and cfg["model"] != cfg["model_repo"]:
        print(f"  {'env.conf MODEL':<18} {cfg['model']}")
    print(f"  {'served as':<18} {cfg.get('served_name','')}")
    print(f"  {'weights on disk':<18} {cfg.get('model_dir','')}")

    print(f"\n{BOLD}Configuration{RESET}  (env.conf)")
    print(f"  {'context window':<18} {cfg.get('context')} tokens")
    print(f"  {'KV quant':<18} {cfg.get('kv')}")
    print(f"  {'MTP depth':<18} {cfg.get('depth')}  (profile {cfg.get('profile')})")
    print(f"  {'thinking':<18} {cfg.get('thinking')}"
          f"  (history {cfg.get('preserve_thinking')})")
    print(f"  {'memory cap':<18} {cfg.get('memory_limit')} GB")
    print(f"  {'batching':<18} {cfg.get('batching')}")

    print(f"\n{BOLD}Hardware{RESET}")
    print(f"  {'chip':<18} {cfg.get('chip','Apple Silicon')}")
    print(f"  {'memory':<18} {s['ram_total_gb']:.0f} GB unified")
    print(f"  {'macOS':<18} {cfg.get('macos','?')}")

    def line(label, value):
        print(f"  {label:<18} {value}")

    print(f"\n{BOLD}Host{RESET}")
    line("cpu", f"{s['cpu']:.1f}%" if s["cpu"] is not None else "--")
    line("gpu", f"{s['gpu']:.0f}%" if s["gpu"] is not None else "--")
    line("ram used", f"{s['ram_used_gb']:.1f} GB / {s['ram_total_gb']:.0f} GB")
    line("ram wired", f"{s['ram_wired_gb']:.1f} GB")
    line("ram free", f"{s['ram_free_gb']:.1f} GB"
                     f"  (+{s['ram_cached_gb']:.1f} GB reclaimable)")
    if s.get("swap"):
        line("swap used", f"{s['swap']['used_gb']:.1f} GB")
    line("thermal", s["thermal"])
    if s.get("ane") is not None:
        line("ane power", f"{s['ane']:.0f} mW")

    print(f"\n{BOLD}Endpoint process{RESET}")
    mem = s.get("srv_mem") or {}
    if mem.get("ok"):
        for label, key in (("weights", "model_weights_bytes"),
                           ("session bank", "session_bank_bytes"),
                           ("generation", "generation_working_bytes"),
                           ("kv cache", "cache_memory_bytes"),
                           ("total", "active_memory_bytes"),
                           ("peak", "peak_memory_bytes")):
            line(label, f"{num(mem.get(key)) / 1024 ** 3:.2f} GB")
    else:
        line("memory", "no telemetry")

    print(f"\n{BOLD}Server{RESET}")
    if not s["server_up"]:
        print(f"  {RED}not running{RESET} - start it with ./start.sh")
        return
    line("state", f"{GREEN}serving{RESET} on {cfg['base']}")
    line("in-flight", str(len(s["in_flight"])))
    line("sessions", str(s.get("sessions_n", 0)))
    sched = s.get("scheduler") or {}
    line("lane", f"{sched.get('active_lane','-')} ({sched.get('scheduler_policy','-')})")
    if s.get("live_tps"):
        line("decode now", f"{s['live_tps']:.1f} t/s")

    logt = s.get("log_totals") or {}
    life = s.get("lifetime") or {}
    if logt or life:
        print(f"\n{BOLD}Tokens generated{RESET}")
        if logt:
            line("output", f"{logt['completion']:,}   (all logged)")
            line("input", f"{logt['prompt']:,}")
            line("total", f"{logt['completion'] + logt['prompt']:,}")
            line("requests", f"{logt['requests']:,}")
        if life:
            line("since restart",
                 f"{int(num(life.get('completion_tokens_total'))):,}")

    print(f"\n{BOLD}Clients{RESET}")
    if s["clients"]:
        for cl in s["clients"]:
            line(cl["ip"], f"{cl['conns']} connection(s)"
                           f"  {', '.join(cl['procs'])[:40]}")
    else:
        line("(none)", "no established connections")

    print(f"\n{BOLD}Connect a client{RESET}")
    print(f"  {'base URL':<18} {cfg.get('lan_url', cfg['base'])}")
    print(f"  {'model':<18} {cfg.get('served_name','')}")
    print(f"  {'api key':<18} {cfg.get('api_key') or '(none - loopback only)'}")
    print(f"\n  {DIM}Live dashboard:  ./status.sh{RESET}")
    print(f"  {DIM}Key only:        ./status.sh --key{RESET}\n")


def print_json(cfg, snap):
    out = dict(snap)
    out.pop("snap", None)          # the whole raw telemetry blob
    out["sessions"] = snap.get("sessions_n", 0)
    out["in_flight"] = [
        {"request_id": r.get("request_id"),
         "age_s": r.get("age_s"),
         "prompt_tokens": r.get("prompt_tokens"),
         "session_id": r.get("session_id"),
         "completion_tokens": (r.get("last_progress") or {}).get("completion_tokens"),
         "decode_tok_s": (r.get("last_progress") or {}).get("decode_tok_s")}
        for r in (snap.get("in_flight") or [])
    ]
    out["live_history"] = len(snap.get("live_history") or [])
    out["config"] = cfg
    print(json.dumps(out, indent=2, default=str))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--power", action="store_true",
                    help="add ANE/GPU power via passwordless sudo powermetrics")
    ap.add_argument("--iterations", type=int, default=0,
                    help="stop after N refreshes (0 = run until Ctrl-C)")
    ap.add_argument("--no-keys", dest="enable_keys", action="store_false",
                    default=True,
                    help="disable the t/m key toggles (display only)")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    if args.help:
        print(__doc__)
        return 0

    cfg = json.loads(os.environ["MTPLX_DASH_CFG"])
    dash = Dashboard(cfg, args)

    if args.once or args.json:
        snap = dash.collect()
        print_json(cfg, snap) if args.json else print_once(cfg, snap)
        return 0 if snap["server_up"] else 1

    dash.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
