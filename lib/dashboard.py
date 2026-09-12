#!/usr/bin/env python3
"""
Real-time dashboard for the Qwen3.8-27B agent endpoint.

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
import urllib.error
import urllib.request
from collections import deque

# ── terminal ─────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
ALT_SCREEN = sys.stdout.isatty()


def c(code):
    return code if USE_COLOR else ""


RESET, BOLD, DIM = c("\033[0m"), c("\033[1m"), c("\033[2m")
RED, GREEN, YELLOW, BLUE, CYAN, MAGENTA = (
    c("\033[31m"), c("\033[32m"), c("\033[33m"),
    c("\033[34m"), c("\033[36m"), c("\033[35m"))
HOME, CLR_EOL, CLR_EOS = "\033[H", "\033[K", "\033[J"
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"


def term_size():
    sz = shutil.get_terminal_size((100, 30))
    return sz.columns, sz.lines


# ── sampling helpers ─────────────────────────────────────────────────────────
def run(cmd, timeout=5):
    """Run a command, return stdout or '' on any failure."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
        self.prev = self._read()

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
            p = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3)
            self.available = (p.returncode == 0)
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


def memory_pressure_free_pct():
    out = run(["memory_pressure"], timeout=6)
    m = re.search(r"System-wide memory free percentage:\s+(\d+)%", out)
    return int(m.group(1)) if m else None


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
        from the socket table instead: it answers "who is connected", not
        "which request came from where".
        """
        if not self.port:
            return []
        out = run(["lsof", "-nP", "-iTCP:%d" % int(self.port),
                   "-sTCP:ESTABLISHED"], timeout=5)
        peers = {}
        marker = ":%d->" % int(self.port)
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[8]
            if marker not in name:
                continue                      # server side of the pair
            peer = name.split("->", 1)[1]
            ip = peer.rsplit(":", 1)[0]
            if not ip:
                continue
            entry = peers.setdefault(ip, {"conns": 0, "procs": set()})
            entry["conns"] += 1
            entry["procs"].add(parts[0])
        return [{"ip": ip, "conns": v["conns"],
                 "procs": sorted(v["procs"])} for ip, v in sorted(peers.items())]


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


def human_gb(gb):
    if gb >= 100:
        return f"{gb:.0f}G"
    if gb >= 10:
        return f"{gb:.1f}G"
    return f"{gb:.2f}G"


def two_col(left_lines, right_lines, total_width):
    """Lay two lists of strings side by side."""
    gutter = 3
    lw = max((len(strip_ansi(s)) for s in left_lines), default=0) + gutter
    # Cap the left column so a long line cannot push the right one off-screen.
    lw = max(24, min(lw, (total_width * 62) // 100))
    rw = total_width - lw
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
        s["thermal"] = thermal_state()

        pid = self.server.server_pid()
        s["proc"] = proc_stats(pid) if pid else None

        # ── server ──
        snap = self.server.snapshot()
        s["snap"] = snap
        s["server_up"] = snap is not None
        s["clients"] = self.server.clients() if snap is not None else []
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
    def render(self, s):
        width, height = term_size()
        width = max(64, min(width, 200))
        L = []
        cfg = self.cfg

        up = time.time() - self.t0
        hh, rem = divmod(int(up), 3600)
        mm, ss = divmod(rem, 60)

        # ── header ──
        title = f"{BOLD}Qwen3.8-27B Agent{RESET}"
        status = (f"{GREEN}\u25cf serving{RESET}" if s["server_up"]
                  else f"{RED}\u25cf not running{RESET}")
        clock = f"up {hh:02d}:{mm:02d}:{ss:02d}"
        pad = max(1, width - len(strip_ansi(title)) - len(strip_ansi(status))
                  - len(clock) - 4)
        L.append(f"{title}{' ' * pad}{status}  {DIM}{clock}{RESET}")
        L.append(DIM + "\u2500" * width + RESET)
        L.append(f"  {DIM}{truncate(cfg['model_repo'], width - 4)}{RESET}")
        bits = [cfg.get("profile", "?"), f"MTP d{cfg.get('depth','?')}",
                f"think {cfg.get('thinking','?')}",
                f"hist {cfg.get('preserve_thinking','?')}",
                f"KV {cfg.get('kv','?')}",
                f"ctx {cfg.get('context','?')}"]
        L.append("  " + DIM + " \u00b7 ".join(bits) + RESET)
        L.append(f"  {CYAN}{cfg.get('lan_url', cfg['base'])}{RESET}"
                 f"{DIM}   key {cfg.get('api_key_short','')}{RESET}")
        L.append("")

        # ── host vs endpoint-process memory ──
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
            ane_bar = bar(min(100, s["ane"] / 20), 16)
            left.append(f"  {BOLD}ANE  {RESET} {GREEN}{ane_bar}{RESET} {s['ane']:>5.0f}mW")
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
        left.append(f"  {DIM}{'  '.join(detail)}{RESET}")

        # The endpoint's own MLX allocation, straight from MTPLX's telemetry.
        mem = s.get("srv_mem") or {}
        right = [f"  {BOLD}ENDPOINT PROCESS{RESET}"]
        if mem and mem.get("ok"):
            def gb(key):
                return mem.get(key, 0) / 1024 ** 3

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
            pk = gb("peak_memory_bytes")
            right.append(f"  {'total':<13}{human_gb(total_gb):>7}  "
                         f"{DIM}peak {human_gb(pk)}{RESET}")
            right.append(f"  {'host free':<13}{human_gb(s['ram_free_gb']):>7}  "
                         f"{DIM}(+{human_gb(s['ram_cached_gb'])} cached){RESET}")
        else:
            right.append(f"  {DIM}no telemetry{RESET}")
        L += two_col(left, right, width)
        L.append("")

        # ── concurrent activity + clients ──
        sched = s.get("scheduler") or {}
        act = [f"  {BOLD}CONCURRENT ACTIVITY{RESET}"]
        nfly = len(s["in_flight"])
        act.append(f"  in-flight {YELLOW}{nfly}{RESET}"
                   f"   sessions {s.get('sessions_n', 0)}"
                   f"   lane {sched.get('active_lane') or '-'}"
                   f"   policy {sched.get('scheduler_policy') or '-'}")
        for req in s["in_flight"][:3]:
            lp = req.get("last_progress") or {}
            rid = (req.get("request_id") or "")[:14]
            age = req.get("age_s") or 0
            toks = lp.get("completion_tokens")
            tps = lp.get("decode_tok_s")
            sess = (req.get("session_id") or "")[:12]
            line1 = (f"  {CYAN}{rid}\u2026{RESET} {age:>5.0f}s "
                     f"{DIM}{sess}{RESET} {req.get('prompt_tokens', 0)} ctx")
            if toks is not None:
                line1 += f"  {GREEN}{toks} tok @ {tps:.1f} t/s{RESET}"
            act.append(line1)
            prev = (req.get("prompt_preview") or "").replace("\n", " ")
            if prev:
                act.append(f"    {DIM}\u201c{truncate(prev, width // 2 - 6)}{RESET}")
        if not s["in_flight"] and s["server_up"]:
            act.append(f"  {DIM}idle{RESET}")

        cli = [f"  {BOLD}CLIENTS{RESET}  {DIM}connected TCP peers{RESET}"]
        if s["clients"]:
            for cl in s["clients"][:5]:
                tag = " (this Mac)" if cl["ip"].startswith("127.") else ""
                cli.append(f"  {cl['ip']:<16} {cl['conns']} conn"
                           f"{DIM}  {','.join(cl['procs'])[:18]}{tag}{RESET}")
        elif s["server_up"]:
            cli.append(f"  {DIM}none connected{RESET}")
        else:
            cli.append(f"  {DIM}-{RESET}")
        L += two_col(act, cli, width)
        L.append("")

        # ── token rate chart + lifetime counters ──
        hist = [h.get("tok_s", 0.0) for h in (s.get("live_history") or [])
                if isinstance(h, dict)]
        roll = s.get("rolling") or {}
        chart_w = max(24, width - 34)
        chart_rows, axis_lo, axis_hi = line_chart(hist, chart_w, height=5)

        life = s.get("lifetime") or {}
        logt = s.get("log_totals") or {}
        counters = [f"  {BOLD}TOKENS GENERATED{RESET}  {DIM}all logged{RESET}"]
        if logt:
            counters.append(f"  {'output':<12}{logt['completion']:>13,}")
            counters.append(f"  {'input':<12}{logt['prompt']:>13,}")
            counters.append(f"  {'total':<12}"
                            f"{(logt['completion'] + logt['prompt']):>13,}")
            counters.append(f"  {'requests':<12}{logt['requests']:>13,}")
        else:
            counters.append(f"  {DIM}no log yet{RESET}")
        if life:
            # Telemetry counters reset with the process, so they are labelled
            # as such rather than presented as an all-time figure.
            counters.append(f"  {DIM}{'\u2500' * 12}{'\u2500' * 13}{RESET}")
            counters.append(f"  {DIM}{'since restart':<12}"
                            f"{life.get('completion_tokens_total', 0):>13,}{RESET}")
            if life.get("cancelled_total"):
                counters.append(f"  {DIM}{'cancelled':<12}"
                                f"{life['cancelled_total']:>13,}{RESET}")

        now = s.get("live_tps")
        head = f"  {BOLD}TOKEN RATE{RESET}  {DIM}live, last {len(hist)} samples{RESET}"
        if hist:
            head += (f"  {DIM}min {min(hist):.1f}  avg {sum(hist)/len(hist):.1f}"
                     f"  max {max(hist):.1f}{RESET}")
        rate_lines = [head]
        for i, row in enumerate(chart_rows):
            label = f"{axis_hi:>6.1f} " if i == 0 else " " * 7
            rate_lines.append(f"  {DIM}{label}{RESET}{GREEN}{row}{RESET}")
        foot = f"{axis_lo:>6.1f} " if hist else " " * 7
        rate_lines.append(f"  {DIM}{foot}{RESET}" +
                          (f"{CYAN}now {now:.1f} t/s{RESET}" if now
                           else f"{DIM}idle{RESET}"))
        if not hist:
            rate_lines.append(f"  {DIM}waiting for traffic{RESET}")

        L += two_col(rate_lines, counters, width)
        L.append("")

        # ── history sparklines ──
        sw = max(20, width - 16)
        L.append(f"  {BOLD}cpu {RESET}{sparkline(self.hist_cpu, sw)}")
        L.append(f"  {BOLD}gpu {RESET}{sparkline(self.hist_gpu, sw)}")
        L.append("")
        L.append(f"  {DIM}refresh {self.args.interval}s \u00b7 Ctrl-C to exit"
                 f"{'' if self.power and self.power.available else ' \u00b7 --power for ANE'}"
                 f"{RESET}")

        if not ALT_SCREEN:
            body = "\n".join(strip_ansi(x) if not USE_COLOR else x for x in L)
            return body + "\n" + "\u2500" * min(width, 78) + "\n"

        while len(L) < height - 1:
            L.append("")
        return HOME + "\n".join(x + CLR_EOL for x in L[:height - 1]) + CLR_EOS

    def run(self):
        if ALT_SCREEN:
            sys.stdout.write(HIDE_CURSOR)
            sys.stdout.flush()
        try:
            while True:
                started = time.time()
                s = self.collect()
                sys.stdout.write(self.render(s))
                sys.stdout.flush()
                if self.args.iterations and self.args.iterations > 0:
                    self.args.iterations -= 1
                    if self.args.iterations == 0:
                        break
                time.sleep(max(0.2, self.args.interval - (time.time() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            if ALT_SCREEN:
                sys.stdout.write(SHOW_CURSOR + "\n")
                sys.stdout.flush()


# ── one-shot report (previously status.sh) ───────────────────────────────────
def print_once(cfg, snap):
    s = snap
    print(f"\n{BOLD}Qwen3.8-27B agent endpoint - status{RESET}\n")

    print(f"{BOLD}Configuration{RESET}  (env.conf)")
    print(f"  {'model':<18} {cfg['model_repo']}")
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
            line(label, f"{mem.get(key, 0) / 1024 ** 3:.2f} GB")
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
            line("since restart", f"{life.get('completion_tokens_total', 0):,}")

    print(f"\n{BOLD}Clients{RESET}")
    if s["clients"]:
        for cl in s["clients"]:
            line(cl["ip"], f"{cl['conns']} connection(s)"
                           f"  {', '.join(cl['procs'])[:40]}")
    else:
        line("(none)", "no established connections")

    print(f"\n  {DIM}Live dashboard:  ./status.sh{RESET}")
    print(f"  {DIM}API key:         {cfg['api_key']}{RESET}")
    print(f"  {DIM}LAN URL:         {cfg.get('lan_url', cfg['base'])}{RESET}\n")


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
