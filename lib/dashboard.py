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
    MTPLX serves the OpenAI API under /v1 but its operational endpoints
    (/health, /metrics, /admin/sessions) hang off the root. Keep both.
    """

    def __init__(self, base, api_key, model_dir=None, pid_file=None):
        self.base = base.rstrip("/")          # .../v1  (OpenAI surface)
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.key = api_key
        self.model_dir = model_dir
        self.pid_file = pid_file

    def _get(self, path, timeout=4):
        """path is relative to the server root, not /v1."""
        req = urllib.request.Request(
            self.root + path,
            headers={"Authorization": f"Bearer {self.key}"} if self.key else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, OSError, TimeoutError):
            return None

    def health(self):
        return self._get("/health")

    def sessions(self):
        return self._get("/admin/sessions")

    def metrics(self):
        return self._get("/metrics", timeout=6)

    def server_pid(self):
        if self.pid_file and os.path.exists(self.pid_file):
            try:
                return int(open(self.pid_file).read().strip())
            except (OSError, ValueError):
                return None
        return None


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
    lw = min(lw, total_width // 2)
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
                                    cfg.get("model_dir"), cfg.get("pid_file"))
        self.hist_cpu = deque(maxlen=80)
        self.hist_gpu = deque(maxlen=80)
        self.hist_tps = deque(maxlen=80)
        self.t0 = time.time()
        self.last_good = 0.0

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
        h = self.server.health()
        s["server_up"] = bool(h and h.get("ok"))
        s["health"] = h
        s["sessions"] = None
        s["metrics"] = None
        s["in_flight"] = None
        s["live_tps"] = None

        if s["server_up"]:
            sess = self.server.sessions()
            if sess and isinstance(sess.get("sessions"), list):
                s["sessions"] = sess["sessions"]
                s["in_flight"] = sum(1 for x in sess["sessions"] if x.get("in_flight"))
            m = self.server.metrics()
            if m:
                s["metrics"] = m
                latest = m.get("latest") or {}
                s["live_tps"] = latest.get("decode_tok_s")
                s["latest"] = latest
                s["recent"] = m.get("recent") or []
                s["tool_counters"] = m.get("tool_parse_counters") or {}
            if s["live_tps"]:
                self.hist_tps.append(float(s["live_tps"]))
        if s["cpu"] is not None:
            self.hist_cpu.append(s["cpu"])
        if s["gpu"] is not None:
            self.hist_gpu.append(float(s["gpu"]))
        return s

    # ── render ───────────────────────────────────────────────────────────────
    def render(self, s):
        width, height = term_size()
        width = max(60, min(width, 200))
        L = []

        cfg = self.cfg
        up = time.time() - self.t0
        hh, rem = divmod(int(up), 3600)
        mm, ss = divmod(rem, 60)

        # ── header ──
        title = f"{BOLD}Qwen3.8-27B Agent{RESET}"
        status = (f"{GREEN}\u25cf serving{RESET}" if s["server_up"]
                  else f"{RED}\u25cf not running{RESET}")
        pad = max(1, width - len(strip_ansi(title)) - len(strip_ansi(status))
                  - len(f"up {hh:02d}:{mm:02d}:{ss:02d}") - 4)
        L.append(f"{title}{' ' * pad}{status}  {DIM}up {hh:02d}:{mm:02d}:{ss:02d}{RESET}")
        L.append(DIM + "\u2500" * width + RESET)

        # ── config line ──
        L.append(f"  {DIM}{truncate(cfg['model_repo'], width - 4)}{RESET}")
        bits = [cfg.get("profile", "?"), f"MTP d{cfg.get('depth','?')}",
                f"think {cfg.get('thinking','?')}",
                f"hist {cfg.get('preserve_thinking','?')}",
                f"KV {cfg.get('kv','?')}",
                f"ctx {cfg.get('context','?')}"]
        L.append("  " + DIM + " \u00b7 ".join(bits) + RESET)
        L.append(f"  {CYAN}{cfg['base']}{RESET}{DIM}   key {cfg.get('api_key_short','')}{RESET}")
        L.append("")

        # ── gauges ──
        def gauge(label, pct, text, width_bar=18):
            if pct is None:
                empty = "\u2591" * width_bar
                return f"  {BOLD}{label:<5}{RESET} {DIM}{empty}   {'--':>7}{RESET}"
            col = color_for(pct)
            return (f"  {BOLD}{label:<5}{RESET} {col}{bar(pct, width_bar)}{RESET} "
                    f"{text:>7}")

        ram_pct = 100.0 * s["ram_used_gb"] / s["ram_total_gb"]
        swap_pct = None
        if s["swap"] and s["swap"]["total_gb"] > 0:
            swap_pct = 100.0 * s["swap"]["used_gb"] / s["swap"]["total_gb"]

        left = [
            f"  {BOLD}SYSTEM{RESET}",
            gauge("CPU", s["cpu"],
                  f"{s['cpu']:.1f}%" if s["cpu"] is not None else "--"),
            # IOAccelerator reports the whole GPU, including WindowServer and
            # browser compositing, so it is labelled as system-wide.
            gauge("GPU", s["gpu"],
                  f"{s['gpu']:.0f}%" if s["gpu"] is not None else "--"),
        ]
        if s.get("ane") is not None:
            left.append(f"  {BOLD}ANE  {RESET} {GREEN}{bar(min(100, s['ane'] / 20), 18)}{RESET}"
                        f" {s['ane']:>5.0f}mW")
        else:
            empty_ane = "\u2591" * 18
            left.append(f"  {BOLD}ANE  {RESET} {DIM}{empty_ane}   n/a{RESET}")

        left.append(gauge("RAM", ram_pct, human_gb(s["ram_used_gb"])))
        left.append(gauge("SWAP", swap_pct,
                          human_gb(s["swap"]["used_gb"]) if s["swap"] else "--"))
        left.append("")
        detail = []
        if s["cpu_user"] is not None:
            detail.append(f"cpu {s['cpu_user']:.0f}u/{s['cpu_sys']:.0f}s")
        if s["gpu_renderer"] is not None:
            detail.append(f"gpu r{s['gpu_renderer']} t{s['gpu_tiler']}")
        detail.append(s["thermal"])
        left.append(f"  {DIM}{'  '.join(detail)}{RESET}")
        left.append(f"  {DIM}gpu = whole system{RESET}")

        # ── server activity column ──
        right = [f"  {BOLD}ACTIVITY{RESET}"]
        inflight = s["in_flight"]
        right.append(f"  {'in-flight':<14} "
                     + (f"{YELLOW}{inflight}{RESET}" if inflight else f"{DIM}0{RESET}"))
        nsess = len(s["sessions"]) if s["sessions"] else 0
        right.append(f"  {'sessions':<14} {nsess}")
        busy = bool(s["in_flight"])
        label = "decode live" if busy else "last decode"
        if s["live_tps"]:
            col = GREEN if busy else DIM
            right.append(f"  {label:<14} {col}{s['live_tps']:.1f} t/s{RESET}")
        else:
            right.append(f"  {'last decode':<14} {DIM}idle{RESET}")

        lat = s.get("latest") or {}
        if lat:
            right.append(f"  {'last prompt':<14} {lat.get('prompt_tokens','?')} tok")
            ttft = lat.get("ttft_s")
            right.append(f"  {'last ttft':<14} {f'{ttft:.2f}s' if ttft else '--'}")
        if s["proc"]:
            right.append("")
            right.append(f"  {'server cpu':<14} {s['proc']['cpu_pct']:.1f}%")
            right.append(f"  {'server rss':<14} {human_gb(s['proc']['rss_gb'])}")

        L += two_col(left, right, width)
        L.append("")

        # ── memory breakdown + recent requests ──
        mleft = [f"  {BOLD}MEMORY{RESET}  {DIM}{s['ram_total_gb']:.0f} GB unified{RESET}"]
        total = s["ram_total_gb"] or 1.0
        rows = (
            ("wired", s["ram_wired_gb"]),        # MLX weights + KV; not reclaimable
            ("app", s["ram_active_gb"]),
            ("cached", s["ram_cached_gb"]),      # reclaimable
            ("compressed", s["ram_compressed_gb"]),
            ("free", s["ram_free_gb"]),
        )
        for label, gb in rows:
            pct = 100.0 * gb / total
            if label == "wired":
                col = RED if pct > 60 else YELLOW if pct > 45 else GREEN
            elif label == "free":
                col = RED if pct < 5 else YELLOW if pct < 12 else GREEN
            else:
                col = DIM
            mleft.append(f"  {label:<11}{human_gb(gb):>7}  {col}{bar(pct, 12)}{RESET}")

        mright = [f"  {BOLD}RECENT REQUESTS{RESET}   {DIM}ctx / decode / tok / stop{RESET}"]
        recent = (s.get("recent") or [])[-8:]
        if not recent:
            mright.append(f"  {DIM}no requests yet{RESET}")
        for r in reversed(recent):
            ctx = r.get("prompt_tokens", 0)
            tps = r.get("decode_tok_s") or 0
            ct = r.get("completion_tokens", 0)
            fr = (r.get("finish_reason") or "")[:9]
            col = GREEN if tps >= 30 else YELLOW if tps >= 15 else DIM
            mright.append(f"  {ctx:>7}  {col}{tps:>6.1f} t/s{RESET}  "
                          f"{ct:>5}  {DIM}{fr}{RESET}")

        L += two_col(mleft, mright, width)
        L.append("")

        # ── sparklines ──
        sw = max(20, width - 16)
        L.append(f"  {BOLD}cpu {RESET}{sparkline(self.hist_cpu, sw)}")
        L.append(f"  {BOLD}gpu {RESET}{sparkline(self.hist_gpu, sw)}")
        L.append(f"  {BOLD}t/s {RESET}{sparkline(self.hist_tps, sw)}")

        L.append("")
        L.append(f"  {DIM}refresh {self.args.interval}s \u00b7 Ctrl-C to exit"
                 f"{'' if self.power and self.power.available else ' \u00b7 --power for ANE'}"
                 f"{RESET}")

        L.append("")
        if not ALT_SCREEN:
            # Piped or redirected: cursor control would be garbage, so just
            # print clean frames.
            body = "\n".join(strip_ansi(x) if not USE_COLOR else x for x in L)
            return body + "\n" + "\u2500" * min(width, 78) + "\n"

        # Pad to a fixed height then clear the remainder, so nothing smears.
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
    print(f"  {'thinking':<18} {cfg.get('thinking')}")
    print(f"  {'memory cap':<18} {cfg.get('memory_limit')} GB")
    print(f"  {'batching':<18} {cfg.get('batching')}")

    print(f"\n{BOLD}Hardware{RESET}")
    print(f"  {'chip':<18} {cfg.get('chip','Apple Silicon')}")
    print(f"  {'memory':<18} {s['ram_total_gb']:.0f} GB unified")
    print(f"  {'macOS':<18} {cfg.get('macos','?')}")

    print(f"\n{BOLD}Live{RESET}")
    def line(label, value):
        print(f"  {label:<18} {value}")
    line("cpu", f"{s['cpu']:.1f}%" if s["cpu"] is not None else "--")
    line("gpu", f"{s['gpu']:.0f}%" if s["gpu"] is not None else "--")
    line("ram used", f"{s['ram_used_gb']:.1f} GB / {s['ram_total_gb']:.0f} GB")
    line("ram wired", f"{s['ram_wired_gb']:.1f} GB  (MLX weights + KV, not reclaimable)")
    line("ram free", f"{s['ram_free_gb']:.1f} GB  (+{s['ram_cached_gb']:.1f} GB reclaimable)")
    if s["swap"]:
        line("swap used", f"{s['swap']['used_gb']:.1f} GB")
    line("thermal", s["thermal"])
    if s.get("ane") is not None:
        line("ane power", f"{s['ane']:.0f} mW")

    print(f"\n{BOLD}Server{RESET}")
    if not s["server_up"]:
        print(f"  {RED}not running{RESET} - start it with ./start.sh")
        return
    h = s["health"] or {}
    line("state", f"{GREEN}serving{RESET} on {cfg['base']}")
    line("generation", h.get("generation_mode", "?"))
    line("profile", (h.get("profile") or {}).get("name", "?"))
    line("in-flight", str(s["in_flight"] if s["in_flight"] is not None else 0))
    line("sessions", str(len(s["sessions"]) if s["sessions"] else 0))
    lat = s.get("latest") or {}
    if lat:
        line("last decode", f"{lat.get('decode_tok_s', 0):.1f} t/s")
        line("last prompt", f"{lat.get('prompt_tokens', 0)} tok")
    print(f"\n  {DIM}Live dashboard:  ./status.sh{RESET}")
    print(f"  {DIM}API key:         {cfg['api_key']}{RESET}")
    print(f"  {DIM}LAN URL:         {cfg.get('lan_url', cfg['base'])}{RESET}\n")


def print_json(cfg, snap):
    out = dict(snap)
    out.pop("health", None)
    out.pop("sessions", None)
    out.pop("metrics", None)
    out.pop("latest", None)
    out["recent"] = [
        {k: r.get(k) for k in ("prompt_tokens", "completion_tokens",
                               "decode_tok_s", "ttft_s", "finish_reason",
                               "mtp_depth", "context_len")}
        for r in (snap.get("recent") or [])[-10:]
    ]
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
