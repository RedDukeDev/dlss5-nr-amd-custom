#!/usr/bin/env python3
"""Builds the translation cache shipped with the runtime (cache\\ComputeCache).

On anything other than an NVIDIA card the network's code has to be translated
before it can run, and translating the largest module takes tens of minutes. A
cache shipped beside the runtime removes that wait: the runtime points ZLUDA at
<runtime>\\cache\\ComputeCache, so whatever this script writes there is found
on the first run. package.py copies it into the package.

Adapted from dlss5-image-enhancer-zluda's tools/prepare_cache.py.

The cache is keyed by ZLUDA's version: it only answers the ZLUDA build that
produced it, so it has to be rebuilt with every ZLUDA that is shipped.

What it does:

  1. reads the code modules straight out of nvngx_dlssnr.dll -- the fatbins as
     they sit in the file, not PTX extracted from them. The driver picks which
     embedded module to translate and derives the cache key from that choice,
     so handing it the same bytes is what makes the key match at run time;
  2. translates each module once per GPU target, several at a time, each in a
     process of its own with a cache database of its own;
  3. merges those databases into one zluda2.db and checks that nothing was
     lost on the way.

Usage:
  prepare_cache.py [nvngx_dlssnr.dll] [--driver zluda_real.dll] [--out DIR]
                   [--work DIR] [--targets gfx1100,gfx11-generic,...]
                   [--jobs N] [--keep]

Defaults: the network is third_party\\nvngx_dlssnr.dll, the one package.py
ships; ZLUDA is the one in the runtime's build (runtime\\build\\bin\\zluda); the
cache goes to runtime\\build\\shipped\\ComputeCache, where package.py takes it
from. After build.bat this needs no argument at all.

This is meant to be run once per ZLUDA build, by whoever prepares a release --
not by users.

Why one database per translation
--------------------------------
Several processes writing one zluda2.db can lose rows without reporting
anything: the cache sets a 30-second busy timeout and WAL, and rows still went
missing when four large modules finished near each other. Rather than hope, each
translation here gets a database nobody else touches, and they are merged at the
end -- the schema has a unique index on (hash, compiler_version, zluda_version,
device, backend_key), so merging with INSERT OR IGNORE is exact rather than
approximate.

It also makes checking possible. A private database must hold exactly one module
when its translation reports success; if it holds none, that is the silent loss
happening, and this script says so and retries instead of shipping a cache with
a hole in it.

Every translation leaves a worker.log beside its database, with the driver's
own messages and the memory the translation actually reached. A failure is
reported with its exit code spelled out and the last lines of that log, so it
says why rather than only that.
"""

import argparse
import ctypes
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)          # runtime\\
BIN = os.path.join(PROJECT, "build", "bin")

# Which GPUs the shipped cache covers.
#
# Exact-device builds for the two RDNA3 parts that carry 50% more physical
# VGPRs (Navi 31/32, the RX 7900/7800/7700 desktop cards): only there is the
# matrix-multiply fusion worth turning on, and a family-generic build cannot
# carry that register feature because other members of the family lack it.
#
# One family-wide build for everything else, fusion off -- generic plus fusion
# was measured as a regression. These catch the remaining RDNA3 parts, all of
# RDNA4 and all of RDNA2. A card with no entry of its own still runs: it
# translates on demand the way it does today, only slowly.
NATIVE_TARGETS = ["gfx1100", "gfx1101"]
GENERIC_TARGETS = ["gfx11-generic", "gfx12-generic", "gfx10-3-generic"]

# Ratio above which a fused kernel is considered to have ballooned. Only used
# for the native targets; see llvm_zluda::compile in ZLUDA.
SELECTIVE_FUSE = "2.13"

# The only ZLUDA variables a translation here is allowed to see. The cache key
# carries ZLUDA's version and the target, not the switches that change what the
# translation produces, so an experiment left set in the shell that runs this
# -- a fusion knob, a lowering, a split codegen -- would ship different code
# under the same key, and nothing would ever notice.
ZLUDA_VARIABLES_SET_HERE = {"ZLUDA_TARGET_ARCH", "ZLUDA_CACHE_DIR", "ZLUDA_SELECTIVE_FUSE"}

FATBIN_MAGIC = struct.pack("<I", 0xBA55ED50)

# CUDA_ERROR_NO_BINARY_FOR_GPU. cuModuleLoadData translates and then loads, and
# a module translated for a family this machine does not belong to cannot be
# loaded on it -- so building the gfx12 and gfx10-3 entries on an RDNA3 card
# ends in this code every time, after the translation has succeeded and been
# filed in the cache. Counting it as a failure is what made every foreign
# target look broken.
NO_BINARY_FOR_GPU = 209

# What a worker exits with in that case, distinct from 0 so the log can say it,
# and distinct from 1 so it is not mistaken for a failure.
EXIT_TRANSLATED_FOR_ANOTHER_GPU = 2

# Peak memory of one translation, measured on the largest module of the network
# -- around nine megabytes of PTX with twenty-odd thousand matrix operations,
# which is the worst case these libraries contain. Every translation is assumed
# to grow this far until it is seen to have stopped; each worker.log reports
# the real figure, which is how this number gets corrected.
PEAK_PER_JOB_MB = 2600

# What to leave for everything else. This runs for a long time and nobody should
# have to stop using the machine for it.
RESERVE_MB = 3000

# Windows reports these as large unsigned numbers; spelled out they say what
# happened rather than only that something did.
EXIT_CODES = {
    1: "error, see worker.log",
    3: "abort()",
    0xC0000005: "access violation",
    0xC0000017: "out of memory",
    0xC00000FD: "stack overflow",
    0xC000013A: "interrupted",
    0xC0000409: "fail-fast (a runtime check or a Rust panic)",
    0xE06D7363: "unhandled C++ exception",
}

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


if IS_WINDOWS:
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                 ctypes.POINTER(ProcessMemoryCounters),
                                                 wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                 ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

# Kept for the life of the process on purpose: closing the last handle to a job
# made with KILL_ON_JOB_CLOSE is what ends every process still in it.
_job = None


def kill_children_when_this_process_ends():
    """Ties every translation to this script's own lifetime.

    Stopping this script used to leave its translations running: each one sits
    inside a single call into the driver for up to half an hour, and neither
    Ctrl+C nor the parent's exit reaches it there. A job object that kills its
    members when its last handle closes ends them however this process ends --
    Ctrl+C, a closed window, Task Manager -- because the handle is closed by
    Windows itself, not by code that might not get to run. Children join the job
    by being created from inside it. The program does the same in gui\\main.cpp.
    """
    global _job
    if not IS_WINDOWS:
        return
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    if ok and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        _job = job
    else:
        # Already inside a job that forbids nesting: rare, and not worth failing
        # over. Ctrl+C is still handled below; only a hard kill would leak.
        print("note: translations will not be stopped automatically if this script is killed")


def free_physical_mb():
    """Physical memory free right now, or 0 if the system will not say."""
    if not IS_WINDOWS:
        return 0
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return status.ullAvailPhys // (1024 * 1024)


def process_memory_mb(handle):
    """(current working set, peak working set, peak commit) in MB, or zeros."""
    if not IS_WINDOWS or not handle:
        return 0, 0, 0
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return 0, 0, 0
    mb = 1024 * 1024
    return (counters.WorkingSetSize // mb, counters.PeakWorkingSetSize // mb,
            counters.PeakPagefileUsage // mb)


def room_for_another(running):
    """Whether to start one more translation right now.

    The obvious test -- is there a translation's worth of memory free? -- is
    wrong at the one moment it matters. A translation just started has not
    allocated anything yet, so the memory it is about to take still looks free,
    and the test passes again, and again: the first version of this let sixteen
    start in the same second. Each running translation is therefore charged
    what it has yet to claim, the peak less what it already holds, and a new
    one is admitted only if that still leaves room for it on top.

    Asked again every time one finishes, so the count follows the machine
    instead of a number fixed at the start. One is always allowed.
    """
    if not running:
        return True
    free_mb = free_physical_mb()
    if not free_mb:
        return len(running) < 2          # no reading available: stay cautious
    still_to_claim = sum(max(0, PEAK_PER_JOB_MB - unit.memory_mb()) for unit in running)
    return free_mb - RESERVE_MB - still_to_claim >= PEAK_PER_JOB_MB


def describe_exit(code):
    unsigned = code & 0xFFFFFFFF
    name = EXIT_CODES.get(unsigned)
    shown = "0x%08X" % unsigned if unsigned > 255 else str(unsigned)
    return "%s (%s)" % (shown, name) if name else shown


def extract_modules(snippet_path, out_dir):
    """Writes each fatbin found in the library to out_dir, largest first."""
    with open(snippet_path, "rb") as f:
        data = f.read()

    modules = []
    offset = 0
    while True:
        offset = data.find(FATBIN_MAGIC, offset)
        if offset < 0:
            break
        header_size, = struct.unpack_from("<H", data, offset + 6)
        body_size, = struct.unpack_from("<Q", data, offset + 8)
        total = header_size + body_size
        # A plausible module: it fits, and it is neither empty nor absurd.
        if header_size < 16 or total < 64 or total > 64 * 1024 * 1024 or offset + total > len(data):
            offset += 4
            continue
        modules.append((offset, data[offset:offset + total]))
        offset += total

    if not modules:
        raise SystemExit(
            "no code modules were found in %s.\n"
            "A copy of the network patched for RTX 20/30/40 cards carries none that can be\n"
            "translated -- use the original nvngx_dlssnr.dll." % snippet_path)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for index, (offset, blob) in enumerate(modules):
        path = os.path.join(out_dir, "fb_%03d_%08x.fatbin" % (index, offset))
        with open(path, "wb") as f:
            f.write(blob)
        paths.append(path)

    # Largest first: the long ones then start immediately instead of being left
    # to the end, where they decide when the whole run finishes.
    paths.sort(key=os.path.getsize, reverse=True)
    print("%d modules extracted from %s" % (len(paths), os.path.basename(snippet_path)))
    return paths


def translate_one(driver_path, module_path):
    """Translates one module through the stand-in driver, in this process.

    One process handles one module and one target: ZLUDA reads its environment
    once and remembers the answer, so a second target or a second cache
    directory needs a second process. Each translation is also a whole compiler
    pipeline over a large module, which is the other reason not to do two in
    one process.

    Everything printed here lands in the unit's worker.log.
    """
    # The driver's own dependencies (the HIP runtime) sit beside it.
    directory = os.path.dirname(os.path.abspath(driver_path))
    if hasattr(os, "add_dll_directory") and os.path.isdir(directory):
        os.add_dll_directory(directory)
    cuda = ctypes.CDLL(os.path.abspath(driver_path), winmode=0)

    cu_init = cuda.cuInit
    cu_device_get = cuda.cuDeviceGet
    cu_ctx_create = getattr(cuda, "cuCtxCreate_v2", None) or cuda.cuCtxCreate
    cu_module_load = cuda.cuModuleLoadData

    print("module %s, target %s" % (os.path.basename(module_path),
                                    os.environ.get("ZLUDA_TARGET_ARCH", "?")), flush=True)
    result = cu_init(0)
    if result != 0:
        print("cuInit returned %d: no usable GPU, or the HIP runtime is missing" % result)
        return 1
    device = ctypes.c_int(0)
    result = cu_device_get(ctypes.byref(device), 0)
    if result != 0:
        print("cuDeviceGet returned %d: no GPU was found" % result)
        return 1
    context = ctypes.c_void_p()
    result = cu_ctx_create(ctypes.byref(context), 0, device)
    if result != 0:
        print("cuCtxCreate returned %d" % result)
        return 1

    with open(module_path, "rb") as f:
        image = f.read()
    # One trailing zero: a module held as text is read as a C string, and a file
    # on disk carries no terminator of its own.
    buffer = ctypes.create_string_buffer(image + b"\0", len(image) + 1)
    module = ctypes.c_void_p()
    started = time.time()
    result = cu_module_load(ctypes.byref(module), buffer)
    elapsed = time.time() - started

    # The one number worth keeping from every run: what a translation of this
    # module for this target actually cost. PEAK_PER_JOB_MB is only as good as
    # the largest of these.
    _, peak_ws, peak_commit = process_memory_mb(kernel32.GetCurrentProcess()) if IS_WINDOWS \
        else (0, 0, 0)
    print("cuModuleLoadData returned %d after %.1f s" % (result, elapsed))
    print("peak memory: %d MB resident, %d MB committed" % (peak_ws, peak_commit), flush=True)
    if result == NO_BINARY_FOR_GPU:
        # The translation was done and filed in the cache; only loading it onto
        # this card failed, which is exactly what building for gfx12 or gfx10-3
        # on an RDNA3 machine must do. Not taken on trust either way: the parent
        # counts the rows in this unit's database, as it does for every success.
        print("translated for a GPU other than this one: filed in the cache, not loadable here")
        return EXIT_TRANSLATED_FOR_ANOTHER_GPU
    return 0 if result == 0 else 1


def rows_in(database):
    """How many modules a cache database holds. Missing file counts as none.

    Opened read-write on purpose: the translation that wrote it exits without
    closing it, which leaves its row in the write-ahead log, and folding that
    log back in is a write. A read-only connection can report such a database
    as empty.
    """
    if not os.path.exists(database):
        return 0
    try:
        connection = sqlite3.connect(database)
        try:
            return connection.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error:
        return 0


def translation_environment():
    """This process's environment without the ZLUDA switches nobody chose here."""
    environment = dict(os.environ)
    stray = sorted(k for k in environment
                   if k.upper().startswith("ZLUDA_") and k.upper() not in ZLUDA_VARIABLES_SET_HERE)
    for name in stray:
        del environment[name]
    return environment, stray


class Unit:
    """One module translated for one target, with a database of its own."""

    def __init__(self, target, module, fused, work_dir, index):
        self.target = target
        self.module = module
        self.fused = fused
        self.name = os.path.basename(module)
        self.directory = os.path.join(work_dir, "unit_%03d_%s" % (index, target))
        self.database = os.path.join(self.directory, "zluda2.db")
        self.log_path = os.path.join(self.directory, "worker.log")
        self.size = os.path.getsize(module)
        self.process = None
        self.log = None
        self.started = 0.0
        self.retry = False

    def environment(self, base):
        environment = dict(base)
        environment["ZLUDA_TARGET_ARCH"] = self.target
        environment["ZLUDA_CACHE_DIR"] = self.directory
        if self.fused:
            environment["ZLUDA_SELECTIVE_FUSE"] = SELECTIVE_FUSE
        else:
            environment.pop("ZLUDA_SELECTIVE_FUSE", None)
        return environment

    def start(self, driver, base_environment):
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.started = time.time()
        # A process group of its own, so Ctrl+C reaches this script alone and
        # this script decides what happens to the translations -- rather than
        # every translation receiving it too and ignoring it, stuck inside the
        # driver where Python cannot act on it.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", self.module,
             "--driver", os.path.abspath(driver)],
            env=self.environment(base_environment), stdout=self.log,
            stderr=subprocess.STDOUT, creationflags=flags)

    def memory_mb(self):
        """What the translation holds right now, in MB."""
        if self.process is None or not IS_WINDOWS:
            return 0
        current, _, _ = process_memory_mb(int(self.process._handle))
        return current

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.log:
            self.log.close()
            self.log = None

    def log_tail(self, lines=3):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                text = [line.rstrip() for line in f if line.strip()]
            return text[-lines:]
        except OSError:
            return []

    def peak(self):
        for line in reversed(self.log_tail(6)):
            if line.startswith("peak memory:"):
                return line[len("peak memory: "):]
        return ""

    def finished(self):
        """(ok, note) once the process has exited, otherwise None."""
        if self.process is None or self.process.poll() is None:
            return None
        if self.log:
            self.log.close()
            self.log = None
        elapsed = time.time() - self.started
        peak = self.peak()
        code = self.process.returncode
        if code not in (0, EXIT_TRANSLATED_FOR_ANOTHER_GPU):
            tail = self.log_tail()
            note = "FAILED after %.0f s, exit %s" % (elapsed, describe_exit(code))
            if tail:
                note += "\n" + "\n".join("        | " + line for line in tail)
            return False, note
        # The private database must hold exactly this one module. Anything else
        # is the silent loss this design exists to catch.
        held = rows_in(self.database)
        if held != 1:
            return False, "translated in %.0f s but the cache holds %d rows" % (elapsed, held)
        note = "%.0f s, %s" % (elapsed, peak) if peak else "%.0f s" % elapsed
        if code == EXIT_TRANSLATED_FOR_ANOTHER_GPU:
            note += " (for another GPU)"
        return True, note


def run_units(units, driver, ceiling, base_environment):
    """Runs the work, keeping as many going as the machine turns out to allow."""
    pending = list(units)
    running = []
    done = failed = 0
    total = len(units)
    started_at = time.time()

    try:
        while pending or running:
            while pending and len(running) < ceiling and room_for_another(running):
                unit = pending.pop(0)
                unit.start(driver, base_environment)
                running.append(unit)

            if not running:
                break
            time.sleep(0.25)

            for unit in list(running):
                outcome = unit.finished()
                if outcome is None:
                    continue
                running.remove(unit)
                ok, note = outcome
                done += 1
                if not ok:
                    failed += 1
                    unit.retry = True
                print("  [%3d/%3d] %-16s %-24s %s%s"
                      % (done, total, unit.target, unit.name, note,
                         "" if ok else "\n        -> will be retried"),
                      flush=True)
    except KeyboardInterrupt:
        # Stopped on purpose: end the translations now rather than leave them
        # running with nobody left to collect them.
        print("\ninterrupted: stopping %d translations in progress" % len(running), flush=True)
        for unit in running:
            unit.stop()
        raise SystemExit("interrupted -- the shipped cache was not written; the translations "
                         "that finished are in %s" % os.path.dirname(units[0].directory))

    elapsed = time.time() - started_at
    print("\n  %d translations in %d min %02d s, %d to retry"
          % (total, int(elapsed) // 60, int(elapsed) % 60, failed))
    return [u for u in units if u.retry]


def retry_sequentially(units, driver, base_environment):
    """Second chance, one at a time.

    A unit that came back empty is worth retrying precisely because the failure
    may not be deterministic. Doing the retries alone removes the only variable
    that distinguishes this run from a run that works.
    """
    print("\n--- retrying %d translations one at a time ---" % len(units), flush=True)
    still_bad = []
    for unit in units:
        unit.retry = False
        try:
            unit.start(driver, base_environment)
            unit.process.wait()
        except KeyboardInterrupt:
            unit.stop()
            raise SystemExit("interrupted during the retries -- the shipped cache was not written")
        outcome = unit.finished()
        ok, note = outcome if outcome else (False, "did not finish")
        print("  %-16s %-24s %s" % (unit.target, unit.name, note), flush=True)
        if not ok:
            still_bad.append(unit)
    return still_bad


def merge(destination, units):
    """Merges every unit's database into one, and reports what is in it.

    Started from a copy of the first database rather than a hand-written CREATE
    TABLE, so the real schema comes from ZLUDA itself -- indexes and triggers
    included -- instead of a second copy of it that has to be kept in sync.
    """
    sources = [u.database for u in units if os.path.exists(u.database)]
    if not sources:
        raise SystemExit("no translation produced a cache database")

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    # Fold the first shard's write-ahead log in before copying it: a copy of
    # the main file alone would leave that shard's module behind.
    rows_in(sources[0])
    shutil.copyfile(sources[0], destination)
    connection = sqlite3.connect(destination)
    for index, source in enumerate(sources[1:], start=1):
        alias = "src%d" % index
        connection.execute("ATTACH DATABASE ? AS " + alias, (source,))
        connection.execute(
            "INSERT OR IGNORE INTO modules "
            "(hash, compiler_version, zluda_version, device, backend_key, binary, last_access) "
            "SELECT hash, compiler_version, zluda_version, device, backend_key, binary, "
            "last_access FROM " + alias + ".modules")
        connection.commit()
        connection.execute("DETACH DATABASE " + alias)
    connection.execute("VACUUM")
    connection.commit()

    rows = connection.execute(
        "SELECT device, COUNT(*), SUM(LENGTH(binary)) FROM modules GROUP BY device ORDER BY device"
    ).fetchall()
    held = {device: count for device, count, _ in rows}
    total_modules = total_bytes = 0
    print("\n%s" % destination)
    for device, count, size in rows:
        size = size or 0
        total_modules += count
        total_bytes += size
        print("  %-18s %3d modules  %7.1f MB" % (device, count, size / 1024 / 1024))
    print("  %-18s %3d modules  %7.1f MB" % ("total", total_modules, total_bytes / 1024 / 1024))
    connection.close()
    return held


def main():
    parser = argparse.ArgumentParser(
        description="Builds the translation cache shipped with the runtime.",
        epilog="Run this once per ZLUDA build.")
    parser.add_argument("network", nargs="?",
                        default=os.path.join(os.path.dirname(PROJECT), "third_party", "nvngx_dlssnr.dll"),
                        help="path to nvngx_dlssnr.dll (default: third_party\\nvngx_dlssnr.dll)")
    parser.add_argument("--driver", default=os.path.join(BIN, "zluda", "zluda_real.dll"),
                        help="ZLUDA's driver (default: build\\bin\\zluda\\zluda_real.dll)")
    parser.add_argument("--out", default=os.path.join(PROJECT, "build", "shipped", "ComputeCache"),
                        help="where to write zluda2.db (default: build\\shipped\\ComputeCache, "
                             "where package.py takes it from)")
    parser.add_argument("--work", default=os.path.join(PROJECT, "build", "cache"),
                        help="scratch directory for the modules and the per-translation databases")
    parser.add_argument("--targets", help="comma-separated GPU targets, overriding the defaults")
    parser.add_argument("--jobs", type=int, default=0,
                        help="how many translations at once; 0 (the default) decides from the "
                             "machine, 1 reproduces the old strictly sequential behaviour")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch directory instead of deleting it")
    parser.add_argument("--resume", action="store_true",
                        help="keep the translations an earlier, interrupted run of this same "
                             "command finished in --work, and do only the rest")
    # Used when this script re-runs itself, one process per module per target.
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.worker:
        return translate_one(arguments.driver, arguments.worker)

    # Unlike in a game, the name does not matter here: only cuModuleLoadData
    # is called, never the network, which is what insists on nvcuda.dll.
    for path, what in ((arguments.network, "the network"), (arguments.driver, "ZLUDA's driver")):
        if not os.path.isfile(path):
            raise SystemExit("%s was not found at %s" % (what, path))

    kill_children_when_this_process_ends()
    base_environment, stray = translation_environment()
    if stray:
        print("ignoring %s from the environment: the cache key does not record them, so they "
              "would change the shipped code unnoticed" % ", ".join(stray))

    if arguments.targets:
        native = []
        generic = [t.strip() for t in arguments.targets.split(",") if t.strip()]
    else:
        native, generic = NATIVE_TARGETS, GENERIC_TARGETS
    targets = native + generic

    modules_dir = os.path.join(arguments.work, "modules")
    if os.path.isdir(modules_dir):
        shutil.rmtree(modules_dir)
    modules = extract_modules(arguments.network, modules_dir)

    # One unit per (module, target), largest module first so the long poles are
    # already running while the short ones fill in around them. Within a module
    # the targets are adjacent, which costs nothing and keeps the log readable.
    units = []
    for module in modules:
        for target in targets:
            units.append(Unit(target, module, target in native, arguments.work, len(units)))

    ceiling = arguments.jobs if arguments.jobs > 0 else (os.cpu_count() or 1)
    free_mb = free_physical_mb()
    print("%d modules x %d targets = %d translations" % (len(modules), len(targets), len(units)))
    print("up to %d at a time%s; %d MB free now, %d MB reserved, %d MB allowed per translation"
          % (ceiling, "" if arguments.jobs > 0 else " as memory allows",
             free_mb, RESERVE_MB, PEAK_PER_JOB_MB))
    print("each translation logs to its own worker.log under %s\n" % arguments.work, flush=True)

    # A unit's directory is named by its place in the list, which the same
    # network and targets always reproduce; one whose database holds its module
    # is done. Nothing here checks which ZLUDA wrote it: resume only with the
    # driver that started the work.
    to_run = units
    if arguments.resume:
        to_run = [u for u in units if rows_in(u.database) != 1]
        print("resuming: %d of %d translations already done\n" % (len(units) - len(to_run), len(units)),
              flush=True)

    bad = run_units(to_run, arguments.driver, ceiling, base_environment) if to_run else []
    if bad:
        bad = retry_sequentially(bad, arguments.driver, base_environment)
    if bad:
        for unit in bad:
            print("  %s / %s never landed in the cache (see %s)"
                  % (unit.target, unit.name, unit.log_path))
        raise SystemExit("%d translations could not be cached -- the shipped cache was not "
                         "written" % len(bad))

    held = merge(os.path.join(arguments.out, "zluda2.db"), units)

    # The check the old shape could not make: every target must hold every
    # module. A shortfall here means rows went missing between a successful
    # translation and the merge, and shipping that cache would make a user wait
    # for exactly the module it is missing.
    missing = {t: len(modules) - held.get(t, 0) for t in targets if held.get(t, 0) != len(modules)}
    if missing:
        for target, short in missing.items():
            print("  %-18s is short by %d of %d modules" % (target, short, len(modules)))
        raise SystemExit("the merged cache is incomplete")

    # The runtime reads this before it starts the network, and refuses a GPU
    # the cache holds no translation for instead of translating in the game.
    with open(os.path.join(arguments.out, "coverage.txt"), "w", encoding="utf-8") as coverage:
        coverage.write("# GPU targets the translation cache holds every module of\n")
        for target in targets:
            coverage.write("target %s\n" % target)

    if not arguments.keep:
        shutil.rmtree(arguments.work, ignore_errors=True)
    print("\nEvery target holds all %d modules. package.py ships this cache with the runtime."
          % len(modules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
