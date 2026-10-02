"""A native kernel on a capture: every dumped buffer, compared over what the kernel changed.
Usage: check_capture.py <capture> <hsaco> <entry> <grid> <block> [bytes per slot to compare]"""
import glob, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the kernels directory
from hip_run import Hip
cap, obj, entry, grid, block = sys.argv[1:6]
limit = int(sys.argv[6]) if len(sys.argv) > 6 else 1 << 30
grid = tuple(int(v) for v in grid.split(",")); block = tuple(int(v) for v in block.split(","))
base = glob.glob(os.path.join(cap, "*_args.bin"))[0][:-9]
params = bytearray(open(base + "_args.bin", "rb").read())
pre = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_pre_+*.bin")}
post = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_post_+*.bin")}
h = Hip(); fn = h.module(obj, entry)
ptrs = {o: struct.unpack_from("<Q", params, o)[0] for o in pre}
lo = min(ptrs.values())
pool = np.zeros(max(ptrs[o] + len(pre[o]) for o in pre) - lo + (16 << 20), np.uint8)
at = {}
for o, d in sorted(pre.items(), key=lambda kv: ptrs[kv[0]]):
    at[o] = ptrs[o] - lo; pool[at[o]:at[o] + len(d)] = np.frombuffer(d, np.uint8)
db = h.upload(pool)
for o in pre: struct.pack_into("<Q", params, o, db + at[o])
h.launch(fn, grid, block, bytes(params))
for o in sorted(post):
    n = min(len(post[o]), limit)
    mine = np.frombuffer(h.download(db + at[o], n), np.uint8); gpu = np.frombuffer(post[o], np.uint8)[:n]
    before = np.frombuffer(pre[o], np.uint8)[:n]
    changed = gpu != before
    if not changed.any() and (mine == gpu).all(): continue
    print("+0x%02X: %d bytes changed by the translation; mine identical on %.4f%% of them, %d bytes differ overall"
          % (o, changed.sum(), 100 * (mine[changed] == gpu[changed]).mean() if changed.any() else 100, (mine != gpu).sum()))
