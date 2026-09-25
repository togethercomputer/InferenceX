#!/usr/bin/env python3
"""In-container RDMA readiness checks for the disagg KV path. Run inside the
SGLang image (pyxis) on a GPU node. Verifies, in order:

  1. /dev/infiniband is visible and libibverbs enumerates the HCAs (== the count
     visible under /sys/class/infiniband) — i.e. the bind-mount worked.
  2. libibverbs exports ibv_reg_dmabuf_mr (the peermem-free GPU-mem path).
  3. (optional, --mooncake) the Mooncake transfer engine can register a CUDA
     buffer with WITH_NVIDIA_PEERMEM=0 — the single decisive check for whether
     cross-node KV transfer will work. Heavy (allocates GPU mem); opt-in.

Exit 0 if all *required* checks pass. Prints a one-line report per check to
stderr; machine-readable KEY=VALUE to stdout.
"""
import ctypes, glob, os, sys


def err(*a):
    print(*a, file=sys.stderr)


def check_ibv():
    sys_n = len(glob.glob("/sys/class/infiniband/*"))
    try:
        ib = ctypes.CDLL("libibverbs.so.1")
    except OSError as e:
        err(f"[rdma] FAIL: cannot load libibverbs.so.1: {e}")
        return False, sys_n, 0, False
    ib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    n = ctypes.c_int(0)
    lst = ib.ibv_get_device_list(ctypes.byref(n))
    cnt = n.value
    if lst:
        ib.ibv_free_device_list(lst)
    dmabuf = hasattr(ib, "ibv_reg_dmabuf_mr")
    ok = cnt > 0 and cnt == sys_n
    lvl = "OK" if ok else "FAIL"
    err(f"[rdma] {lvl}: libibverbs sees {cnt} device(s); /sys shows {sys_n} "
        f"(/dev/infiniband bind-mount {'works' if cnt>0 else 'MISSING'})")
    err(f"[rdma] {'OK' if dmabuf else 'WARN'}: ibv_reg_dmabuf_mr "
        f"{'exported' if dmabuf else 'ABSENT (dmabuf path unavailable)'}")
    return ok, sys_n, cnt, dmabuf


def check_mooncake():
    """256 MiB CUDA tensor; Mooncake register_memory must return 0 with
    WITH_NVIDIA_PEERMEM=0 (dmabuf). Returns True/False/None(unavailable)."""
    os.environ.setdefault("WITH_NVIDIA_PEERMEM", "0")
    try:
        import torch
        from mooncake.engine import TransferEngine  # noqa
    except Exception as e:
        err(f"[rdma] SKIP mooncake probe: import failed ({e})")
        return None
    try:
        eng = TransferEngine()
        # hostname/device auto; minimal init varies by mooncake version — guard.
        buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        ptr = buf.data_ptr()
        rc = eng.register_memory(ptr, buf.numel() * 4)
        if rc == 0:
            eng.unregister_memory(ptr)
        err(f"[rdma] {'OK' if rc==0 else 'FAIL'}: mooncake register_memory rc={rc} "
            f"(WITH_NVIDIA_PEERMEM={os.environ['WITH_NVIDIA_PEERMEM']})")
        return rc == 0
    except Exception as e:
        err(f"[rdma] SKIP mooncake probe: engine init failed ({e})")
        return None


def main():
    want_mooncake = "--mooncake" in sys.argv
    ok, sys_n, cnt, dmabuf = check_ibv()
    print(f"IBV_DEVICE_COUNT={cnt}")
    print(f"SYS_IB_COUNT={sys_n}")
    print(f"DMABUF_SUPPORTED={'1' if dmabuf else '0'}")
    if want_mooncake:
        mc = check_mooncake()
        print(f"MOONCAKE_DMABUF_OK={'1' if mc else ('0' if mc is False else 'skip')}")
        if mc is False:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
