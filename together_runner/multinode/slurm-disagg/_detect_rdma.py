#!/usr/bin/env python3
"""Auto-detect the GPU-adjacent RDMA NIC list for --disaggregation-ib-device.

Portable across clusters/HW (B200/GB200/GB300, IB or RoCE): parses
`nvidia-smi topo -m` for each GPU's closest NIC (best PCIe relationship), maps
NIC<k> -> mlx5 name via the topo legend, then keeps only NICs whose port is
ACTIVE/LinkUp and whose link_layer matches the *majority* fabric among the
GPU-adjacent NICs (so InfiniBand clusters drop the Ethernet storage NICs, and
RoCE clusters keep their Ethernet NICs).

Emits to stdout:
  IB_DEVICES=<comma list in GPU order>     (the value for --disaggregation-ib-device)
  IB_LINK_LAYER=<InfiniBand|Ethernet>
and a human-readable report to stderr. Exit non-zero if detection fails so the
caller can fall back to an explicit IB_DEVICES or abort.

Run on the host node OR inside the container (needs nvidia-smi + /sys/class/infiniband).
"""
import os, re, subprocess, sys
from collections import Counter

SYS_IB = "/sys/class/infiniband"
# PCIe relationship preference (closest first). PIX = same switch (ideal).
RANK = {"PIX": 0, "PXB": 1, "PHB": 2, "NODE": 3, "SYS": 4}


def err(*a):
    print(*a, file=sys.stderr)


def port_attr(dev, name):
    try:
        with open(f"{SYS_IB}/{dev}/ports/1/{name}") as f:
            return f.read().strip()
    except OSError:
        return ""


def topo():
    """Return (gpu_to_nics, nic_to_dev): per-GPU NIC indices ranked best-first,
    and NIC index -> mlx5 device name (from the legend)."""
    out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi topo -m failed: {out.stderr.strip()}")
    # nvidia-smi underlines the header with ANSI escapes — strip them so the
    # GPU0/NIC0 labels are plain word tokens.
    lines = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout).splitlines()

    # NIC legend: "  NIC3: mlx5_3"
    nic_to_dev = {}
    for ln in lines:
        m = re.match(r"\s*NIC(\d+):\s*(\S+)", ln)
        if m:
            nic_to_dev[int(m.group(1))] = m.group(2)

    # Header row: locate the column index of each GPU and NIC label.
    header = None
    for ln in lines:
        if re.search(r"\bGPU0\b", ln) and re.search(r"\bNIC0\b", ln):
            header = ln
            break
    if header is None:
        raise RuntimeError("could not find topo matrix header (GPU0..NIC0)")
    cols = header.split()  # ['GPU0','GPU1',...,'NIC0',...,'CPU','Affinity',...]
    col_label = {}  # position-in-cols -> label
    for i, tok in enumerate(cols):
        if re.fullmatch(r"GPU\d+", tok) or re.fullmatch(r"NIC\d+", tok):
            col_label[i] = tok

    gpu_to_nics = {}
    for ln in lines:
        toks = ln.split()
        if not toks or not re.fullmatch(r"GPU\d+", toks[0]):
            continue
        gpu = int(toks[0][3:])
        # The cells align to the header columns after the row label. Cells are
        # the matrix entries (X / NV# / PIX / PHB / SYS ...). Re-split the data
        # row the same way as header so column positions line up.
        # toks[0] is the GPU label; remaining tokens are cells then CPU-affinity.
        ranked = []
        for i, tok in enumerate(cols):
            lbl = col_label.get(i)
            if not lbl or not lbl.startswith("NIC"):
                continue
            # cell for this column = toks[i+1] (row label shifts everything by 1)
            if i + 1 >= len(toks):
                continue
            cell = toks[i + 1]
            if cell in RANK:
                ranked.append((RANK[cell], int(lbl[3:])))
        ranked.sort()
        gpu_to_nics[gpu] = [n for _, n in ranked]
    return gpu_to_nics, nic_to_dev


def main():
    try:
        gpu_to_nics, nic_to_dev = topo()
    except Exception as e:
        err(f"[detect_rdma] ERROR: {e}")
        return 2
    if not gpu_to_nics:
        err("[detect_rdma] ERROR: no GPUs found in topo matrix")
        return 2

    # Collect each GPU's best-rank NIC candidates (those tied at the top rank).
    candidates = []  # (gpu, [dev,...]) usable RDMA devices at best PCIe rank
    for gpu in sorted(gpu_to_nics):
        nics = gpu_to_nics[gpu]
        devs = []
        for n in nics:
            dev = nic_to_dev.get(n)
            if not dev or not os.path.isdir(f"{SYS_IB}/{dev}"):
                continue
            state = port_attr(dev, "state")          # "4: ACTIVE"
            phys = port_attr(dev, "phys_state")       # "5: LinkUp"
            ll = port_attr(dev, "link_layer")         # InfiniBand | Ethernet
            if "ACTIVE" not in state or "LinkUp" not in phys:
                continue
            devs.append((dev, ll))
        candidates.append((gpu, devs))

    # Majority fabric among all GPU-adjacent usable NICs -> drops the off-fabric
    # NICs (e.g. Ethernet storage NICs on an IB cluster).
    fabric = Counter(ll for _, devs in candidates for _, ll in devs)
    if not fabric:
        err("[detect_rdma] ERROR: no ACTIVE/LinkUp RDMA NIC adjacent to any GPU")
        return 3
    majority_ll = fabric.most_common(1)[0][0]

    chosen = []
    for gpu, devs in candidates:
        pick = next((d for d, ll in devs if ll == majority_ll), None)
        if pick is None:
            err(f"[detect_rdma] WARN: GPU{gpu} has no {majority_ll} NIC at best rank "
                f"(candidates: {[d for d,_ in devs] or 'none'}) — skipped")
            continue
        chosen.append(pick)

    if len(chosen) != len(candidates):
        err(f"[detect_rdma] WARN: matched {len(chosen)}/{len(candidates)} GPUs to a NIC")
    if not chosen:
        err("[detect_rdma] ERROR: no NICs chosen")
        return 3

    err(f"[detect_rdma] fabric={majority_ll}  per-GPU NIC: " +
        ", ".join(f"GPU{g}->{d}" for (g, _), d in zip(candidates, chosen)))
    print(f"IB_DEVICES={','.join(chosen)}")
    print(f"IB_LINK_LAYER={majority_ll}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
