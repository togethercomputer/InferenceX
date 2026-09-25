#!/usr/bin/env bash
# together_runner: GPU vendor abstraction.
#
# Everything that differs between an NVIDIA box and an AMD/ROCm box lives here,
# so the runners in local/ stay vendor-agnostic. Source after lib/common.sh.
#
# Provides:
#   vendor_detect                -> echoes "nvidia" | "amd" | "none"
#   vendor_smi                   -> name of the working smi binary
#   vendor_gpu_count             -> number of GPUs on the host
#   vendor_gpu_busy_count <MiB>  -> GPUs with more than <MiB> resident
#   vendor_gpu_names             -> one line per GPU
#   vendor_docker_gpu_flags [ids]-> prints docker flags to expose GPUs
#   vendor_power_sample_cmd <s>  -> host command that streams a power CSV
#   vendor_container_gpu_check   -> in-container command that counts GPUs

VENDOR="${VENDOR:-}"

vendor_detect() {
    if [[ -n "$VENDOR" ]]; then echo "$VENDOR"; return; fi
    # Check AMD first: an AMD box never has nvidia-smi, but some NVIDIA boxes
    # ship an unrelated `amd-smi`-named tool via vendor packages.
    if [[ -e /dev/kfd ]] && command -v rocm-smi &>/dev/null; then echo amd
    elif command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then echo nvidia
    elif [[ -e /dev/kfd ]]; then echo amd
    else echo none; fi
}

vendor_smi() {
    case "$(vendor_detect)" in
        nvidia) echo nvidia-smi ;;
        amd)    command -v amd-smi &>/dev/null && echo amd-smi || echo rocm-smi ;;
        *)      echo "" ;;
    esac
}

vendor_gpu_count() {
    case "$(vendor_detect)" in
        nvidia) nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0 ;;
        amd)    rocm-smi --showid --csv 2>/dev/null | grep -c '^card' \
                  || ls -d /sys/class/kfd/kfd/topology/nodes/*/ 2>/dev/null | wc -l ;;
        *)      echo 0 ;;
    esac
}

# GPUs holding more than $1 MiB (default 2048) — "is this box busy?"
vendor_gpu_busy_count() {
    local thresh="${1:-2048}"
    case "$(vendor_detect)" in
        nvidia)
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
              | awk -v t="$thresh" '$1>t{c++} END{print c+0}' ;;
        amd)
            rocm-smi --showmeminfo vram 2>/dev/null \
              | grep -oE "VRAM Total Used Memory \(B\): [0-9]+" | grep -oE "[0-9]+$" \
              | awk -v t="$thresh" '$1/1048576>t{c++} END{print c+0}' ;;
        *) echo 0 ;;
    esac
}

vendor_gpu_names() {
    case "$(vendor_detect)" in
        nvidia) nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null ;;
        amd)    rocm-smi --showproductname 2>/dev/null | grep -oE "Card Series:.*" | sed 's/Card Series:[[:space:]]*//' ;;
    esac
}

# ---------------------------------------------------------------------------
# AMD render-node mapping.
#
# rocm-smi numbers GPUs by ascending PCI address, so sorting the PRIMARY render
# nodes (those bound straight to a PCI device, not the amdgpu_xcp_* partition
# nodes) by PCI address reproduces its indexing.
# ---------------------------------------------------------------------------
vendor_amd_render_nodes() {
    for d in /sys/class/drm/renderD*; do
        local tgt pci
        tgt=$(readlink -f "$d/device" 2>/dev/null) || continue
        pci=$(basename "$tgt")
        [[ "$pci" == *:*:*.* ]] || continue
        echo "$pci $(basename "$d")"
    done | sort | awk '{print $2}'
}

# vendor_docker_gpu_flags [comma-separated GPU ids]
#
# No ids -> all GPUs.
#
# AMD: we expose device nodes rather than setting ROCR_VISIBLE_DEVICES. The
# InferenceX recipes contain
#     if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
#         export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi
# and ROCr filters FIRST, so after ROCR_VISIBLE_DEVICES=N the container holds
# one device numbered 0; mirroring N into HIP then selects a device that no
# longer exists and the engine dies with "No HIP GPUs are available" for every
# N except 0. Exposing device nodes sidesteps it: the container sees exactly
# the GPUs we grant, renumbered 0..N-1, with both variables unset. It also
# scopes amd-smi to our GPUs, so power sampling excludes other tenants.
vendor_docker_gpu_flags() {
    local ids="${1:-}"
    case "$(vendor_detect)" in
        nvidia)
            if [[ -z "$ids" ]]; then echo "--gpus all"; else echo "--gpus \"device=${ids}\""; fi ;;
        amd)
            local flags="--device=/dev/kfd --group-add video --group-add render"
            mapfile -t _nodes < <(vendor_amd_render_nodes)
            if [[ -z "$ids" ]]; then
                flags+=" --device=/dev/dri"
            else
                local g
                IFS=',' read -ra _ids <<< "$ids"
                for g in "${_ids[@]}"; do
                    local node="${_nodes[$g]:-}"
                    if [[ -z "$node" ]]; then
                        trerr "GPU index $g has no render node (host has ${#_nodes[@]})"
                        return 1
                    fi
                    flags+=" --device=/dev/dri/$node"
                done
            fi
            echo "$flags" ;;
        *) trerr "no GPU vendor detected"; return 1 ;;
    esac
}

# Array form of vendor_docker_gpu_flags, to avoid quoting problems with
# --gpus "device=0,1". Sets the global VENDOR_GPU_FLAGS array.
#   vendor_set_gpu_flags [ids]; docker run "${VENDOR_GPU_FLAGS[@]}" ...
vendor_set_gpu_flags() {
    local ids="${1:-}"
    VENDOR_GPU_FLAGS=()
    case "$(vendor_detect)" in
        nvidia)
            if [[ -z "$ids" ]]; then VENDOR_GPU_FLAGS=(--gpus all)
            else VENDOR_GPU_FLAGS=(--gpus "device=${ids}"); fi ;;
        amd)
            VENDOR_GPU_FLAGS=(--device=/dev/kfd --group-add video --group-add render
                              --security-opt seccomp=unconfined)
            mapfile -t _nodes < <(vendor_amd_render_nodes)
            if [[ -z "$ids" ]]; then
                VENDOR_GPU_FLAGS+=(--device=/dev/dri)
            else
                local g
                IFS=',' read -ra _ids <<< "$ids"
                for g in "${_ids[@]}"; do
                    local node="${_nodes[$g]:-}"
                    if [[ -z "$node" ]]; then
                        trerr "GPU index $g has no render node (host has ${#_nodes[@]})"
                        return 1
                    fi
                    VENDOR_GPU_FLAGS+=("--device=/dev/dri/$node")
                done
            fi ;;
        *) trerr "no GPU vendor detected"; return 1 ;;
    esac
}

# Streams a power/clock CSV on the host; $1 = sample interval seconds.
vendor_power_sample_cmd() {
    local iv="${1:-1}"
    case "$(vendor_detect)" in
        nvidia) echo "nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu --format=csv -l $iv" ;;
        amd)
            # amd-smi watch mode prints a "'CTRL' + 'C' to stop watching" preamble
            # and repeats the header on every tick. Same awk filter as
            # benchmarks/benchmark_lib.sh:start_gpu_monitor so the CSV starts at
            # the header row and contains exactly one.
            echo "amd-smi metric -p -c -t -u -w $iv --csv 2>/dev/null | awk '/^timestamp,/{if(!h){print;h=1};next} h{print}'" ;;
    esac
}

# In-container GPU count check (images carry their own smi).
vendor_container_gpu_check() {
    case "$(vendor_detect)" in
        nvidia) echo "nvidia-smi -L | wc -l" ;;
        amd)    echo "rocm-smi --showid 2>/dev/null | grep -c GPU || echo 0" ;;
    esac
}
