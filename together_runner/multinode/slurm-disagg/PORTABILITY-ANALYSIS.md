# slurm-disagg — 坑梳理与可移植性重构设计

整个 2-node SGLang PD-disagg bring-up 过程中遇到的坑、根因、以及让方案在**新集群**
上仍然稳定的重构计划。bring-up 已在 slinky(Slurm-on-k8s, 2×8B200)用 Qwen3-32B
1P1D 跑通并出 benchmark(见 BENCHMARK-RECORD),本文是「跑通 → 可靠固化」的依据。

## 坑分类（关键维度：换新集群会不会爆）

| # | 坑 | 类别 | 新集群风险 |
|---|---|---|---|
| 1 | enroot import temp 必须 ext4 | A 环境硬约束 | 低 |
| 2 | enroot nvidia hook sed 补丁 (`--no-persistenced --no-fabricmanager`) | C 可消除 hack | 中 |
| 3 | bind-mount `/dev/infiniband` | A 环境硬约束 | 低 |
| 4 | router `--jobid=<prefill> --overlap` | A Slurm gang sched | 低 |
| 5 | router 必须挂 model dir | A 配置正确性 | 低 |
| 6 | `WITH_NVIDIA_PEERMEM=0`（强制 dmabuf） | B 需检测 | 高 |
| 7 | `IB_DEVICES` GPU↔NIC 邻接表硬编码 | B 需检测 | 最高 |
| 8 | CUDA graph 必须开 | D 性能默认 | 低 |

## 根因

- **#1** pod 的 `/` 是 overlayfs，`enroot import` 的 `mknod` whiteout 在 overlay-on-overlay
  上被内核禁止 → `aufs2ovlfs ... Operation not permitted`。已用 `ENROOT_TEMP_PATH=/scratch`（ext4）解。正确解，非 hack。
- **#2** `nvidia-container-cli` 想 bind-mount persistenced/fabricmanager 的 socket，pod 里没有 → 启动失败。当前用 `sudo sed -i` 改全局 hook（最脏：sudo + 改系统文件 + 重启重做）。
- **#3** pyxis 不自动透传 IB 字符设备。bind-mount 即正确解。
- **#4** prefill 用满 `gres=gpu:8`，Slurm 不在满载节点上调度新 job；router 不要 GPU，用 overlap step 进 prefill 的 alloc。
- **#5** router 加载 tokenizer，没有本地 model dir 就 404 到 HF。
- **#6** `nvidia_peermem` 模块缺失（驱动 580，仅 gdrdrv+nvidia_fs）；mooncake 默认走 peermem `ibv_reg_mr` 注册 GPU 显存失败 → 3072× `Failed to register memory` → `KVTransferError`。`=0` 强制 dmabuf（`ibv_reg_dmabuf_mr`，驱动 ≥535 支持）。探针：256MiB CUDA tensor，无 env 返回 -202，`=0` 返回 0。
- **#7** `mlx5_9,10,11,12,4,5,6,7` 是本机 GPU0..7→NIC 顺序，且避开 mlx5_0–3（存储 NIC）。物理拓扑，每集群/机型不同（GB200/GB300≠B200）。
- **#8** eager 比 graph 慢 ~22×、65% 超时。默认 ON。

## 决策（已确认）

1. **环境相关项（#6/#7）默认自动探测**：`IB_DEVICES`、`WITH_NVIDIA_PEERMEM` 默认留空 =
   preflight 自动探测填充；手动 export 可覆盖。
2. **#2 尝试消除 → 实验结论：当前栈上无法干净消除,退回加固 sed 补丁。**
   - 用户级 hook 覆盖**不可行**:enroot 4.0.1 `runtime.sh:93` 对 `[系统, 用户]` 两个
     hooks.d **各跑一遍、不按文件名去重**,用户那份 `98-nvidia.sh` 只会追加执行,无法
     替换系统那份(系统那份照样先跑、照样失败)。
   - `ENROOT_SYSCONF_PATH` 重定向**不可行**:实测 `srun --export=ALL,ENROOT_SYSCONF_PATH=<weka>`
     启容器,哨兵 hook 未生效 → **pyxis 忽略该变量**。改 pyxis plugstack.conf 又需 root,与 sudo 同级,无收益。
   - 故保留 `sudo sed` patch,但做成:幂等 + sudo 不可用清晰报错 + patch 后 grep 校验,失败自动还原 .bak。

## 重构计划

1. **新增 `01_preflight.sh`** + 在 `disagg_lib.sh` 加探测函数：
   - `detect_ib_devices`：`nvidia-smi topo -m` / `/sys/class/infiniband/<dev>/device` PCIe
     affinity → 按 GPU 顺序生成 `IB_DEVICES`，过滤非 GPU NIC；与手动值比对告警。
   - `check_ib_ports`：每节点逐端口 `state==ACTIVE` & `phys_state==LinkUp`，记录 `link_layer`
     (IB vs RoCE)、`rate`。
   - `probe_dmabuf`：容器内 256MiB CUDA tensor mooncake register_memory，带/不带
     `WITH_NVIDIA_PEERMEM=0` 各一次 → 自动判定路径；并查 `modinfo nvidia_peermem` + 驱动版本。
   - `check_ibv_in_container`：ctypes `ibv_get_device_list` 数 == 主机枚举数；
     `libibverbs` 是否导出 `ibv_reg_dmabuf_mr`。
2. **`config.env`**：`IB_DEVICES` / `WITH_NVIDIA_PEERMEM` 默认空 → 自动探测。
3. **#2 hook**：试用户级 hook 覆盖；失败退回幂等 sed + 自检。
4. **就绪判定**：`wait_health` 外补「cuda graph capture done」标志，避免半就绪。
5. **收尾**：BENCHMARK-RECORD 收进本目录（README 已相对引用，文件实际在 `~/enroot/`）；补 `.gitignore`。

## 第二轮：消除剩余 hardcode（换集群零改动）

审计发现 IB/peermem 之外还有 5 处 cluster 特定项，处理如下：

| 项 | 原来 | 现在 |
|---|---|---|
| 节点名 `slinky-0/1` | `--nodelist` 写死 | **单个 2-node allocation**(`salloc --no-shell`),prefill/decode/router/bench 全是 `--jobid=$ALLOC --overlap` step;节点由 Slurm 分配,持久化到 `disagg_nodes.env` |
| 分区 `slinky` | 写死 | 空=自动选「含 ≥2 idle GPU 节点」的分区(可 `PARTITION=` 覆盖) |
| `GPUS_PER_NODE`/`TP`=8 | 写死 | **保留写死**(B200 专用,刻意不探测) |
| enroot temp `/scratch` | 写死、假设 ext4 | 00_setup 在节点上自动探测首个可写的非 overlay/tmpfs 目录(`/scratch /raid /var/tmp ...`,可 `ENROOT_SCRATCH=` 覆盖) |
| 路径 `/data/home/johnson/...` | 写死 | 默认 `$HOME/...`;`ENROOT_DIR`/`MODELS_ROOT` 必须在跨节点共享盘(preflight 验证) |
| `_detect_rdma` 取 `ports/1` | 假设单端口 | (已知小限制)多端口 HCA 需扩展;当前取端口 1 |

架构收益:teardown 只需 `scancel` 一个 allocation;所有 step 共享它;再不依赖任何节点名。
`load_nodes`(仅节点)给 preflight 用,`load_resolved`(节点+IB/peermem)给 launch/bench 用——
preflight 每次从干净 config 重新探测,不会把自己上轮产出的 detected.env 误当成用户显式配置。

端到端在 slinky 复测(allocation 模型):Slurm 自动分到 slinky-[0-1],自动探测 IB 列表与
之前手写一致,KV 传输正常。

完成后端到端重跑出 benchmark，再提交（push fork，PR body→md，无 Claude trailer）。

相关记忆：[[disagg-enroot-node-setup]] · [[inferencex-together-runner]]
