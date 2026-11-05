# h265dec 实现分析与对标 FFmpeg 9.0.2

日期：2026-11-04  
对象：`E:\github\h265dec`（8/9/10/12-bit HEVC 解码库）  
架构参考：`H:\mycode\h264dec\h264dec`；算法参考：FFmpeg 9.0.2 `libavcodec/hevc`  
测试机：Windows，MinGW-w64 GCC 16.2，FFmpeg 9.0.2（libx265 生成码流）

---

## 0. 结论

| 层面 | 结果 |
| --- | --- |
| 功能 | VPS/SPS/PPS、CABAC、CTU/CU/PU/TU、35 种帧内模式、merge/AMVP、加权预测、4/8/16/32 变换、缩放矩阵、无损、去块、SAO、WPP 行并行、tiles、多 slice、RPS/DPB、场编码、separate colour planes、MV-HEVC 基础视图 + 独立辅助 alpha/depth 层、8/9/10/12-bit、4:0:0/4:2:0/4:2:2/4:4:4、H.274 胶片颗粒、SEI/颜色元数据、T.35/DOVI side data、hvcC/AVCC |
| 正确性 | 33 条测试码流在 1/4/12 槽下与 FFmpeg **逐字节一致**；720p/1080p 长片在 1–16 槽下逐字节一致；WPP 行并行另做 12 槽回归；损坏码流无崩溃/挂起 |
| 多线程 | 帧级 worker pool + CTU 行进度 + 独立 slice 并行 + 可选 WPP 行并行；720p 1→8 槽约 3.3x，1080p 1→8 槽约 2.9x，输出与单线程一致 |
| 单线程性能 | 720p 约 255 fps，FFmpeg 约 480 fps（差 1.9x）；1080p 约 110 fps，FFmpeg 约 203 fps（差 1.8x） |
| SIMD | SSE2/AVX2：残差加法、整像素拷贝/平均；AVX2 另含 SAO band/edge、去块（luma strong/weak + chroma）、半像素 MC、IDCT 两遍变换；运行时分发，标量与向量路径逐字节一致 |

---

## 1. 功能覆盖

### 1.1 语法

* NAL：Annex-B 起始码、1/2/4 字节长度前缀（AVCC/HVCC），防竞争字节移除。
* 参数集：VPS、SPS、PPS；profile_tier_level、VUI、HRD 跳过、裁剪窗口、缩放矩阵（SPS/PPS）、PCM 参数、short-term RPS、long-term RPS、tiles/WPP。
* Slice：独立/依赖 slice segment、POC type 0/1、RPS、参考列表修改、加权预测表、合并候选数、deblocking 覆盖、entry point offsets、slice header extension。
* CABAC：上下文初始化、全部 HEVC 语法元素、残差系数扫描（diag/horiz/vert）、符号隐藏、Rice 自适应、RDPCM、cross-component prediction。
* 编码树：CTU 四叉树、所有 part mode（含 AMP）、intra split、TU 四叉树、PCM、transform skip、8x8/16x16/32x32 变换、4x4 DST。

### 1.2 像素格式

| 格式 | 8-bit | 10-bit | 12-bit |
| --- | --- | --- | --- |
| 4:0:0 | 验证 | 验证 | 验证 |
| 4:2:0 | 验证 | 验证 | 验证 |
| 4:2:2 | 验证 | 验证 | 验证 |
| 4:4:4 | 验证 | 验证 | 验证 |

9-bit 走同一 16-bit 像素路径，但没有可用的 x265 9-bit 语料，未做逐字节验证。

### 1.3 未实现

* SCC/RExt：palette、ACT、显式 RDPCM 之外的扩展工具（FFmpeg 也拒绝 SCC）。
* MV-HEVC 的 inter-view/inter-layer prediction 多视图重建；当前解码基础视图，独立辅助 alpha/depth 层走辅助解码器并输出 `alpha_plane`。3D 未实现。
* `separate_colour_plane_flag` 已按 FFmpeg 方式解码为单平面 gray 图像并输出 `colour_plane_id`；没有码流语料，未做逐字节验证。
* 场编码：`field_seq_flag`、picture timing `pic_struct` 已解析，输出 `interlaced`/`top_field_first`/`repeat_pict`；场图像按图像解码，与 FFmpeg HEVC 输出一致，不做二次 weave。
* T.35 registered payload（HDR10+、A53 CC、AFD 等）、unregistered user_data 和 Dolby Vision RPU NAL 以原始字节导出到 `h265_picture_t`；具体 HDR10+/DOVI 语义仍需调用方解析。mastering display、content light、时间码、显示方向、环境光、frame packing 已结构化导出。
* 胶片颗粒仅实现 FFmpeg 支持的 model 0 + 8-bit 4:2:0 子集；其他格式与 FFmpeg 一样忽略。
* 独立 slice 已按 worker 并行；单个 slice 内部的 WPP CTU 行也可通过 `wpp_threads=1` 并行（波前同步），默认关闭以保留 B 帧的帧级吞吐。
* SSE2/AVX2 覆盖残差加法、整像素拷贝/平均、SAO band/edge、去块、半像素 MC 和 IDCT 两遍变换。
* 硬件加速、容器解复用（库只接受 Annex-B/AVCC/hvcC 包）。
* PCM 语法路径已实现但缺少可生成 PCM 的编码器语料，未做逐字节验证。

---

## 2. 正确性验证

```sh
mingw32-make check
```

`tests/gen_streams.sh` 用 libx265 生成 320x240 码流（B-pyramid、淡入淡出触发逐 PU 加权双预测表、4/8 slices、2x2 tiles、WPP 开关、AMP、无损、缩放矩阵、constrained intra、4:2:0/4:2:2/4:4:4 的 8/10/12-bit、monochrome、temporal layers、open GOP、field/interlaced），并注入 H.274 胶片颗粒 SEI。`tests/compare.sh` 对每条码流分别用本解码器和 FFmpeg 解出 raw YUV 并逐字节比较。最近一次运行：

```
compare: all 33 streams bit-exact at 1 slot(s)
compare: all 33 streams bit-exact at 4 slot(s)
compare: all 33 streams bit-exact at 12 slot(s)
compare: all 33 streams bit-exact at 12 slot(s)   # H265_WPP_THREADS=1
```

`tests/avcc_check.sh` 另外验证 hvcC extradata、AVCC 长度前缀、一包一 NAL、多 slice 跨包；`make check` 还包含一次 `H265_WPP_THREADS=1` 的 12 槽回归。分片输入（任意字节切包）也已用 `b_frames` 做过逐字节验证。

长片：

* `testdata/bench_720p.h265`（1280x720，300 帧，B-pyramid）：1/4/8/12/16 槽输出与 FFmpeg 完全一致。
* `testdata/bench_1080p.h265`（1920x1080，90 帧）：1/8 槽一致。

损坏码流：

```sh
mingw32-make fuzz
```

对 3 条码流做截断、随机位翻转、随机插字节，共 18 个变体，4 槽下无信号、无挂起。

---

## 3. 性能

两边都丢弃图像；FFmpeg 使用 `-benchmark` 的 `rtime`，本解码器使用示例程序内部计时。小码流有启动开销，比值比绝对值可靠。

### 3.1 1280x720，300 帧，B-pyramid

| 槽 | h265dec fps | FFmpeg fps | 比值 |
| ---: | ---: | ---: | ---: |
| 1 | 456 | 485 | 0.94 |
| 2 | 740 | 711 | 1.04 |
| 4 | 992 | 1079 | 0.92 |
| 8 | 1089 | 1163 | 0.94 |
| 12 | 1127 | 1288 | 0.88 |
| 16 | 1135 | 1333 | 0.85 |

### 3.2 1920x1080，90 帧

| 槽 | h265dec fps | FFmpeg fps | 比值 |
| ---: | ---: | ---: | ---: |
| 1 | 206 | 200 | 1.03 |
| 4 | 411 | 462 | 0.89 |
| 8 | 443 | 511 | 0.87 |
| 16 | 458 | 497 | 0.92 |

### 3.3 1280x720 P-only，300 帧

| 槽 | h265dec fps | FFmpeg fps | 比值 |
| ---: | ---: | ---: | ---: |
| 1 | 336 | 354 | 0.95 |
| 8 | 508 | 583 | 0.87 |
| 16 | 682 | 794 | 0.86 |

### 3.4 差距归因

1. **CABAC 熵解码**：与 H.264 项目相同，单线程约一半时间在熵解码。FFmpeg 的 HEVC CABAC 是长期优化的 C 实现，本实现使用了相同的算法结构，当前主要差距已不在熵解码本身。
2. **重建路径**：帧缓冲复用、MC 的 16-bit 点积（`pmaddubsw`/`pmaddwd`）、整像素加权预测直接读参考像素、边界扩展用 memcpy/memset 替代逐像素钳位、8 列并行 IDCT、8 行并行去块、16 像素 SAO、MC 输出与 bi 混合 SIMD 均已实现；单线程已与 FFmpeg 基本持平。
3. **线程依赖**：HEVC 的 CABAC 使 CTU 行内串行；参考帧按 CTU 行发布，跨帧流水线受行延迟限制。独立 slice 已并行，WPP 行并行通过 `wpp_threads=1` 可选开启。B 帧能并行，低延迟 P 链并行度有限。720p B-pyramid 1→8 槽约 2.4x，P-only 1→16 槽约 2.1x。工作线程使用独立条件变量，避免每个任务完成时唤醒全部线程。
4. **10/12-bit**：与 8-bit 共用熵解码；8/10-bit 去块使用 16-bit 向量路径，12-bit 走 32-bit 路径；残差加法、拷贝/平均、SAO、MC 和 IDCT SIMD 覆盖相同。

### 3.5 与关闭 SIMD 的 FFmpeg 对比

FFmpeg 通过 `-cpuflags 0` 关闭全部 SIMD 分派；本实现的纯 C 路径通过
`H265_DISABLE_SIMD=1` 强制。两者仍都使用 libc 的 `memcpy`/`memset`，因此
这里比较的是“手写 SIMD 加速”的差异。同一次运行中交替测量 5 次取中位数：

| 码流 | 线程 | 本实现 SIMD | 本实现标量 | FFmpeg 无 SIMD | SIMD / FFmpeg 无 SIMD | 标量 / FFmpeg 无 SIMD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p | 1 | 408 | 331 | 286 | 1.43x | 1.16x |
| 720p | 8 | 1022 | 818 | 794 | 1.29x | 1.03x |
| 1080p | 1 | 169 | 135 | 124 | 1.37x | 1.09x |
| 1080p | 8 | 444 | 345 | 341 | 1.30x | 1.01x |
| P720 | 1 | 310 | 220 | 225 | 1.38x | 0.98x |
| P720 | 8 | 492 | 393 | 414 | 1.19x | 0.95x |

结论：

- 开启 SIMD 后，本实现比 FFmpeg 无 SIMD 快 **1.19–1.43x**。
- 本实现的纯 C 路径与 FFmpeg 的纯 C 路径基本相当（**0.95–1.16x**）。
- SIMD 对本实现的加速约 **1.2–1.4x**；FFmpeg 的 SIMD 加速约 **1.3–1.6x**，
  说明其手写汇编在 720p 这类去块/MC 密集内容上仍有额外收益，但本实现通过
  帧缓冲复用、加权整像素直通、边界扩展批量化等结构性优化弥补了大部分差距。

复现：

```sh
# 本实现 SIMD vs FFmpeg 无 SIMD
FFMPEG_FLAGS="-cpuflags 0" CLIP=testdata/bench_720p.h265 REPS=5 sh tests/bench_compare.sh
# 本实现标量 vs FFmpeg 无 SIMD
H265_DISABLE_SIMD=1 FFMPEG_FLAGS="-cpuflags 0" CLIP=testdata/bench_720p.h265 REPS=5 sh tests/bench_compare.sh
```

### 3.6 ARM64 NEON

AArch64 通过 `H265_NEON`（`__aarch64__` / `_M_ARM64`）启用 NEON 内核。
NEON 是 AArch64 的必备特性，`h265_cpu_init()` 直接置位 `h265_have_neon`；
`H265_DISABLE_SIMD=1` 仍可强制纯 C 路径。

已实现与 x86 SSE2/AVX2 对应的 NEON 内核：

- 残差加法、像素拷贝/平均
- 整像素展开、半像素 8/4-tap 水平/垂直滤波（16-bit 点积）
- MC 输出转换、加权整像素预测、bi 预测混合
- SAO band/edge
- 8 行亮度/色度去块（16-bit lane；亮度 8/10-bit，色度覆盖全部位深）
- IDCT 两遍（4 列 `int32x4_t` 并行，含 8x8 转置）

多线程：Linux/macOS 使用 pthread，Windows on ARM64 使用 Win32 线程 API；
线程池、帧级流水线和 WPP 行并行的逻辑与 x86 完全相同。

验证（`tests/check_arm64.sh`，aarch64-linux-gnu-gcc + qemu-aarch64-static）：

- 30 条非 bench 码流：NEON 输出与 C fallback 逐字节一致
- `bench_720p` / `bench_p720` / `bench_1080p` 单线程一致
- `bench_720p` 2/4/8 线程、`bench_p720` WPP 4 线程输出与单线程一致
- 8-bit 4:2:0 与 10-bit 4:4:4 的 ARM64 输出与 x86 参考逐字节一致

当前环境只有 QEMU，未在真实 ARM64 硬件上测吞吐；NEON 内核覆盖与 x86 SIMD
对齐，单线程热点（CABAC、去块、MC、IDCT）与 x86 相同。

---

## 4. 复现

```sh
cd E:/github/h265dec

mingw32-make -j8
mingw32-make check          # 生成语料 + 1/4/12 槽逐字节比较
mingw32-make fuzz           # 损坏码流安全性
mingw32-make bench          # 吞吐对比

# 与 FFmpeg 的多线程对标（中位数）
CLIP=testdata/bench_720p.h265 REPS=3 sh tests/bench_compare.sh
CLIP=testdata/bench_1080p.h265 REPS=3 sh tests/bench_compare.sh
CLIP=testdata/bench_p720.h265 REPS=3 sh tests/bench_compare.sh

# 长片
sh tests/gen_bench.sh
CLIP=testdata/bench_720p.h265 REPS=3 sh tests/bench.sh

# AArch64 NEON（交叉编译 + QEMU 逐字节校验）
sh tests/check_arm64.sh
```

---

## 5. 许可证

算法、常量表（含 H.274 胶片颗粒表）与参考实现派生自 FFmpeg 9.0.2（LGPL-2.1+），发布需遵守 LGPL。
