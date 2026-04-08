# `reg_pingpong_256` Comparison

This note compares the four `_256` kernels:

- `reg_pingpong_256`
- `reg_pingpong_256_mma`
- `reg_pingpong_256_colb`
- `reg_pingpong_256_colb_mma`

The goal is to explain why `reg_pingpong_256` is usually faster than the other three, using fair kernel-only timings and the actual implementation differences in `gemm.cu`.

## Bottom Line

`reg_pingpong_256` is usually faster because it has the cheapest end-to-end kernel pipeline:

1. It stages `B` directly with `cp.async` into the final shared-memory layout.
2. It keeps the entire `A`/`B` prefetch path asynchronous.
3. It does not perform any in-kernel transpose of `B`.
4. It uses WMMA fragments instead of a fully software-managed MMA register path.
5. Its epilogue is cheaper than the manual MMA epilogue.

The fair benchmark shows that wrapper-side `B.transpose(...).contiguous()` was not the main reason for the performance gap. Even after timing `_colb` kernels on pretransposed `B`, `reg_pingpong_256` is still usually ahead.

## Fair Benchmark Setup

Fair benchmark file:

- `benchmark_fair.py`

Result file:

- `results/l4-benchmark-20260407-185137.json`

Fairness rule:

- `reg_pingpong_256` and `reg_pingpong_256_mma` are timed directly on row-major `B`
- `reg_pingpong_256_colb` and `reg_pingpong_256_colb_mma` are timed on precomputed `B_col_major`
- transpose cost is measured separately and reported, not included in kernel-only timing

## Kernel-Only Results

### K-means-like shapes

| Shape | `_256` | `_256_mma` | `_256_colb` | `_256_colb_mma` | Separate transpose |
|---|---:|---:|---:|---:|---:|
| `16384x256x256` | `47.30` | `20.14` | `47.38` | `23.90` | `0.0125 ms` |
| `16384x512x256` | `54.90` | `18.99` | `45.39` | `22.84` | `0.0122 ms` |
| `16384x1024x256` | `43.07` | `16.62` | `42.89` | `22.13` | `0.0118 ms` |
| `32768x1024x256` | `28.76` | `17.82` | `45.54` | `24.14` | `0.0116 ms` |

All TFLOPS numbers above are kernel-only.

Observations:

- `_256` is clearly faster than `_256_mma` on all four shapes.
- `_256_colb` ties `_256` on `16384x256x256` and `16384x1024x256`, and wins on `32768x1024x256`.
- `_256_colb_mma` is always much better than `_256_mma`, even without counting transpose cost.
- The separate transpose cost is tiny for these small-`K` shapes, so it does not explain the large MMA gap.

### Larger shapes

| Shape | `_256` | `_256_mma` | `_256_colb` | `_256_colb_mma` | Separate transpose |
|---|---:|---:|---:|---:|---:|
| `4096x4096x4096` | `57.46` | `40.39` | `51.21` | `51.56` | `0.543 ms` |
| `8192x8192x8192` | `49.20` | `40.99` | `46.90` | `46.30` | `2.495 ms` |
| `16384x4096x4096` | `53.82` | `42.72` | `51.29` | `53.10` | `0.583 ms` |
| `4096x4096x16384` | `48.22` | `41.19` | `44.33` | `46.09` | `2.458 ms` |

Observations:

- `_256` is still the strongest kernel overall.
- `_256_mma` narrows the gap on larger problems, but it still loses.
- `_256_colb_mma` becomes competitive on large shapes once the wrapper transpose is removed.
- Even there, `_256` is still hard to beat consistently.

## Factor-By-Factor Comparison

### 1. `B` staging path

`reg_pingpong_256` writes `B` directly into the exact shared layout the WMMA path consumes:

- Prologue copy: `gemm.cu:1176-1183`
- Steady-state prefetch: `gemm.cu:1230-1244`

That is the cheapest possible path in this codebase:

- global `B`
- `cp.async`
- final shared-memory layout
- `wmma::load_matrix_sync`

`reg_pingpong_256_colb` also has a direct path once `B` is already column-major:

- Prologue copy: `gemm.cu:1443-1449`
- Steady-state prefetch: `gemm.cu:1472-1475`

So `_256` and `_256_colb` both have efficient direct staging.

By contrast, `reg_pingpong_256_mma` does not stage `B` directly into its final layout. It first loads a row-major `int4`, then manually transposes those 8 half values into shared:

- Transpose helper: `gemm.cu:1652-1677`
- Prologue call site: `gemm.cu:1819`
- Steady-state call site: `gemm.cu:1842-1845`

That adds extra work on every stage:

- one vector load from global
- then eight scalar shared stores

This is a structural disadvantage relative to `_256`.

### 2. Async overlap

`reg_pingpong_256` keeps both operands on the async path:

- `A` `cp.async`: `gemm.cu:1164-1171`
- `B` `cp.async`: `gemm.cu:1176-1183`
- prefetch for both operands: `gemm.cu:1230-1244`

`reg_pingpong_256_colb` does the same:

- `A` `cp.async`: `gemm.cu:1435-1440`
- `B_col_major` `cp.async`: `gemm.cu:1443-1449`
- prefetch helper: `gemm.cu:1472-1475`

`reg_pingpong_256_mma` keeps `A` async, but `B` is no longer an async shared copy in the same sense:

- `A` prefetch helper: `gemm.cu:1699-1705`
- `B` handled by manual transpose helper: `gemm.cu:1707`

So `_256_mma` gives up one of the main advantages of the rotate-3 pipeline: cheap overlapped staging for both operands.

### 3. Fragment representation

`reg_pingpong_256` uses WMMA fragments:

- fragment declarations: `gemm.cu:1136-1138`
- compute calls: `gemm.cu:1222-1227`, `gemm.cu:1264-1269`, `gemm.cu:1284-1288`, `gemm.cu:1307-1311`, `gemm.cu:1326-1337`

This keeps the fragment format opaque and lets the compiler/runtime lower the matrix instructions from WMMA intrinsics.

`reg_pingpong_256_mma` uses explicit register fragments and explicit MMA PTX:

- explicit registers: `gemm.cu:1785-1787`
- low-level fragment load: `gemm.cu:1600-1613`
- manual MMA loop: `gemm.cu:1616-1649`

That means more software-managed state:

- `a_regs[2][4][4]`
- `b_regs[2][8][2]`
- `c_regs[4][8][4]`

Even though ptxas showed no spills, this still means more explicit instructions and more scheduling burden in the compute path.

### 4. Epilogue cost

`reg_pingpong_256` epilogue:

- `wmma::store_matrix_sync` to per-warp shared scratch: `gemm.cu:1347`
- cooperative warp writeback: `gemm.cu:1353-1364`

This is relatively structured:

- fragment store
- warp-cooperative shared readout
- scalar global write

`reg_pingpong_256_mma` epilogue is fully scalarized from raw accumulator lanes:

- output mapping and scalar values: `gemm.cu:1879-1889`
- per-element global store loop: `gemm.cu:1891-1895` and following lines

This is heavier:

- no WMMA fragment store step
- no compact scratch-tile staging
- direct scalar reconstruction from `c_regs`

This is another reason `_256_mma` loses even after correctness is fixed.

### 5. Why `_256_colb_mma` often beats `_256_mma`

This is the most important clue.

`reg_pingpong_256_mma` reuses the exact same fragment loader and exact same MMA math as `_256_colb_mma`:

- `_256_mma` loader delegates to `_colb_mma`: `gemm.cu:1600-1613`
- MMA math is the same low-level `HMMA16816_F32` path: `gemm.cu:1616-1649` and `gemm.cu:1710-1744`

So the difference between `_256_mma` and `_256_colb_mma` is not mainly in the tensor-core compute loop.

The difference is upstream:

- `_256_colb_mma` receives `B` already in the right global layout
- `_256_mma` must transpose `B` inside the kernel on every stage

That is why `_256_colb_mma` is usually faster than `_256_mma` in the fair benchmark.

### 6. Why `_256_colb` can match `_256`

On some shapes the fair benchmark shows `_256_colb` nearly tied with `_256`, and occasionally better.

This makes sense because:

- both have direct `cp.async` staging
- both use WMMA fragments
- both use the same rotate-3 structure
- both use the same shared-scratch style epilogue

The main difference is logical layout:

- `_256`: `B` is row-major in global and row-major in shared
- `_256_colb`: `B_col_major` is column-major in global and column-major in shared/WMMA

Once wrapper transpose cost is removed, the two kernels are much closer than the original benchmark suggested.

## Resource Usage

Measured from ptxas verbose output during the same kernel family build:

- `reg_pingpong_256`: `240` registers
- `reg_pingpong_256_colb`: `236` registers
- `reg_pingpong_256_colb_mma`: `234` registers
- `reg_pingpong_256_mma`: `234` registers
- spill stores: `0`
- spill loads: `0`

So:

- the big gap is not caused by spills
- `_256` is actually using slightly more registers than the others
- the performance advantage comes from better data movement and a cheaper compute/epilogue path, not from lower register count

## Ranking by Dominant Cause

### Why `reg_pingpong_256` beats `reg_pingpong_256_mma`

1. Direct `B` `cp.async` staging instead of in-kernel transpose
2. Better overlap in the rotate-3 pipeline
3. Less software-managed fragment work
4. Cheaper epilogue

### Why `reg_pingpong_256` often beats `reg_pingpong_256_colb`

1. No need for alternate `B` layout management
2. Slightly simpler fragment/load path
3. Same basic pipeline and epilogue quality, so small implementation details decide the gap

### Why `reg_pingpong_256` often beats `reg_pingpong_256_colb_mma`

1. WMMA path is still cheaper than manual MMA path here
2. `_colb_mma` has the more expensive explicit MMA epilogue
3. `_256` keeps the simpler direct row-major staging path

## Practical Conclusions

- If the goal is best raw performance today, keep using `reg_pingpong_256`.
- If the goal is a low-level MMA kernel that can compete, the next optimization target is not the MMA instruction itself. It is the `B` staging path.
- The current no-transpose MMA kernel pays too much to convert row-major `B` into the shared layout consumed by the `_colb_mma` fragment loader.
- A faster `reg_pingpong_256_mma` likely needs a warp-cooperative transpose/staging strategy for `B` that preserves the pipeline quality of the `_256` kernel.
