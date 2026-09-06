# Live expert containers

Where expert weights physically reside in working memory at runtime, and which
declaration is the deepest "container of multiple experts that reside in
working memory". This is a code-structure map; every measured size and speed
lives in [MEASUREMENTS.md](../MEASUREMENTS.md), design intent in
[PLAN.md](../PLAN.md).

## The deepest container

**`SlotPool.pools: [MLXArray]`** — `Sources/Slotstream/ExpertStore.swift:311`

```swift
// pools, same order as the source's readBatch pieces
public private(set) var pools: [MLXArray] = []
```

One `[slots, dim1, dim2]` MLXArray per weight *piece* (for Qwen:
gate/up/down × weight/scales/biases — `qwenPieceShapes`). Row `s` of each
array is one expert's quantized record, byte-identical to the checkpoint. Slot
↔ expert bookkeeping sits beside it: `map: [ExpertKey: Int]` and its inverse
`keyOf`, plus the CLOCK bits (`refBit`/`pinned`/`hand`).

The pool is **global and shared across all layers** — hot layers borrow from
cold ones (doc comment on the declaration). It is the terminus of every copy:

- **Written** by `ensure` (scatter of miss batches from staging) and `admit`
  (final sweep pass, each layer's fair share by frequency, never during the
  sweep itself — see PLAN §3.3).
- **Resized** by `resize(to:)`, which the governor calls under the generation
  lock; grow gathers the occupied prefix into a bigger allocation, shrink
  frees before allocating cold.
- **Read** by `gatherResident` (sweep staging copies) and directly by the
  decode GEMMs (`MoELayer.cached`, `DS4Model.routedExpertsViaPool`).
- **Reported** as `poolBytes` — the number the governor, the plan, and
  `/api/ps` quote as the pool's size (via `Engine.publishPoolSnapshot`, which
  reads a published copy, never the mutable pool).

Backing memory: allocated as `MLXArray.zeros` in `allocatePools` → MLX/Cmlx
Metal shared storage in unified memory. `preallocate()` proves the backing
store by aliasing each array through `asMTLBuffer(device: noCopy: true)` and
memsetting, faulting the demand-zero pages into RSS. Below `pools` there is
only MLX's own buffer — no further Swift-level container in this repo.

## Inventory of containers

| Container | Declaration | Role |
| --- | --- | --- |
| **The pool (deepest)** | `SlotPool.pools: [MLXArray]` — `ExpertStore.swift:311` | The resident working set of routed experts, shared across layers. |
| Pool references | `Qwen4ExpModel.pool` (`Model.swift:11`), `DS4Model.pool` (`DeepSeek4/DS4Model.swift:46`), `MoELayer.pool` (`Layers.swift:587`), `EngineModel.pool` (`Engine.swift:100–105`) | Handles into the one pool; no storage of their own. |
| Pool snapshot | `Engine._poolSnapshot` (`Engine.swift:278`) | Published size numbers for metadata endpoints; not weights. |
| Checkpoint store | `ExpertStore.refs: [[TensorRef]]` (`ExpertStore.swift:27`) | The checkpoint tensor map — offsets only, no bytes in memory. |
| Staging (raw) | `ExpertStore.allocateStaging → [UnsafeMutableRawPointer]` (`ExpertStore.swift:186–202`) | One aligned `posix_memalign` buffer per piece, host RAM for up to `SLOTSTREAM_EXPERT_LOAD_BATCH` records; filled by parallel `pread`s. |
| Staging (MLX) | `ExpertStore.stagingArrays: [MLXArray]` (`ExpertStore.swift:208–225`) | The raw buffers wrapped zero-copy into MLX arrays; a materialized container of experts that never enters the pool. DS4 twin: `DS4ExpertStore` (`DeepSeek4/DS4Experts.swift:292–374`). |
| Sweep groups | `w: [MLXArray]` in `MoELayer.sweep` (`Layers.swift:721–776`) | Per-group copies of at most a bounded group of experts — either copied out of the pool (`gatherResident`) or read from staging — fed to the grouped GEMM, then discarded; at most two groups in flight. |
| MTP draft head | `ResidentMoE.gp / up / dp` (`MTP.swift:71–73`) | The draft head's `switch_mlp` with every expert of that one layer resident — same math as `MoELayer`, minus the slot pool. |
| Resident trunk | `ResidentWeights.arrays: [String: MLXArray]` (`Weights.swift:57`) | Permanently resident weights; excludes routed experts but holds each layer's shared-expert projections. Has a parity-rig escape hatch (`includeLayerExperts`) to pin routed experts resident. |

Not expert containers: `NgramStore` (PLE n-gram embedding rows), `DiskCache`
(KV states on disk), `WeightStore` (download/disk management).

## Chain of ownership

```
Server / CLI
  └─ Engine.model: EngineModel                    Engine.swift:95–152
       └─ .pool → Qwen4ExpModel.pool              Model.swift:11
            (or DS4Model.pool; built in Engine.bootDS4)
            └─ SlotPool                           ExpertStore.swift:303
                 ├─ pools: [MLXArray]             :311  ← deepest container
                 │    allocated by allocatePools  :340
                 │    filled by ensure / admit    :500, :603
                 │    resized by resize(to:)      :397   (governor path)
                 │    read by gatherResident      :556   and decode GEMMs
                 ├─ map / keyOf / CLOCK bits      :313–317
                 └─ PoolSource → ExpertStore / DS4ExpertStore
                      ├─ refs: [[TensorRef]]      :27
                      └─ staging buffers          :186 → :208 (transient)
```

Governor path: `Governor.apply` takes `engine.withExclusive` →
`engine.model.pool.resize(to: target)` → `engine.publishPoolSnapshot()`; the
resize re-allocates the `pools` arrays above.

## Why `pools` is the deepest container

1. It is the terminus of every copy — staging scatters into it, the sweep
   copies out of it or alongside it, decode reads it directly.
2. It is what the system reports — `poolBytes` is the pool size every surface
   quotes, and resize is the operation the governor performs on it.
3. Its backing memory *is* the pool's working memory — MLX Metal shared
   storage in unified memory, with `preallocate()` demonstrating one-to-one
   aliasing from the Swift `MLXArray` to a `MTLBuffer`.
