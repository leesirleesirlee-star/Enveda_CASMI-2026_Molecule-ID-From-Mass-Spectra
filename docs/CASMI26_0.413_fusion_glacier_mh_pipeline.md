# CASMI26 Fusion + GLACIER + [M+H]+ (LB 0.413) — 可实施方案

**分析对象**：Kaggle notebook `gengsr/casmi26-fusion-glacier-mh-lb-0-413`
("CASMI26 Fusion GLACIER MH LB 0.413")，作者 Gengsr，version 3，lastRun 2026-10-01，
T4 GPU，internet **off**。

**取证方式（全部一手代码，非推测）**：无鉴权的 Kaggle kernel pull API 直接返回了完整
notebook JSON（21 cells，其中 10 个可执行 code cell）：

- `https://www.kaggle.com/api/v1/kernels/pull/gengsr/casmi26-fusion-glacier-mh-lb-0-413`
  → **VERIFIED**（web_fetch / curl 均可，无需 auth）
- notebook 内嵌模块（写在 `/kaggle/working/ours/`，不依赖额外数据集）：
  `fusion_core.py`, `frag_rescore.py`, `eng/casmi_engine.py`, `eng/eng_runner.py`,
  `eng/pv.py`, `eng/pv_fp.py`, `pc/pc_runner.py`, `pc/probe_core2.py` → **全部逐字取得**
- ICEBERG/GLACIER runner 不在 notebook 里，来自数据集；已下载 zip 并逐字取得
  `gl_fuse.py` / `gl_runner.py` / `fuse.py` / `README.md` / `MANIFEST.json` /
  `LICENSE-NOTICE.txt` → **VERIFIED**

本地副本：`.deepworks/tmp/notebook.json`、`.deepworks/tmp/nb/`（拆分的 cells）、
`.deepworks/tmp/embed/`（内嵌模块）、`.deepworks/tmp/glacier/`、`.deepworks/tmp/iceberg/`。

notebook 头部自述：这是原作者打分提交 **56792011** 的可执行复现页，
得分核对于 **2026-10-03**；是"对公开工作的改编"，包含的公开组件：
Huseyin Emre Aksoy(`casmi26-sub-v4b-0-409-on-the-public`) → Seyit Kaan Gunes
(`casmi26-v4n-engine-fusion-union-lb-0-399`) → Wang Penghua
(`casmi26-v29-fusion-popglmh`) + Ahmed Berat Ozer / prvsiyan / megayak /
bobthebot369 / Dmitrii Gluzdov 的数据与模型。

---

## 0. 全局结构（10 个 code cell）

| cell | 作用 |
|---|---|
| 2 | `EMBED` 字典：8 个内嵌模块源码字符串 |
| 4 | `CFG` 全部旋钮 + **build 覆盖行**（见 §4.0） |
| 6 | 离线装 RDKit wheel(2026.03.3)、定位输入、MANIFEST sha256 校验、写 `ours/` |
| 8 | SMOKE / VALIDATION / RERUN 开关；`build_validation()` |
| 10 | **engine 2** 子进程 → `ENG` |
| 12 | **PubChem-only 通道** 子进程 → `PC` |
| 14 | **主引擎**：library + pool + FPNet bank + fe_v4 families + LightGBM ranker |
| 16 | 所有分子 → `BASE` 列表（top 60） |
| 18 | **ICEBERG + GLACIER + FRAG 打分**（只打分，不改序） |
| 20 | `fusion_core.build_submission` → `submission.csv` |

> 关键设计：cell 18 只产出 score dict，**排序全在 fusion_core 里**；
> `DUMP_STAGES=True` 会把 `base/pc/eng/ice/gl/frag/meta` 全 dump 成 JSON，
> 于是融合可以在 CPU 上离线重跑/调参，不动 GPU 阶段。这是很值得抄的工程点。

---

## 1. Candidate generation（候选生成）

### 1.1 主引擎候选池（engine 1）

- **池大小 710,701 structures**（`ahmedberatozer/casmi26-v2-pool`，1.68 GB）。
  **VERIFIED**（数据集描述 + `dmitriigluzdov/casmi26-pubchem-popularity-prior` 描述明写
  `casmi26-v2-pool (710,701 structures)`）。
  组成按 notebook/数据集自述 = **训练集结构 ∪ COCONUT(CC BY 4.0)**；engine 2 另外并上
  **ChEBI + LIPID MAPS**（`prvsiyan/chebi-lipidmaps-casmi26`，60 MB，CC BY-NC-SA 4.0）。
- **质量窗口**：`eng/pv.py` → `class CFG`：

```python
class CFG:
    PPM_WIN = 10.0
    PPM_FALLBACK = 30.0
    INT_FLOOR = 0.002
    MAX_PEAKS = 256
    MZ_TOL = 0.01
    ...
    ANALOG_WIN = 200.0
    N_ANALOG = 80
```

  用法（`eng/casmi_engine.py::compute_channels`）：

```python
cand = pool_window(target, CFG.PPM_WIN)
if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)
```

  `pool_window` = 在 mass-sorted `mass` 数组上二分，`target*(1±ppm/1e6)`。
  → **10 ppm 中性质量窗，空则退化到 30 ppm**。**VERIFIED**
- **target = 该 molecule 所有谱的中性质量中位数**，中性质量由 30 条目 adduct 表换算
  （`pv.ADDUCTS`，含 `[M+H]+ [M+NH4]+ [M+Na]+ [M+K]+ [M-H2O+H]+ [M-2H2O+H]+ [M+2H]2+ [M-H]- [M-H2O-H]- [M+CH2O2-H]- [M+C2H4O2-H]- [M+Cl]- [2M+H]+ [2M+Na]+ [2M-H]- [3M+H]+ …`）。
- **no formula filter on this channel** —— 只按质量窗。formula 只用于后面的「同分子式分组」再排序。
- 有一个值得抄的修正（`formula_mass`，prvsiyan fix）：

```python
def formula_mass(f):
    """ADDED: neutral monoisotopic mass by arithmetic on molecular_formula (prvsiyan fix; immune to
    riken's 0.005 Da precursor precision and gnps adduct mislabels). NaN if an element is untabulated."""
```

  → **当 `molecular_formula` 可用时用它算中性质量，优先于 adduct 换算**；否则退回 adduct。
  这同时被 cell 16 用来补偿「ppm 窗把 121,805 条谱丢掉」类问题。**VERIFIED（代码）**

### 1.2 排序与截断

- 主引擎 LightGBM ranker（`casmi26-v4b-models` 的 `ranker`，多 booster 平均）：
  `rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))`，
  `X` = Engine 特征、`F` = fe_v4 families 特征。**cell 16 VERIFIED（调用形式）；
  特征名清单在数据集内，未取得 → UNVERIFIED**
- 按 tautomer-canonical **InChIKey14** 去重，取 **`TOPN=60`**，保存
  `BASE[mid] = [smiles, keys, lib_max, ranker_scores, formulas, pool_ids]`，
  `pid < 0` 表示「生成候选」（generated）。
- `lib_max` = 该 molecule 在**实测谱库**（`train.parquet`，多来源库）上的最佳熵相似度；
  `LIB_TAU = 0.9` → `lib_max ≥ 0.9` 判定为 **library hit，整张列表后续保持不动**。

### 1.3 PubChem-only 通道（大池，105,875,489 structures）

- 数据集 `ahmedberatozer/casmi26-pubchem-tier`：**105,875,489 structures**，7.21 GB，
  mass-sorted，`CID-SMILES + CID-Mass`，**stereo stripped，只留 organic CHNOPS+halogens，
  150–1250 Da**。**VERIFIED**（数据集描述）
- 两段打分（`pc/probe_core2.py`，逐字）：

```python
PC_N1 = int(os.environ.get('CASMI_PC_N1', 5000))   # ours: overridable from CFG['PC_N1']
PPM = 10.0
K = 25
...
#   3. pass 1 (src/casmi/pubchem.py tier): partial f.z on the ECFP4 block of the selected bits, keep top PC_N1 = 1000.
#   4. pass 2: full selected-bit fingerprints, full f.z, sort descending.
#   5. walk down the f.z order, compute the metric score key (tautomer-canonical InChIKey14); keep a structure only if
#      its key is NOT in the candidate pool (train U COCONUT) and not already listed; stop at 25.
```

  → **±10 ppm 窗内不抽样**，pass1 用 ECFP4 block 近似打分取 top `PC_N1`（CFG 设 5000），
  pass2 用完整 selected bits 精确打分，沿序下行，**排除已在池中的 key**，
  最多吐 **25 条 PubChem-only 结构**（`K=25`）。f.z 同时记录以便后面做门控。
- CFG：`USE_PC=True, PC_N1=5000, PC_WORKERS=4`。**VERIFIED**

### 1.4 Engine 2（第二套独立候选池 + 排序）

- `eng/casmi_engine.py::build_pool`：`COCONUT prvsiyan 指纹` ∪ `bio_fp.npy`(ChEBI+LIPID MAPS，
  按 key 去重) ∪ 自己对 `train.parquet` 结构现算指纹；指纹 = Morgan r2/r3(4096) + RDKitFP(2048)
  + MACCS，按 `fp_bits.npy` 选出 **6,930 bit** 后 `np.packbits`。**VERIFIED**
- 同样的 `pool_window(target, 10 ppm → 30 ppm fallback)`。
- 两个 ranker（`W_A=(0.35,0.55)` × `SEEDS=(0,1)`，HistGradientBoosting，共 4 个 GBM）+
  prvsiyan ranker（`W1_PRIORS=(0.30,0.60)` × 4 seeds，共 8 个）按 **rank blend**
  `blend_scores(a,b,wa=0.88)` 融合；再 `ORDER_K=80` → 去重 → **`TOPK=40`**。
  **VERIFIED**（`eng/eng_runner.py` 逐字）
- 通道包含 analog propagation（`ANALOG_WIN=200.0` Da，`N_ANALOG=80/200`，`SIM_POWER=3.0`）
  与 MetFrag-lite `explain_score`。

### 1.5 每个 query 最终候选数

| 阶段 | 数量 |
|---|---|
| BASE（engine 1 排序后） | **top 60**（`TOPN=60`） |
| ICE/GL 实际打分的候选 | 上述 60 中**同分子式且组内 ≥2 个**的成员（`ice_candidates`） |
| + engine 2 并集 | `ICE_UNION_K=40`（engine2 top-40，key 不在 BASE 的加入） |
| + PubChem-only（门控通过时） | 最多 **25** |
| engine 2 列表 | top **40**（`ENG_TOPK=40`） |
| 提交 | 每个 molecule **≤ 25** SMILES，`;` 连接 |

**VERIFIED**（`ice_candidates` 逐字）：

```python
def ice_candidates(smis, formulas, top_n=40):
    """SMILES worth predicting for one molecule: members of same-formula groups with >= 2 members inside the top_n
    (the only candidates rerank can move). Keeps ranked order; duplicates removed."""
```

---

## 2. GLACIER

### 2.1 是什么

**GLACIER** = Coley group `ms-pred` 的**单阶段（DETR 式）MS/MS 前向预测网络**：
Graphormer backbone + 可学习 object query 做「断裂位点检测」+ 强度头，
Graph Learning of Atomic Component Instances via End-to-End Recognition。
论文 arXiv:2606.29161；MassSpecGym top-1 retrieval 70.0%（无 CF 69.7%），
比两阶段基线约 **8× 推理加速**。**VERIFIED**
（`https://ar5iv.labs.arxiv.org/html/2606.29161`，`https://github.com/coleygroup/ms-pred`）

**在这个 pipeline 里 GLACIER 的用法 = 前向碎片谱预测 + 谱相似度打分**（不是 likelihood、
也不是生成候选）：对每个候选分子预测一张 MS/MS，与实测谱算 **entropy similarity**，
得到标量 `gl_ex_mean`，作为**同分子式组内的第三个 z-score 项**。

### 2.2 打包与离线化（可抄的工程量）

Kaggle 数据集 **`ahmedberatozer/casmi26-glacier`**，61,108,325 bytes（58.3 MB）：

- `ckpt/glacier.pt` — 60,531,887 bytes；**264 tensors / 15,108,154 params**；
  由官方 "GLACIER_checkpoint.zip → best.ckpt"（sha256 `5a47cecca707d3ab…`）
  **只保留 `{hyper_parameters, state_dict}` 重存，张量 bit-identical fp32**
  （去掉 optimizer/trainer/scheduler）。
- `gl_runner.py`（39,880 B）— `ice_runner.py` 的 fork，**完全相同的 CLI / 输入格式 /
  job 构造（adduct / instrument token / CE bucket）/ 打分定义 / chunking / 预算 / 断点续写 /
  meta JSON / 失败行为**，只把模型加载+预测换成 GLACIER。
- `gl_fuse.py`（7,533 B）— notebook 侧胶水：`run_gl`（起子进程，永不抛异常，失败返回 `{}`）
  与 `rerank_multi`（多 re-scorer 版 `rerank`）。
- `src/ms_pred/` — ms-pred 推理子集，**commit `708148c2a8eb`**，**MIT**
  （`Copyright (c) 2023 Samuel Goldman`），只保留下载推理需要的模块。
- `shim/dgl`、`shim/torch_scatter`、`shim/LinSATNet` — **纯 torch 替身**
  （LinSATNet 独立重写为 equality-constrained Sinkhorn top-k 投影，只用于断点选择）。
  → Kaggle 镜像里没有 `dgl` / `torch_scatter` 也能跑，这是"离线可运行"的关键。
- **不带 wheel**：复用 ICEBERG 数据集的 `wheels/*.whl.ice`（其中 **RDKit 2025.3.6**）。
  刻意命名为 `*.whl.ice`，因为 notebook 用 glob `/kaggle/input/**/rdkit-*.whl` 装主进程的
  RDKit 2026.03.3，绝不能被这里污染。**VERIFIED（LICENSE-NOTICE.txt / MANIFEST.json）**

**License**：GLACIER 权重本身下载页无独立 licence，按 ms-pred 仓库 **MIT** 使用；
数据集整体标注 "Other (specified in description)"。**VERIFIED（LICENSE-NOTICE.txt 逐字）**
官方权重来源 = ms-pred README 的 Dropbox 链接 "checkpoint of GLACIER trained on the
MassSpecGym dataset"。**VERIFIED（ms-pred README）**

### 2.3 打分定义（`gl_runner.py` 头部 docstring，逐字）

```
Score = the A1 panel definition `gl_ex_mean` (... = the ice_ex_mean definition):
  covered spectrum : mode +1 and adduct in {[M+H]+, [M+Na]+}
  instrument token : 'Orbitrap' if instr_family(instrument_type) == 1 else 'QTOF'
  CE               : units == 'NCE' -> mean of the numbers in collision_energy_orig, else mean(collision_energy_ev)
                     (NaN if empty); bucket = np.round(ce / 5) * 5, min 5; NaN -> 'sum' prediction (CE 20+40+60 merged)
  molecule         : predict_smis_joint.prepare_entry canonicalisation (valid atoms only, RemoveHs, <= 100 heavy
                     atoms, MolToSmiles); rejected -> null
  prediction       : TreeProcessor.featurize_tree -> IntenDataset.collate_fn -> JointModel.predict_inten_frag_batch
                     (collision_engs = CE bucket, instrument = token, adduct as given, precursor 0) -> drop rows with
                     intensity or m/z <= 0 -> merge identical m/z (round 4) by MAX -> top 100 -> sort by m/z ->
                     max 1 (float32)
  query            : raw peaks -> clean_peaks(prec, floor 0.001, top 512, <= prec + 2) (library cache, float32) ->
                     clean_peaks again (score.py) -> prep_query(floor 0.002, top 256, power 1, entropy weighting)
  similarity       : engine entropy_sim, 0.01 Da / 20 ppm;   gl_ex_mean = mean over the molecule's covered spectra
                     that have a prediction (NaN sims skipped).
```

常数（逐字）：`COV = ('[M+H]+', '[M+Na]+')`、`SUM_CE = (20.0, 40.0, 60.0)`、
`FLOOR, TOPK, POWER, ENTW, TOL, PPM = 0.002, 256, 1.0, True, 0.01, 20.0`、
`RDKIT_REQ = '2025.03'`。

批大小：`batch_size_for(N) = max(1, min(bs_max, budget // N²))`，`budget = 40*40*32`，`bs_max=32`
（`ATOM_BUDGET_PER_BS = 40 * 40`）。失败保护：`FAILFAST_N = 64`。

### 2.4 模型规模 / 运行

- **15.1 M 参数 / 58 MB fp32**，T4 上跑 400 个 molecule 的匹配候选。
- 默认 `--budget` 在本 notebook 设为 **`GL_BUDGET=4000` s**；ICEBERG 参考文档给的是
  ICE 在 T4 / N=60 / 400 molecule 约 **25–30 min**（`~25–32 predictions/s`），
  论文称 GLACIER 约 **8× 推理加速**。**VERIFIED（数值）+ LIKELY（换算）**

---

## 3. "MH" 是什么

**MH = `[M+H]+`（质子化正离子）—— 不是 molecular formula、不是 MetFrag、不是 MSHub。**

**VERIFIED**，三条独立证据：

1. notebook 第一行 markdown 标题，逐字：`# CASMI26 Fusion + GLACIER [M+H]+`
2. notebook 自述，逐字：
   > Our v8 change restricts GLACIER to **[M+H]+** spectra; that idea also appears in
   > Wang Penghua's public work.
3. cell 18 代码，逐字（这就是 "MH" 的全部实现，只有两行）：

```python
# v8 knob (v29/v27, public LB +0.006): GLACIER scores [M+H]+ spectra only
gl_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items]
gl_items = [it for it in gl_items if it['spectra']]
```

即：GLACIER runner 本身支持 `[M+H]+` 和 `[M+Na]+` 两类 covered spectrum，
**这个 notebook 在送进 GLACIER 之前把 `[M+Na]+` 谱全部过滤掉**，
只有 `[M+H]+` 的 molecule 才会拿到 GLACIER 分数（拿不到 = `{}` = 该项 z 记 0，退回 ICEBERG）。
代码注释标注该改动的公开榜增益为 **+0.006**（与榜分噪声 ±0.006 同一量级）。
版权归属：Wang Penghua `casmi26-v29-fusion-popglmh`。

> 注意区分：pipeline 里确实有一个 MetFrag 风格项，但它叫 **`frag` / `frag_rescore.py`**，
> 不是 MH。`frag_rescore.py` 的 docstring 说它是"论坛消融得到的 241 结构同分子式面板
> **+0.065 MRR**"的项，配置里以 `FRAG_LAM=0.5, FRAG_MODE='gap'` 使用。

---

## 4. Fusion / ranking

### 4.0 本 notebook 实际生效的 CFG（**最关键的一段**）

cell 4 定义 `CFG` 后，最后一行是 build 期覆盖 —— **这才是 0.413 那一次的实际配置**：

```python
CFG.update({'VERSION': 'v4b-libgate-pop025', 'POP_MU': 0.25, 'FRAG_LAM': 0.5, 'FRAG_MODE': 'gap',
            'ICE_LAM_LIB': 0.0, 'GL_LAM_LIB': 0.0})   # --set overrides from build_notebook.py
```

默认值里 `POP_MU=0.0, FRAG_LAM=0.0`，被覆盖成 **`POP_MU=0.25`**、
**`FRAG_LAM=0.5` 且只在 ICE 覆盖不到的分子上用（`gap`）**、
**library hit 的 ICE/GL 权重强制为 0**。**VERIFIED（逐字）**

其他相关默认（未被覆盖）：

```python
TOPN=60, FP_BANK='full1',
LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2,4,6,8,10], SLOTS_GENTLE=[4,8,12,16,20],
USE_PC=True, PC_N1=5000, PC_WORKERS=4,
USE_ICE=True, ICE_LAM=1.0, ICE_BUDGET=5400, ICE_PC=True, ICE_UNION=True, ICE_UNION_K=40,
USE_GL=True, GL_LAM=1.0, GL_BUDGET=4000,
USE_ENG=True, ENG_W_PV=0.88, ENG_W_A=[0.35,0.55], ENG_SEEDS=[0,1], ENG_N_ANALOG=200,
ENG_ORDER_K=80, ENG_TOPK=40,
FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FUSE_ENG_K=40,
POST_ICE_LAM=None, POST_GL_LAM=None, FUSE_SKIP_LIB=None, FILL_25=False,
POP_LIB_OFF=False, FRAG_LIB_OFF=False,
SMOKE_N=12, SMOKE_ICE_BUDGET=300, FULL_ON_COMMIT=False, DUMP_STAGES=True, STAGE_TIMEOUT_S=4*3600,
```

`fusion_core.DEFAULTS` 侧：`LIB_TAU=0.9, REL_TH=600.0, TOPN=60, ICE_LAM=1.0, GL_LAM=1.0,
ICE_PC=True, FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FALLBACK='CCO', POP_MU=0.0,
FUSE_ENG_K=40, FRAG_LAM=0.0, FRAG_MODE='all'`。**VERIFIED**

> ⚠️ 注意本 notebook 用 **`ICE_LAM=GL_LAM=1.0`**（非库命中时），而两个数据集 README
> 的推荐是 **0.5 / 0.5**。这是一个被上榜配置改过的旋钮。

### 4.1 三个 stage

`fusion_core.build_submission` 的 docstring 逐字：

```
  stage A  z(ranker) + ICE_LAM z(ICEBERG) + GL_LAM z(GLACIER) inside same-formula groups (base + PubChem lists)
  stage B  gated PubChem merge: untouched if lib_max >= LIB_TAU, else PubChem-only structures into fixed slots
  stage C  weighted reciprocal-rank fusion with engine-2 lists, then the ICE/GL re-rank again on the fused list
```

**Stage A**（同分子式组内 z 线性融合，逐字核心）：

```python
def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
    ...
    fused = _z([scores[i] for i in idx])
    n_act = 0
    for v, lam, c in cols:
        if c >= 2:                                   # n < 2 -> z 0 (fuse._z); skip the no-op addition
            zi = _z(v)
            fused = [a + lam * b for a, b in zip(fused, zi)]
            n_act += 1
    if n_act >= 2 and tie_eps > 0:
        fused = [round(x / tie_eps) for x in fused]  # monotone: never reverses an order, only merges float noise
    new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
    for slot, src in zip(idx, new):                  # idx is ascending = the group's slots
        order[slot] = src
```

z 的定义（pandas `groupby.transform` 语义，逐字）：

```python
def _z(vals):
    """= fuse._z: pandas groupby-transform z: (x - mean) / std(ddof=1) over non-missing; missing / std 0 / n < 2 -> 0."""
```

**三个必须抄对的细节**：
1. **只在 top_n 内的同分子式组内部重排，且 slot-preserving**（成员只在自己原来的
   位置上互换）——这样不会把一个弱候选整体抬到前面。
2. **组内至少要有 ≥2 个被该 re-scorer 覆盖的成员**才动（`min_covered=2`）。
3. 多个 re-scorer 时用 `tie_eps` 网格 round，避免"ICE 和 GL 都反对 ranker 导致
   `±0.7071*(1-0.5-0.5)=0`"这类**精确算术平局被浮点噪声随机打破**。
4. `lam=0` 且 `covered` 时该组不动 → 这就是 **library hit（`lib_max≥0.9`）用
   `ICE_LAM_LIB=GL_LAM_LIB=0.0` 冻结实测谱命中的实现方式**。

顺序（cell 18 + fusion_core）：`popularity_reorder`（若开）→ `ICE rerank` → `GL rerank_multi`
→ `frag_rerank`（gap 模式只在 ICE 无覆盖时）。

**Stage B — 门控 PubChem 合并**（逐字）：

```python
bp = p.get('best_pool_fz')
rel = p['pc_fz'][0] - bp if bp is not None and np.isfinite(bp) else 1e9
aggressive = rel > c['REL_TH']
ms['aggressive' if aggressive else 'gentle'] += 1
final = merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if aggressive else c['SLOTS_GENTLE'])
```

```python
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    """PubChem-only structures (not already in base) go into `slots` (1-based); base fills the rest."""
```

- `lib_max ≥ 0.9` → **完全不动**（`untouched`）。
- 否则把 PubChem-only 结构**插进固定槽位**：f.z 明显超过池内最优（`rel > REL_TH=600`）
  → 激进槽 `[2,4,6,8,10]`；否则温和槽 `[4,8,12,16,20]`。其余位置由 base 顺序填。
- **门控用的是重排前的 `pc_fz[0]`**（代码注释：`the gate keeps the ORIGINAL top f.z,
  so gate decisions do not move`）——门控与重排解耦，避免反馈环。
- PubChem 通道近乎全错，所以只给 5 个槽，而不是整表替换。

**Stage C — 加权倒数排名融合（RRF）+ 复排**（逐字）：

```python
def fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n):
    """sum 1/(krr + r) over the v4 list plus alpha/(krr + r) over the engine list, keyed by the metric key."""
```

`alpha = FUSE_ALPHA = 0.6`，`krr = FUSE_KRR = 3.0`，`n = FUSE_N = 40`，
engine 只取前 `FUSE_ENG_K = 40`；融合完**再跑一次 ICE / GL 复排**（`POST_ICE_LAM/POST_GL_LAM`
默认沿用 `ICE_LAM/GL_LAM`）。最后 `FILL_25=False` → 不补齐到 25 条，短列表就是短的。

### 4.2 融合用到的特征 / 通道总览

| 通道 | 信号 | 用法 |
|---|---|---|
| LightGBM ranker（engine1） | `rank_score([X,F])` | z 融合的主项（每个同分子式组内） |
| 实测谱库匹配 | `lib_max`（entropy sim） | **硬门控** `LIB_TAU=0.9` 冻结；不参与 z |
| FPNet（谱→指纹）bank | `FP_BANK='full1'`（v4m，训练用了全部 fold）；PubChem 通道用 A+B | 产出门控分 `f.z` 与 best_pool_fz |
| ICEBERG 2.1 | `ice_ex_mean` | z 项，`ICE_LAM=1.0` / 库命中 0.0 |
| **GLACIER** | `gl_ex_mean`（**仅 `[M+H]+`**） | z 项，`GL_LAM=1.0` / 库命中 0.0 |
| MetFrag-parsimony（`frag_rescore.py`） | ≤2 键断裂碎片解释强度的加权占比 | `FRAG_LAM=0.5`，**只在 ICE 覆盖不到时**（`gap`） |
| PubChem 流行度先验 | `log1p(SIDs)+log1p(PMIDs)` | `z(ranker) + 0.25 * pop[pid]`，见下 |
| engine 2 | 名次 | `alpha=0.6` 的 RRF |
| analog propagation | Tanimoto × sim^3/^6 等 | 进 ranker 特征（engine1 `_analog_feats` 给 5 维） |

流行度先验（逐字，注意它**不是**加权求和进 z，而是只在池候选内重排）：

```python
def popularity_reorder(smis, keys, scs, forms, pids, pool_pop, mu):
    """bobthebot369 v10 prior: f = z(ranker) + mu * pop[pid] for pool candidates; they are re-sorted among the pool
    slots, generated candidates (pid < 0) keep their slots and get a '|gen' formula tag (own same-formula group)."""
    s = np.asarray(scs, np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids); isg = pid < 0
    f = z + mu * np.where(isg, 0.0, pool_pop[np.maximum(pid, 0)])
```

**"GBDT 还是线性？"** —— 候选级是 **GBDT**（共 4 个 `HistGradientBoostingClassifier`
× 每个 500 棵树，或主引擎的 LightGBM ranker）；**通道融合本身是线性 z 求和 + RRF**，
没有在融合层再上 GBDT。这与项目内「优先线性融合、GBDT 必须过 V-C」的纪律一致。

---

## 5. 0.413 靠什么（相对一个 0.13 级管线）

> ⚠️ 我没能定位到任何具体的 0.13 分 notebook，所以本节**不是"对某个 0.13 基线做的实测消融"**，
> 而是从这份 0.413 代码里逐条指认「哪些是额外通道」，以及 notebook 自己标注的增益。
> 逐条的实现证据是 VERIFIED；"这是 0.13 管线缺的东西"这层因果判断标记为 **LIKELY / 推理**。

**（a）实测谱库匹配 + 硬门控（最大头）。** pipeline 第一件事就是拿 `train.parquet`
的多来源实测库做熵相似度搜索（同一个 molecule 的所有谱都搜，还带
`search_shift_rows`——按 `query_prec - ref_prec` 平移参考谱，让同一化合物以不同 adduct
测到的谱也能匹配），`lib_max ≥ 0.9` 时**把整条链路的其余部分全部旁路**。
只在 COCONUT/PubChem 里做指纹排序、完全不利用实测库的管线，会直接丢掉这一整类分子。
`LIKELY`（代码 VERIFIED，因果为推理）。

**（b）双引擎候选池，独立于 PubChem。** 710,701 结构的池（训练结构 ∪ COCONUT，engine2 再并
ChEBI+LIPID MAPS）比"直接从 PubChem 取同质量/同分子式候选"精度高得多；
PubChem 105.9M 结构只作为**受门控的补充通道**（5 个固定槽），不是主力。`LIKELY`

**（c）同分子式组内的 z 融合，而不是全局线性加权。** 只在**同分子式且 ≥2 个成员**的组内
z-score 重排、且 **slot-preserving**。这既解决了"不同分子式的分数不可比"，
又保证**不会让一个原本排 30 的候选整体跃到第 3**——这是域外稳健性的关键。
一个全局加权分数（或对可见 test 调权重）的管线在这里最容易崩。`LIKELY`

**（d）两个前向模型而不是一个。** ICEBERG 2.1（`msg_all`，MassSpecGym）+ GLACIER
（单阶段，MassSpecGym）**独立**预测再各自 z 化。二者失败模式不同；
`gl_fuse.rerank_multi` 的多项 z 和是分开加权的，任一项缺失（`None`）自动记 0 而不是报错。
`LIKELY`（超参 `ICE_LAM=GL_LAM=1.0`、`POST_*` 可分开调，说明作者确实做了 separate weighting）。

**（e）[M+H]+ 限制（= MH），+0.006。** notebook 明确标注。`VERIFIED（标注值）`
机制上的理由（我的推断）：GLACIER 在 MassSpecGym 上 `[M+Na]+` 覆盖/精度差，
把噪声项喂进 z 融合会污染 ICEBERG 已经做对的组。`UNVERIFIED（机制）`

**（f）MetFrag-parsimony 只在 ICE 覆盖不到时生效（`FRAG_MODE='gap'`），+0.065（面板）。**
`frag_rescore.py` docstring 逐字：

```
Defaults follow the forum ablation on a 241-structure same-formula isomer panel (+0.065 MRR over the common
baseline): w2=0.6, linear intensity, tol 0.005 Da, H shifts -2..+3, at most 2 cleavages.
BASELINE_CFG is the common/pv.py-style setting.
```

配置用 `gap` 模式，`FRAG_LIB_OFF=False`，即**只在 ICEBERG 没覆盖的分子（如负模式、
`[M+NH4]+` 等）上**加这一项。这样既拿到面板增益，又不动主路径。`VERIFIED（文档）/
LIKELY（归因）`

**（g）PubChem 流行度先验，`POP_MU=0.25`（对照 bobthebot369 v10 LB 0.401 用的是 0.15）。**
只作用于**池候选**（`pid ≥ 0`），generated 候选（`pid < 0`）保持原位并被标成 `|gen`
公式标签（自成一组，不参与池内重排）。`VERIFIED（代码+注释）`

**（h）工程性：每阶段失败都静默降级。** `USE_ENG/USE_PC/USE_ICE/USE_GL/FRAG` 全部
`try/except`，任一阶段抛异常 → 该通道置空 → 保留上游顺序 → 仍交出一个合法
`submission.csv`（`fusion_core.validate` 做最终断言）。这在"提交必须成功"的竞赛里等于
**把最坏情况锁死在 rollback 基线**，而不是 0 分。`VERIFIED`

**（i）温度/切分纪律。** cell 8 的 `build_validation()` 自带**身份不相交留出集**构造：
从 `enveda-np-examples` 抽 250 个结构，把它们**所有** train 行删掉（`purge='all'`，
模拟 class-2「库完全不认识」）或只删该库的行（`purge='lib'`，模拟 class-1「同一化合物被
别的库测过」），并用 metric 口径（tautomer-canonical InChIKey14）算 MRR@25。
这正好对应本项目要求的 **V-C / V-A 两折**思路。`VERIFIED`

---

## 6. Runtime 与离线打包

### 6.1 附着的 Kaggle 数据集（14 个，来自 kernel metadata `datasetDataSources`）

| # | 数据集 | 大小 | 许可 |
|---|---|---|---|
| 1 | `prvsiyan/casmi26-fp-models-v2` | 288 MB | CC0 |
| 2 | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | 2.75 GB | Other |
| 3 | `prvsiyan/casmi26-ranker-features` | 9.5 MB | CC0 |
| 4 | `megayak/casmi26-simulated-ranker-rows` | 366 MB | CC0 |
| 5 | `ahmedberatozer/casmi26-fpnet-full1` | 199 MB | Other（含 CC BY-NC 成分） |
| 6 | `ahmedberatozer/casmi26-glacier` | 61 MB | Other（内含 MIT 的 ms-pred） |
| 7 | `ahmedberatozer/casmi26-iceberg` | 67 MB | Other（内含 MIT 的 ms-pred） |
| 8 | `ahmedberatozer/casmi26-pubchem-tier` | 7.21 GB | Other（NCBI PubChem 公开数据） |
| 9 | `ahmedberatozer/casmi26-v2-pool` | 1.68 GB | Other（含 CC BY-NC 成分） |
| 10 | `ahmedberatozer/casmi26-v3-models` | 412 MB | Other |
| 11 | `ahmedberatozer/casmi26-v4b-models` | 1.68 GB | Other |
| 12 | `prvsiyan/chebi-lipidmaps-casmi26` | 60 MB | **CC BY-NC-SA 4.0** |
| 13 | `prvsiyan/coconut-casmi26-candidates` | 423 MB | **CC BY 4.0** |
| 14 | `metric/rdkit-2026-3-3-wheel` | 149 MB | Unknown |

外加竞赛数据 `enveda-CASMI26-molecule-id-mass-spectra`。
总计约 **15.2 GB** 输入。**VERIFIED**（`datasetDataSources` 字段 + 各数据集 metadata `totalBytes`）

### 6.2 环境

- `enableGpu=true`, `machineShape=NvidiaTeslaT4`, **`enableInternet=false`**
- docker image `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
- RDKit：cell 6 从数据集离线 `pip install --no-index --no-deps rdkit-*.whl`
  （必须命中 `cp{major}{minor}` tag），**断言 `rdkit.__version__ == '2026.03.3'`**
  （metric 口径钉死 = tautomer-canonical InChIKey14）。
- ICE/GL runner 在**自己的 site 目录**装 **RDKit 2025.3.6**（`wheels/*.whl.ice`，
  `pip --no-index --no-deps --target`），runner 内部校验 `2025.03` 否则拒绝运行
  （理由逐字：`2026.03 kekulizes differently and changes predictions`）。
  → **同一次运行里同时存在两个 RDKit**，靠子进程 + `--target` 隔离。这是最有价值的可抄点之一。
- `casmi26-v4b-models` 用 `MANIFEST.json` 的 sha256 逐个校验，不一致直接 `assert` 失败；
  `derivation` family 有 `self_test()`，断言 RDKit 构建一致（`assert n_bad == 0`）。
- `NUMBA_CACHE_DIR` 指向 `/kaggle/working/numba_cache`（numba 编译缓存可写）。

### 6.3 时间预算与运行时长

| 阶段 | 预算 |
|---|---|
| ICEBERG | `ICE_BUDGET = 5400 s`（smoke run 时 `SMOKE_ICE_BUDGET = 300`） |
| GLACIER | `GL_BUDGET = 4000 s` |
| engine 2 / PubChem 通道子进程 | `STAGE_TIMEOUT_S = 4 * 3600`（各自） |
| ICE/GL 子进程硬杀 | `budget + grace_s`，`grace_s = 300` |

- ICEBERG 参考文档：T4、400 molecule、N=60 → **约 25–30 min**，`~25–32 predictions/s`
  （RTX 5090 laptop 实测 79/s）。**VERIFIED**
- GLACIER 论文：相对两阶段基线 **≈8× 推理加速**。**VERIFIED**
- **整份 notebook 的实际总 wall-clock：UNVERIFIED** —— pull API 返回的 JSON 里所有 cell 的
  `outputs` 都是空的（`cells with outputs: 0`），拿不到运行日志。
  按预算上限与上述速率，**LIKELY 约 2–3 小时**（T4）。
- 断点续跑设计：ICE/GL runner 在**开始前就写好 all-null 的 `out.json`**，
  之后每个 chunk 覆写一次，超预算就干净停下，进程只要写出过 `out.json` 就 `exit 0`。
  → **预算耗尽不会毁掉提交**，只是那些分子退回上一级排序。
- `DUMP_STAGES=True` → `/kaggle/working/stage_cache/{base,pc,eng,ice,gl,frag,meta}.json`，
  可用 `tools/refuse.py`（本地）在 CPU 上**离线重跑融合调参**，不必再动 GPU。
- 收尾：`shutil.rmtree('/kaggle/working/ice_site')` —— wheel site 有几千个文件，
  删掉以免版本输出列表过大。**VERIFIED**

### 6.4 提交格式

```python
v4 = [(mid, rows.get(mid, c['FALLBACK'])) for mid in sample_ids]   # FALLBACK = 'CCO'
# rows[mid] = ';'.join(final[:25])
```

`semicolon` 分隔、按 `sample_submission.csv` 的 `molecule_id` 顺序、最多 25 条；
`fusion_core.validate` 断言：行数/唯一性一致、`smiles` 非空、至少 1 条、至多 25 条。

---

## 7. WHAT I COULD NOT VERIFY

1. **总运行时长 / 各阶段实测耗时**。kernel pull JSON 的 `outputs` 全空，没有
   `[ 123s] ...` 日志、没有 `run_manifest.json`。只有代码里的预算上限和 ICE 的
   第三方速率估计。§6.3 的 2–3 h 是**推算**。
2. **0.413 分数与这份发布的 CFG 是否严格一一对应**。notebook 说"原提交 56792011，
   分数核对于 2026-10-03"，本页 `lastRunTime` 是 2026-10-01，且 Kaggle 明确提示
   "competition-score badge requires a scored submission from this new page"。
   `CFG.update(...)` 那一行我逐字读到了，但**无法证明它就是上榜时用的那组超参**。
3. **主引擎 v1（`casmi26-v4b-models` 的 `casmi` 包）的内部**：1.68 GB 未下载。因此
   `lib_max` 的精确计算、`FEATURES`（ranker 特征名与个数）、`fe_v4` families、
   `EngineCfg(generate=True)` 的生成候选逻辑、`fpnet.ModelBank` 的 bank 结构
   都是**从调用点推断**，不是读源码确认。
4. **`casmi26-fp-models-v2` / `casmi26-v3-models` / `DreamsFP D` / `casmi26-ranker-features`
   / `casmi26-simulated-ranker-rows` 的内部**：未下载，只知用途与字节数。
   engine 2 的 `BLOCKS`（特征块）我只看到调用方式，没读到 31+ 维的具体构成。
5. **"0.13 分管线"到底是什么**：没有定位到任何具体的 0.13 notebook 或该分数的公开记录，
   所以 §5 的对照是**结构性推理**（"这份多了哪些通道"），不是实测 ablation vs 0.13。
6. **`[M+H]+` 限制为何 +0.006 的机制**。代码注释给了数值，没给消融表。
7. **PubChem 通道 pass1 的 `PC_N1` 到底是 5000 还是 1000**。`probe_core2.py` 的
   docstring 写 `keep top PC_N1 = 1000`，但模块常量 `PC_N1 = 5000` 且 CFG 也设 5000。
   实际生效值 = 环境变量 `CASMI_PC_N1`（由 CFG 注入）= **5000**；docstring 是过时注释。
8. **`casmi26-pubchem-tier` 的 105,875,489 结构是否与 notebook 实际装载一致**——
   数据集描述与 popularity-prior 描述一致，但我没有下载 7.21 GB 去核对 `pc_mass.npy` 长度。
9. **许可的商用可行性**。多个数据集标注 "Other (specified in description)"，且自述含
   CC BY-NC 成分（Enveda 竞赛训练数据、ChEBI/LIPID MAPS 是 CC BY-NC-SA 4.0）。
   GLACIER/ICEBERG 代码本身是 MIT，但**权重与训练数据链条受 NC 限制**。
