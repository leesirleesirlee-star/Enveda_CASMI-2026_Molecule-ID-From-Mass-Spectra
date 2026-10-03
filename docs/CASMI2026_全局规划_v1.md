# CASMI 2026 全局规划（v1 评审稿）

**日期**：2026-10-03
**状态**：待你确认（尚未开工）
**依据**：`CASMI竞赛详细说明.md`、`CASMI_2026_PRD_v2.1.docx.docx` + 本次对竞赛 API / 数据 / 本机环境的实测核查

---

## 0. 一句话结论

**方向对，但 PRD 的"验证方法"和"主引擎选型"这两块地基是坏的，必须先修地基再盖楼。**
好消息有三条：竞赛真实且仍在进行（还剩 **72.7 天**）；公开可复现的最高分工件已经被人写明白，照着抄能到 **0.33–0.34**；你的机器本地 GPU 完全够用（已实测），真正的约束是 Kaggle 那边的 9 小时/算力配额，而不是你的笔记本。

---

## 1. 已核实的事实（与 PRD 的差异用 ⚠️ 标出）

| 项目 | 核实结果 | 来源 |
|---|---|---|
| 竞赛性质 | **Code Competition（只能提交 Notebook）**，静态 CSV 提交必然 0 分 | Kaggle API `isKernelsSubmissionsOnly: true` |
| 每日提交次数 | **5 次**（不是常见的 10 次） | Kaggle API `maxDailySubmissions: 5` |
| 截止时间 | 最终 **2026-12-14 23:59 UTC**；组队截止 **2026-12-07** | Kaggle API |
| 当前规模 | **2,242 支队伍**（9/17 才 498 支，增长很快） | Kaggle API `teamCount` |
| 评测指标 | MRR@25，InChIKey 前 14 位比对，RDKit **固定 2026.03.3** 互变异构规范化 → **立体化学与互变异构被忽略** | 官方 metric notebook |
| 数据训练集 | **2,539,608 行 × 18 列**，2.82 GB，21 个 row group | 我直接读远端 parquet 元数据 |
| 数据测试集 | 本地可见 `test.parquet` = **1,213 谱 / 400 分子**，12 列 | 我直接读取 |
| 数据来源 | `ingest_lib` 共 11 个库：enveda-180（约 115 万谱）、pluskal_ms2、riken、gnps、massbank、mona、spectraverse、msdial、drug_plus、masaryk、**enveda-np-examples** | 我读 row group 21 组统计 |
| ⚠️ **可见测试集是占位样本** | 评分时 Kaggle 会换成**隐藏测试集**（分子 ID 都不同）。**在可见 test 上调参毫无意义** | 多份选手笔记 + 0.000 反面案例 |
| ⚠️ Kaggle 环境无 RDKit | **镜像里没有 rdkit/matchms/faiss**，且评分时**断网**。所有依赖必须提前打包成 Kaggle Dataset | Kaggle docker-python 仓库 |
| ⚠️ P100 已下线 | 2026-09-15 退役，现在只有 **T4×2** 一类，GPU 配额约 30 小时/周 | Kaggle 公告 |
| 公开分位 | 公开榜 SOTA **0.409–0.425**（9/25–10/2）；主流可复现区间 **0.33–0.34**；纯检索 0.143 | 多份选手仓库 |

### 榜单分位（供你定位目标）

```
0.44 ── 榜首区间（9/29 曾出现 0.440）
0.409 ─ 当前公开 SOTA（seyitkaangunes，四通道融合 + popularity prior）
0.336 ─ beraterolelk "Analog Ranker" notebook
0.328 ─ 公开四通道 analog 基线的复现分
0.30  ─ PRD 的"最终目标"   ← 已经被公开方案大幅超过
0.25  ─ PRD 的"最低可接受"
0.176 ─ PRD 记录的"已有检索基线"（这条我查不到任何来源 ⚠️）
0.143 ─ 有人实测的纯检索基线
0.000 ─ 天真方案（只索引 Enveda 两个库 / 直接传 CSV）
```

**结论：PRD 把目标定在 0.30 是"低于公开可复现水平"。合理目标应改为：先稳到 0.32–0.34，再冲 0.36+。**

---

## 2. PRD 必须修正的三处硬伤（这是本次核查最重要的产出）

### 2.1 验证策略是坏的 —— 这是 0.873 验证分 vs 0.14 Kaggle 分的真正原因

PRD 第 7 章要求：**Murcko 骨架完全不相交 + 评估时从检索库中动态移除正确答案**。

这个设计在数学上就把"检索"这条路堵死了：答案不在库里，检索永远不可能召回正确分子，只能靠生成。而公开事实是——**测试集里相当一部分分子，其参考谱就在 train 里**（最强表述："many test molecules share identical or near-identical experimental spectra in train.parquet"）。0.145 → 0.335 这一跳正是靠加合物质量位移的库匹配实现的。

于是你的流水线陷入一个自相矛盾的循环：
- 验证集（无正确答案在库）→ 得到 0.873/0.14 这种诡异落差
- 真实榜（正确答案大多在库）→ 检索才是主力，但你的验证集惩罚检索

**修正后的验证协议（三段式）**：

| 折 | 构造方式 | 用途 | 对应真实榜 |
|---|---|---|---|
| **V-A 仪器校准折** | `ingest_lib == 'enveda-np-examples'`（**已确认：第 21 个 row group，1,184 谱 / 250 个天然产物**，与测试集仪器完全一致） | 主校准指标，直接对标公开榜 | Class-1/2 主力 |
| **V-B 同源折** | 全库随机留出，**答案保留在库内** | 度量纯检索上限 | 泄漏部分 |
| **V-C 严格折** | 身份/骨架不相交，**答案从库中移除** | 度量真实外推能力 | Class-3 尾部 |

并输出 **召回天花板曲线（oracle recall@K）**，直接判断瓶颈在"召回"还是在"排序"。**每次改进必须同时在 V-A 和 V-C 上报告**，只看 V-A 会重复"过拟合泄漏"的错误。

### 2.2 主引擎选型押错了 —— DreaMS+SimMS 不是获胜路径

PRD 第 3 章把 DreaMS（1024 维嵌入）+ SimMS（GPU 相似度）当作核心引擎。实测核查后：

| PRD 假设 | 实际情况 |
|---|---|
| DreaMS checkpoint ~1.2GB | 实际**需要两个 checkpoint，约 2.6GB**（embedding 1.24GB + ssl 1.39GB） |
| 本地 GPU 批量推理 250 万谱 | 无人公布过吞吐量；依赖极重（pyopenms 等约 20 个包）；**在 Kaggle 免费额度内跑不动** |
| SimMS 可 `pip install simms` | **PyPI 上的 `simms` 是完全无关的射电干涉包**。真正的 SimMS 只能 `pip install git+...`，**评分断网时无法安装** |
| 用 FAISS 建索引 | Kaggle 镜像**没有 faiss** |

**而公开 0.33–0.34 的方案没有一个用 DreaMS。** 它们用的是：

> **质量窗口召回 → 四通道证据 → 线性融合 → 输出 Top-25**

四通道 =
1. **直接谱匹配**（modified cosine / spectral entropy），
2. **Analog 传播**（关键创新：**不是改结构**，而是把库命中的**峰按 Δm 平移**，再用指纹相似度把注释传播到候选分子 —— `AnalogScore(c)=max_a[Sim_mod(q,a)²·Tanimoto(fp_c,fp_a)]`），
3. **虚拟碎片**（in-silico fragmentation，RDKit 自实现即可），
4. **神经指纹**（FPNet）—— 这一条是可选加分项。

**并且必须"线性融合"，不要在最上面套 GBDT 重排器**：公开数据里 LightGBM LambdaMART 在域内 0.7517、域外直接崩到 0.2301。这正是 PRD 3.4 节"XGBoost/LightGBM 重排"的陷阱。

### 2.3 "9 小时"与本地算力：方向没错，但约束点判断反了

- 9 小时限制：**只有选手笔记佐证，官方 Rules 页读不到**（Kaggle 页面是 JS 渲染的，我给三个来源都试过）。按 9 小时规划，但**留足余量，按 6 小时设计**。
- 本地 RTX 5060：**我已实测可用**（见下），PRD 说"8GB 显存受限不适合大模型"是对的，但结论应是"**不做大模型，做检索+轻量打分**"，而不是"降级生成模型"。

---

## 3. 本机环境实测结果（已跑通，可直接进入下一步）

| 项目 | 实测值 | 评价 |
|---|---|---|
| GPU | RTX 5060 Laptop，sm_120，26 SM，7.93GB 显存 | ✅ 可用 |
| GPU 算力 | fp32+TF32 约 **14–15 TFLOPS** | ⚠️ 偏低，因功耗被锁 |
| 功耗墙 | 当前 60W（默认 50W，**上限 85W**），满载 SM 2145MHz / 最大 3090MHz | ⚠️ **接通电源前提下可解锁到 85W，约 +40%** |
| 驱动 / CUDA | 616.92 / CUDA 13.4；PyTorch 2.15.0+cu130 | ✅ 匹配 |
| CPU / 内存 / 磁盘 | 16 逻辑核 / ~32GB / C: 238GB 空闲、D: **529GB 空闲** | ✅ 充足 |
| 现有 conda 环境 | `Digital_Resin`（py3.10.21）：torch ✅、**rdkit 2026.03.6 ✅**（比评分器 2026.03.3 新）、numpy/pandas/sklearn ✅；matchms/xgboost/lightgbm/faiss ❌ | 半可用，需补包 |
| 网络 | GitHub / HuggingFace / PyPI **全通**；Kaggle 官网通、API 需鉴权 | ✅ 数据可获得 |

### 两个环境层面的重要发现

1. **沙箱会屏蔽 GPU**：在当前受限模式下 `torch.cuda.is_available()` 返回 `False`，解除后正常。**所有 GPU 相关命令需要用提权方式运行。**（已确认非硬件故障）
2. **不需要 Kaggle 账号也能拿到数据**：HuggingFace 上有竞赛数据的完整镜像，我已校验：
   - `pradeepss007/Casmi26`：`train.parquet` **2.82GB**、`test.parquet` 4.6MB、`sample_submission.csv`
   - `lonelyforever/enveda.casmi`：同样三个文件（备用镜像）
   - 另有 `cuonguyenphu/...` 镜像含他人方案中间件（`fp_model_v2.pt`、`library.pkl`、`ranker_*.parquet`）
   - 我已实际下载 `test.parquet` + `sample_submission.csv` + 一份完整公开方案代码（70KB）到 `.deepworks/tmp/`

   **建议：仍然申请 Kaggle 账号**（提交必须有），但开发阶段不必被账号卡住。

---

## 4. 建议的技术架构（替换 PRD 第 3 章）

```
                    ┌─────────────── 离线（本地，一次性） ───────────────┐
 train.parquet ──▶  过滤"类测试"谱（timsTOF + enveda-np-examples 等）
 (2.82GB/254万行)   谱清洗 → 紧凑谱库 library.pkl
                    构建候选结构目录（train 结构 + COCONUT 2.0 + ChEBI/LIPID MAPS）
                    预计算：指纹矩阵 / 质量索引 / 位移索引
                    └──────────────────────┬────────────────────────────┘
                                           ▼  打包上传为 Kaggle Dataset（断网可用）
                    ┌─────────────── 在线（Kaggle Notebook，<6h） ───────┐
 test.parquet ──▶   分子级聚合（同分子多谱、多碰撞能量）
                    ① 质量窗口召回（±ppm + ¹³C 位移，>99.9% 前体召回）
                    ② 四通道证据：
                       直接匹配 │ analog 位移传播 │ 虚拟碎片 │ 神经指纹
                    ③ 线性融合 + 按 query 做 z-score 标准化
                    ④ 25 槽位：铺"骨架多样性"，不要浪费在立体异构体
                    ──▶ /kaggle/working/submission.csv
                    └───────────────────────────────────────────────────┘
```

**关键设计决策（与 PRD 相反的地方）**：
- ❌ 不用 DreaMS / SimMS 作为主引擎（保留为后期可选实验，不进主线）
- ❌ 不在顶端套 GBDT 重排（除非在严格折 V-C 上验证不崩）
- ✅ 用 matchms 的 `ModifiedCosineGreedy` / `NeutralLossesCosine`（Apache-2.0，PyPI 可得）
- ✅ RDKit 自实现虚拟碎片（PRD 对 MetFrag 的修正是**正确且重要**的，保留这条）
- ✅ 25 槽位做**骨架多样化** —— 因为指标忽略立体/互变异构，堆立体异构体是纯浪费

---

## 5. 分阶段计划（10.4 周，对齐 12/14 截止）

| 阶段 | 周次 | 交付物 | 验收标准 |
|---|---|---|---|
| **P0 地基修正** | 第 1 周 | 三段式验证集（V-A/V-B/V-C）+ oracle 召回曲线；环境补齐（matchms/xgboost/lightgbm/pyarrow）；数据落盘 | **先测出"公开方案在你机器上的真实基线"**，拿到可信基线数字 |
| **P1 检索骨干** | 第 2–3 周 | 质量窗口召回 + 直接谱匹配 + 分子级聚合；端到端跑通并产出 submission.csv | V-A 上 MRR@25 ≥ 0.25；提交格式 100% 合法（先拿到第一个非零榜分） |
| **P2 Analog 通道** | 第 3–5 周 | Δm 峰平移 + 指纹门控传播 | V-A ≥ **0.32–0.34**（对齐公开可复现水平）；**Kaggle 榜分 ≥ 0.30** |
| **P3 碎片 + 指纹通道** | 第 6–8 周 | 虚拟碎片通道；FPNet 指纹通道（可选）；线性融合调参 | V-A ≥ 0.36；Kaggle > 0.33 |
| **P4 冲榜与固化** | 第 9–10.4 周 | 端到端 Notebook < 6h；技术报告；外部数据合规清单 | 稳定提交，Kaggle 冲 0.36+ |

**每周固定动作**：V-A/V-C 双指标报告 + 一次 Kaggle 提交（5 次/天的额度足够）。

---

## 6. 风险登记表（新增/修正）

| 风险 | 影响 | 缓解 |
|---|---|---|
| **可见 test 是占位集，误用于调参** | 严重误导，白干数周 | 只信 V-A/V-C，test 只做格式校验 |
| **断网 + 无 RDKit 导致评分 rerun 静默失败** | 提交 0 分 | P1 结束前完成离线打包演练（wheel + 模型全进 Dataset） |
| GPU 功耗锁 60W | 本地离线预计算慢约 40% | 接电源 + 解锁到 85W（需你确认） |
| 域偏移导致重排器崩溃（0.75→0.23） | 榜分暴跌 | 坚持以 V-C 为准；优先线性融合 |
| Kaggle GPU 配额 30h/周 | 大模型方案不可行 | 主线不依赖 GPU，GPU 只用于本地预计算 |
| COCONUT/ChEBI 等外部结构库许可证 | 获奖需开源+合规 | 现在就登记每份外部资源的许可证（PRD 已有该清单，保留） |
| `enveda-np-examples` 只有 250 个分子 | 主校准折方差大 | 与 V-B/V-C 联用；注意榜分噪声约 ±0.006 |

---

## 7. 需要你现在决策的 3 件事

1. **验证协议**：是否接受"三段式验证 + 废弃 PRD 的纯骨架不相交"？（这是最关键的一条）
2. **主引擎**：是否接受"检索+四通道线性融合"作为主线、把 DreaMS/SimMS 降级为 P3 之后的可选实验？
3. **数据获取**：走 HuggingFace 公开镜像先开工（我可立即下载 2.82GB），还是你先提供 Kaggle API Token 走官方渠道？（提交环节反正必须有账号）

另外两件小事，一并确认：
4. **Ponytail**：已查到就是 [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)（MIT，约省 54% 代码量 / 20% token）。但注意——**它的定位是"写代码时别过度设计"，而本项目的瓶颈是 ML 实验设计，不是代码量**。我建议装它（无害且省 token），但**不要指望它解决本项目的主要风险**。你本机还装了 OpenCode（`.opencode/` + `opencode.jsonc`），ponytail 有官方 OpenCode 适配；同时它也能被 DSH 当 skill 用。要我装吗？装到哪个？
5. **GPU 功耗解锁**：要我把 RTX 5060 从 60W 提到 85W 吗？（可回退，需要你接电源）

---

## 附录：本次核查中我实际执行的动作

- 修复了 `D:\CASMI竞赛` 的 Windows 文件权限（原权限缺失导致所有 shell 命令失败），备份与回滚脚本在 `D:\CASMI竞赛-acl-recovery\`
- 用 HTTP Range 只读远端 parquet 的文件尾，拿到 2,539,608 行 × 18 列的完整 schema，**未下载 2.82GB 全量文件**
- 用 row group 列统计确认 `enveda-np-examples` 位于第 21 个 row group（1,184 行）→ 精确锁定主校准折
- 实测 GPU 可用性与算力（沙箱内被屏蔽 vs 解除后正常）
- 下载并核查了公开方案代码（`.deepworks/tmp/solution.py`，3,318 行的 LightGBM LambdaRank 全流程）
