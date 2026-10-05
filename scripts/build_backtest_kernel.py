"""
Build the leak-free backtest kernel (Method A, component 3).

Derived from notebooks/v44_base/notebook.ipynb with four changes:

  cell 5   replace the competition test set with a synthetic, held-out query set
           built in-notebook from train.parquet (so no dataset upload is needed)
  cell 13  install a class-level monkeypatch on Engine.run that injects the
           exclusion hooks. Patching the *class* (not the instance) means it works
           regardless of how V1FE.run calls the engine, and it is installed before
           V1FE is constructed, so even a captured bound method is covered.
  cell 15  per query: hide that structure's spectra from the library and analog
           channels and drop its pool entry
  cell 31  replace the (irrelevant, and here fatal) champion lock with the MRR@25
           scorer, reported overall and per library regime

The three hook index spaces were read off the source, not guessed:
  exclude          bool mask over LIBRARY SPECTRA (len == L.sid == 2,539,608)
  exclude_sid      structure id (L.sid value space)
  drop_pid         index into the candidate POOL

Usage: python scripts/build_backtest_kernel.py [--limit 400] [--fold 0] [--slug ...]
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks", "v44_base", "notebook.ipynb")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
DATASETS = [
    "prvsiyan/casmi26-fp-models-v2", "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "prvsiyan/casmi26-ranker-features", "megayak/casmi26-simulated-ranker-rows",
    "ahmedberatozer/casmi26-fpnet-full1", "ahmedberatozer/casmi26-glacier",
    "ahmedberatozer/casmi26-iceberg", "ahmedberatozer/casmi26-pubchem-tier",
    "ahmedberatozer/casmi26-v2-pool", "ahmedberatozer/casmi26-v3-models",
    "ahmedberatozer/casmi26-v4b-models", "prvsiyan/chebi-lipidmaps-casmi26",
    "prvsiyan/coconut-casmi26-candidates", "metric/rdkit-2026-3-3-wheel",
]

# ---- cell 5: build the held-out query set in-notebook -----------------------
CELL5 = '''import pandas as pd, hashlib
# ---- V45 BACKTEST: replace the competition test set with a held-out fold ----
BT_FOLD, BT_REGIME, BT_LIMIT = {fold}, {regime!r}, {limit}
NP_LIBS = {{'enveda-np-examples'}}; SYN_LIBS = {{'enveda-180', 'pluskal_ms2', 'drug_plus'}}
def _regime(lib):
    return 'np' if lib in NP_LIBS else ('syn' if lib in SYN_LIBS else 'other')
COL = ['inchikey14', 'normalized_smiles', 'ingest_lib', 'ms2_mzs', 'ms2_normalized_intensities',
       'base_peak_intensity', 'adduct', 'ionization_mode', 'instrument_type', 'precursor_mz',
       'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units']
import pyarrow.parquet as _pq
_tr = pd.read_parquet(os.path.join(COMP, 'train.parquet'), columns=COL)
_tr = _tr[_tr.inchikey14.notna() & (_tr.inchikey14 != '')]
_rank = {{'np': 0, 'syn': 1, 'other': 2}}
_t = _tr[['inchikey14', 'ingest_lib']].copy(); _t['regime'] = [_regime(x) for x in _t.ingest_lib]
st = _t.groupby('inchikey14').regime.agg(lambda s: min(s, key=lambda x: _rank[x])).reset_index()
_rng = np.random.default_rng(20261005); st['fold'] = -1
for _reg, _g in st.groupby('regime'):
    _idx = _g.index.to_numpy(); st.loc[_idx, 'fold'] = _rng.permutation(len(_idx)) % 5
_sel = st[(st.fold == BT_FOLD) & (st.regime == BT_REGIME)]
_keys = sorted(_sel.inchikey14.tolist())
if BT_LIMIT and len(_keys) > BT_LIMIT:
    _keys = sorted(np.random.default_rng(20261005).choice(_keys, BT_LIMIT, replace=False).tolist())
_keyset = set(_keys)
q = _tr[_tr.inchikey14.isin(_keyset)].copy()
q = q[q.ms2_mzs.map(len) > 0]
assert q.inchikey14.nunique() == len(_keyset), 'a held-out structure lost all its spectra'
q = q.rename(columns={{'inchikey14': 'molecule_id'}})
q['spectrum_id'] = q.molecule_id + '_' + q.groupby('molecule_id').cumcount().astype(str)
_ref = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=None).head(0)
BT_COLS = list(_ref.columns)
q = q[BT_COLS]
SM = '/kaggle/working/bt_comp'; os.makedirs(SM, exist_ok=True)
q.to_parquet(os.path.join(SM, 'test.parquet'), index=False)
pd.DataFrame({{'molecule_id': sorted(_keyset), 'smiles': ['CCO'] * len(_keyset)}}).to_csv(
    os.path.join(SM, 'sample_submission.csv'), index=False)
for _f in ['train.parquet']:
    _d = os.path.join(SM, _f)
    if not os.path.exists(_d):
        os.symlink(os.path.join(COMP, _f), _d)
_sm = _tr.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles.to_dict()
BT_TRUTH = {{k: _sm.get(k) for k in _keyset}}     # molecule_id (inchikey14) -> SMILES
assert all(BT_TRUTH.values()), 'a held-out structure has no SMILES'
COMP = SM
IS_RERUN, ICE_BUDGET = True, 300
print(f'V45 BACKTEST fold={{BT_FOLD}} regime={{BT_REGIME}} molecules={{len(_keyset)}} '
      f'spectra={{len(q)}} COMP={{COMP}}', flush=True)'''

# ---- cell 13: install the class-level patch, before V1FE is constructed ------
PATCH = '''
# ---- V45 BACKTEST: inject the engine's own leak-hiding hooks -----------------
# exclude      bool mask over LIBRARY SPECTRA (len == len(E.L.sid))
# exclude_sid  structure id (E.L.sid value space)
# drop_pid     index into the candidate POOL
_V45EX = dict(exclude=None, sid=-1, pid=-1)
_bt_mask = np.zeros(len(E.L.sid), bool)
_bt_sids = E.L.sid
_bt_k2pid = dict(zip(E.pool.key, np.arange(len(E.pool.key))))
_EngineCls = type(E)
_bt_orig_run = _EngineCls.run


def _bt_run(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
    return _bt_orig_run(self, spectra, target,
                        exclude=_V45EX['exclude'] if exclude is None else exclude,
                        exclude_sid=_V45EX['sid'] if exclude_sid < 0 else exclude_sid,
                        exclude_lib=exclude_lib,
                        drop_pid=_V45EX['pid'] if drop_pid < 0 else drop_pid)


_EngineCls.run = _bt_run
print('V45 backtest: Engine.run patched; library spectra', len(_bt_mask), flush=True)'''

# ---- cell 15: per-query exclusion ------------------------------------------
HOOK_ANCHOR = "        target = float(np.median(nms))"
HOOK_BODY = """        target = float(np.median(nms))
        # V45 BACKTEST: hide this query from library, analog channel and pool
        _ik = BT_TRUTH.get(str(mid))
        _V45EX['exclude'], _V45EX['sid'], _V45EX['pid'] = None, -1, -1
        if _ik is not None:
            _V45EX['exclude'] = _bt_mask_view(_bt_mask, _bt_sids, _bt_k2pid, _ik)
            _V45EX['sid'] = _bt_k2sid.get(_ik, -1)
            _V45EX['pid'] = _bt_k2pid.get(_ik, -1)"""

HELPER = '''
# ---- V45 BACKTEST helper: build one query's exclusion mask -------------------
_bt_k2sid = {}
for _k, _s in zip(E.L.struct_key, np.arange(len(E.L.struct_key))):
    _bt_k2sid.setdefault(_k, _s)


def _bt_mask_view(mask, sids, k2pid, key):
    mask[:] = False
    sid = _bt_k2sid.get(key, -1)
    if sid >= 0:
        mask[sids == sid] = True
    else:
        print('V45 backtest WARNING: no library sid for', key, flush=True)
    return mask
'''

# ---- cell 31: replace the champion lock with the MRR scorer -----------------
# Cell 31 must not keep the lock: its keys are the real test's molecule_ids, which
# cannot exist here, so it would raise KeyError and skip the rest of the notebook.
# NOTE ON PLACEMENT: cell 32 rewrites submission.csv (clean_and_validate), so the
# number printed here is the *pre-clean* one. A second call is appended after cell 32
# and is the number to quote; this one exists so a cell-32 failure still leaves data.
SCORER = '''# ---- V45 BACKTEST: MRR@25 of the backtest fold -------------------------------
assert 'BT_TRUTH' in globals(), 'backtest query set was not built'


def bt_score(tag):
    _sub = pd.read_csv('submission.csv')
    _sub['molecule_id'] = _sub.molecule_id.astype(str)
    _ranks = []
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _t = BT_TRUTH.get(_m)
        if _t is None:
            continue
        _tk = chem.score_key(_t) or _t
        _r = 0
        for _i, _c in enumerate([x for x in str(_s).split(';') if x][:25], start=1):
            if (chem.score_key(_c) or _c) == _tk:
                _r = _i
                break
        _ranks.append(_r)
    _ranks = np.asarray(_ranks, dtype=float)
    _hit = _ranks > 0
    print()
    print(f'V45 BACKTEST RESULT [{tag}]')
    print(f'  molecules        {len(_ranks)}')
    print(f'  MRR@25           {np.mean(np.where(_hit, 1.0/np.maximum(_ranks,1), 0.0)):.4f}')
    print(f'  top-1            {np.mean(_ranks == 1):.4f}')
    print(f'  hit@25           {_hit.mean():.4f}')
    print(f'  rank histogram   1:{(_ranks==1).sum()} 2:{(_ranks==2).sum()} '
          f'3-5:{((_ranks>=3)&(_ranks<=5)).sum()} 6-10:{((_ranks>=6)&(_ranks<=10)).sum()} '
          f'11-25:{((_ranks>=11)&(_ranks<=25)).sum()} miss:{(_ranks==0).sum()}')
    print(f'V45 BACKTEST DONE [{tag}]', flush=True)


bt_score('pre-clean')'''

# appended after cell 32, which rewrites submission.csv
SCORER_TAIL = "bt_score('final')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--regime", default="other")
    ap.add_argument("--limit", type=int, default=400,
                    help="query molecules. Keep this at 400 (= the real test's size) whenever "
                         "ICE/GL are ON: their budgets are fixed wall-clock caps, so a larger "
                         "query set silently starves them and measures a different pipeline "
                         "(at 2,000 queries ICE would cover ~1/5 as many molecules). Tier 1 "
                         "policy screening is the exception: it runs with --degrade no_ice at "
                         "--limit 2000, where the budgets are off by construction.")
    ap.add_argument("--slug", default="casmi26-v45-backtest")
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "backtest"))
    ap.add_argument("--degrade", default="none", choices=["none", "top1", "no_ice"],
                    help="deliberately cripple the pipeline to calibrate the backtest "
                         "(component 4). 'top1' keeps a single candidate, 'no_ice' zeroes "
                         "the forward-model weights. Both MUST score below the baseline; "
                         "if they do not, the backtest cannot rank configurations and "
                         "method A is worthless.")
    a = ap.parse_args()

    nb = json.loads(open(SRC, encoding="utf-8").read())
    cells = nb["cells"]

    def setsrc(i, text):
        cells[i]["source"] = text.splitlines(keepends=True)

    setsrc(5, CELL5.format(fold=a.fold, regime=a.regime, limit=a.limit))

    s13 = "".join(cells[13]["source"])
    anchor = "E = Engine(L, P, tfp, EngineCfg(generate=True), bank)"
    assert s13.count(anchor) == 1, "engine construction line not found exactly once"
    setsrc(13, s13.replace(anchor, anchor + "\n" + PATCH + HELPER))

    s15 = "".join(cells[15]["source"])
    n = s15.count(HOOK_ANCHOR)
    assert n >= 1, f"cell 15 anchor found {n} times"
    # the first occurrence is inside the molecule loop
    setsrc(15, s15.replace(HOOK_ANCHOR, HOOK_BODY, 1))

    assert "_champion_top1" in "".join(cells[31]["source"]), "cell 31 is not the lock cell"
    setsrc(31, SCORER)
    # cell 32 rewrites submission.csv, so score the final artefact too
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": SCORER_TAIL.splitlines(keepends=True)})

    # --- calibration cripples (component 4) ---------------------------------
    if a.degrade == "top1":
        s15 = "".join(cells[15]["source"])
        old = "                    if len(smis) >= TOPN: break"
        assert s15.count(old) == 1, "candidate cap line not found exactly once"
        setsrc(15, s15.replace(old, "                    if len(smis) >= 1: break  # V45 CALIB"))
        print("CALIB: candidate list capped at 1")
    elif a.degrade == "no_ice":
        s17 = "".join(cells[17]["source"])
        old = "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 300, True"
        s3 = "".join(cells[3]["source"])
        assert s3.count(old) == 1, "ICE constant line not found exactly once"
        setsrc(3, s3.replace(old, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 0.0, 0, False  # V45 CALIB"))
        print("CALIB: ICE/GL weights zeroed (forward models off)")

    out = os.path.join(a.out, "bt" if a.degrade == "none" else f"bt_{a.degrade}")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "notebook.ipynb")
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    meta = {"id": f"nicholasnicklee/{a.slug}", "title": "CASMI26 V45 Backtest",
            "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": False,
        # Pinned to the image the V44 reference itself ran on. Kaggle's default image is
        # now Python 3.13, but the ICEBERG/GLACIER runners install a bundled cp312 RDKit
        # wheel and abort with 'not a supported wheel on this platform'; the notebook's
        # try/except swallows that and silently drops the whole forward-model channel.
        # See docs/EXPERIMENTS.md, 2026-10-05 (fourth segment).
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "docker_image_pinning_type": "original",
            "dataset_sources": DATASETS, "competition_sources": [COMP],
            "kernel_sources": [], "model_sources": []}
    mp = os.path.join(out, "kernel-metadata.json")
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {p} ({os.path.getsize(p):,} bytes) and {mp}")
    print(f"backtest config: fold={a.fold} regime={a.regime} limit={a.limit}")


if __name__ == "__main__":
    main()
