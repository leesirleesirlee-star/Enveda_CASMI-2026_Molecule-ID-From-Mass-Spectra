# ---- v17 patch: popularity prior (log1p SID + log1p PMID from PubChem, mapped onto this store by stereo-stripped SMILES) ----
_POP_DIR = os.environ.get('CASMI_POP_DIR'); _POP_LAM = float(os.environ.get('CASMI_POP_LAM', '0')); _POP_UNION = int(os.environ.get('CASMI_POP_UNION', '0'))
_init_worker0 = init_worker


def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    _init_worker0(pc_dir, bits_path, pool_meta_path, code_dir)
    if _POP_DIR:
        _W['ls'] = np.load(os.path.join(_POP_DIR, 'pc_lsid.npy'), mmap_mode='r')
        _W['lp'] = np.load(os.path.join(_POP_DIR, 'pc_lpmid.npy'), mmap_mode='r')


def probe_one(task):
    mid, target, z = task
    t0 = time.time()
    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']
    tol = float(target) * PPM * 1e-6
    a0 = int(np.searchsorted(_W['mass'], float(target) - tol, 'left'))
    smis = _window(float(target))
    n = len(smis)
    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0, fz_top=None)
    if n == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)
    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    diag['n_pass1'] = n1
    if n1 == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]
    use_pop = ('ls' in _W) and _POP_LAM > 0
    pop = (np.asarray(_W['ls'][a0:a0 + n], np.float32) + np.asarray(_W['lp'][a0:a0 + n], np.float32)) if use_pop else np.zeros(n, np.float32)
    if use_pop and _POP_UNION > 0:
        pp = np.where(ok, pop, -np.inf); ku = min(_POP_UNION, int(ok.sum()))
        top1 = np.union1d(top1, np.argpartition(-pp, ku - 1)[:ku])
    bits = _W['bits']
    fz, idx = [], []
    for i in top1:
        fp = chem.raw_fingerprint(smis[i])
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
    diag['n_pass2'] = len(idx)
    if not idx:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    fz = np.asarray(fz, np.float64); idx = np.asarray(idx)
    keyc = {}
    def key(j):
        if j not in keyc:
            keyc[j] = chem.score_key(smis[idx[j]]); diag['n_keyed'] += 1
        return keyc[j]
    for j in np.argsort(-fz, kind='stable'):        # gate quantity = v16's pc_fz[0] (best non-pool structure by raw f.z)
        k = key(j)
        if k is not None and k not in _W['pool_keys']:
            diag['fz_top'] = float(fz[j]); break
    zf = (fz - fz.mean()) / (fz.std() + 1e-9)
    sc = zf + _POP_LAM * pop[idx]
    order = np.argsort(-sc, kind='stable')
    out, out_s, out_k, out_p, out_sc, seen = [], [], [], [], [], set()
    for j in order:
        k = key(j)
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); out_p.append(float(pop[idx[j]])); out_sc.append(float(sc[j]))
        if len(out) >= K:
            break
    diag['n_listed'] = len(out)
    diag['S'] = float(out_sc[0]) if out_sc else 0.0
    diag['top_pop'] = float(out_p[0]) if out_p else 0.0
    diag['secs'] = time.time() - t0
    return mid, out, out_s, out_k, diag
