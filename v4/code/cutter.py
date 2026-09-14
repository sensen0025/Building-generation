"""Stage B: cut oversized builds (>64 per axis) into one..several <=64^3 chunks.

Algorithm (lock onto building mass, not terrain):
1. building mask from classifier (building class).
2. Slide windows (size min(64,dim) per axis, stride 8 + edge), score =
   UNCOVERED building count via 3D summed-area table (vectorized).
3. Greedy: pick best window, crop to building bbox + margin (still <=64),
   mark covered; stop at coverage>=92%, or MAX_CUTS, or best<MIN_BUILDING.

Output per cut: origin (y,z,x), dims (w,h,l), stats.
"""
import numpy as np

MAX_CUTS = 8
MIN_BUILDING = 700
COVERAGE_TARGET = 0.92
STRIDE = 8
MARGIN = 2


def building_mask_from_names(names):
    import classifier
    cls = np.zeros(len(names), dtype=bool)
    for i, nm in enumerate(names):
        s = nm.split("[")[0]
        if s.endswith(":air") or s == "air":
            continue
        if classifier.is_building(nm):
            cls[i] = True
    return cls


def _axis_starts(dim, win, stride=STRIDE):
    if dim <= win:
        return np.array([0])
    n = (dim - win) // stride
    xs = list(range(0, n * stride + 1, stride))
    if xs[-1] != dim - win:
        xs.append(dim - win)
    return np.array(xs)


def _window_sums(mask_int, ys0, zs0, xs0, wh, wl, ww):
    """Vectorized window sums via SAT. Returns (len(ys0),len(zs0),len(xs0)) int array."""
    sat = np.zeros((mask_int.shape[0] + 1, mask_int.shape[1] + 1,
                    mask_int.shape[2] + 1), dtype=np.int32)
    sat[1:, 1:, 1:] = mask_int.cumsum(0).cumsum(1).cumsum(2)
    ys1, zs1, xs1 = ys0 + wh, zs0 + wl, xs0 + ww
    S111 = sat[np.ix_(ys1, zs1, xs1)]
    S011 = sat[np.ix_(ys0, zs1, xs1)]
    S101 = sat[np.ix_(ys1, zs0, xs1)]
    S110 = sat[np.ix_(ys1, zs1, xs0)]
    S001 = sat[np.ix_(ys0, zs0, xs1)]
    S010 = sat[np.ix_(ys0, zs1, xs0)]
    S100 = sat[np.ix_(ys1, zs0, xs0)]
    S000 = sat[np.ix_(ys0, zs0, xs0)]
    return S111 - S011 - S101 - S110 + S001 + S010 + S100 - S000


def _largest_component(sub, ys_, zs_, xs_):
    """Bbox of largest 26-connected component via run-based union-find (fast).

    Returns (ys, zs, xs) index arrays limited to the winning component's bbox
    extents (min/max per axis).
    """
    h, l, w = sub.shape
    padded = np.zeros((h, l, w + 2), dtype=bool)
    padded[:, :, 1:-1] = sub
    d = np.diff(padded.view(np.int8), axis=2)
    st = np.argwhere(d == 1)   # run starts (y,z,x) with sub-x == x
    en = np.argwhere(d == -1)  # run ends exclusive
    if len(st) == 0:
        return ys_, zs_, xs_
    # group runs by (y,z) row key
    key = st[:, 0] * l + st[:, 1]
    order = np.argsort(key, kind="stable")
    st, en = st[order], en[order]
    keys = key[order]
    row_runs = {}
    for i in range(len(st)):
        row_runs.setdefault(int(keys[i]), []).append((int(st[i, 2]), int(en[i, 2])))
    # union-find
    parent = {}

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    run_id = {}
    for rk, runs in row_runs.items():
        for j, (s, e) in enumerate(runs):
            run_id[(rk, j)] = (rk, j)
            parent[(rk, j)] = (rk, j)
    for rk, runs in row_runs.items():
        y, z = divmod(rk, l)
        for j, (s, e) in enumerate(runs):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dy == 0 and dz == 0:
                        continue
                    ny, nz = y + dy, z + dz
                    if not (0 <= ny < h and 0 <= nz < l):
                        continue
                    nk = ny * l + nz
                    nbr = row_runs.get(nk)
                    if not nbr:
                        continue
                    for k, (s2, e2) in enumerate(nbr):
                        if s2 <= e + 1 and s <= e2 + 1:  # overlap/adjacent
                            union((rk, j), (nk, k))
    # largest component by run length
    comp_len = {}
    comp_bbox = {}
    for node in parent:
        r = find(node)
        rk, j = node
        s, e = row_runs[rk][j]
        comp_len[r] = comp_len.get(r, 0) + (e - s)
        y, z = divmod(rk, l)
        bb = comp_bbox.get(r)
        if bb is None:
            comp_bbox[r] = [y, y, z, z, s, e - 1]
        else:
            bb[0] = min(bb[0], y); bb[1] = max(bb[1], y)
            bb[2] = min(bb[2], z); bb[3] = max(bb[3], z)
            bb[4] = min(bb[4], s); bb[5] = max(bb[5], e - 1)
    if not comp_len:
        return ys_, zs_, xs_
    best_root = max(comp_len, key=comp_len.get)
    if comp_len[best_root] < 0.5 * (len(ys_)):  # no dominant component; keep all
        return ys_, zs_, xs_
    y0, y1, z0, z1, x0, x1 = comp_bbox[best_root]
    keep = (ys_ >= y0) & (ys_ <= y1) & (zs_ >= z0) & (zs_ <= z1) & \
           (xs_ >= x0) & (xs_ <= x1)
    return ys_[keep], zs_[keep], xs_[keep]


def find_cuts(vol, names, max_cuts=MAX_CUTS, min_building=MIN_BUILDING,
              coverage_target=COVERAGE_TARGET):
    h, l, w = vol.shape
    bmask_full = building_mask_from_names(names)[vol]
    # crop to building bbox + 8 margin to shrink SAT cost
    ys_, zs_, xs_ = np.nonzero(bmask_full)
    if len(ys_) == 0:
        return []
    by0 = max(0, int(ys_.min()) - 8); by1 = min(h, int(ys_.max()) + 1 + 8)
    bz0 = max(0, int(zs_.min()) - 8); bz1 = min(l, int(zs_.max()) + 1 + 8)
    bx0 = max(0, int(xs_.min()) - 8); bx1 = min(w, int(xs_.max()) + 1 + 8)
    vol = vol[by0:by1, bz0:bz1, bx0:bx1]
    bmask = bmask_full[by0:by1, bz0:bz1, bx0:bx1]
    off = (by0, bz0, bx0)
    h, l, w = vol.shape
    nb_total = int(bmask.sum())
    wh, wl, ww = min(h, 64), min(l, 64), min(w, 64)
    ys0 = _axis_starts(h, wh)
    zs0 = _axis_starts(l, wl)
    xs0 = _axis_starts(w, ww)
    covered = np.zeros_like(bmask)
    cuts = []
    for _ in range(max_cuts):
        remaining = bmask & ~covered
        counts = _window_sums(remaining.astype(np.int32), ys0, zs0, xs0,
                              wh, wl, ww)
        flat = int(counts.argmax())
        iy, iz, ix = np.unravel_index(flat, counts.shape)
        c = int(counts[iy, iz, ix])
        if c < min_building:
            break
        y0, z0, x0 = int(ys0[iy]), int(zs0[iz]), int(xs0[ix])
        sub = remaining[y0:y0 + wh, z0:z0 + wl, x0:x0 + ww]
        ys_, zs_, xs_ = np.nonzero(sub)
        if len(ys_) > 0:
            ys_, zs_, xs_ = _largest_component(sub, ys_, zs_, xs_)
        ry0 = max(0, int(ys_.min()) - MARGIN); ry1 = min(wh, int(ys_.max()) + 1 + MARGIN)
        rz0 = max(0, int(zs_.min()) - MARGIN); rz1 = min(wl, int(zs_.max()) + 1 + MARGIN)
        rx0 = max(0, int(xs_.min()) - MARGIN); rx1 = min(ww, int(xs_.max()) + 1 + MARGIN)
        cutvol = vol[y0 + ry0:y0 + ry1, z0 + rz0:z0 + rz1, x0 + rx0:x0 + rx1]
        import classifier
        m = classifier.classify_volume(names, cutvol)
        cuts.append({
            "origin": [int(y0 + ry0) + off[0], int(z0 + rz0) + off[1],
                       int(x0 + rx0) + off[2]],  # y,z,x in FILE coords
            "dims": [int(rx1 - rx0), int(ry1 - ry0), int(rz1 - rz0)],  # w,h,l
            "building": m["building"],
            "nonair": m["nonair"],
            "redstone": m["redstone"],
            "natural": m["natural"],
            "redstone_ratio": round(m["redstone_ratio"], 4),
            "natural_ratio": round(m["natural_ratio"], 4),
            "mapart_like": round(m["mapart_like"], 4),
        })
        covered[y0:y0 + wh, z0:z0 + wl, x0:x0 + ww] |= True
        if int((bmask & ~covered).sum()) == 0:
            break
        if int((bmask & covered).sum()) / nb_total >= coverage_target:
            break
    return cuts
