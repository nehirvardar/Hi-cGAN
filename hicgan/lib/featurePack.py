import csv
import logging
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

log = logging.getLogger(__name__)

_NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# Memory-level cache so repeated chromosome lookups within a process do not touch disk
_BIN_CACHE_MEM: Dict[str, Dict[str, np.ndarray]] = {}
_PAIR_CACHE_MEM: Dict[Tuple[str, str], csr_matrix] = {}


def is_feature_pack_source(pPath: Optional[str]) -> bool:
    """Return True if pPath points to a feature_paketi directory or its .xlsx/.csv files."""
    if not pPath or not isinstance(pPath, str):
        return False
    if os.path.isdir(pPath):
        for fname in ("bin_features.xlsx", "feature_tiers.csv"):
            if os.path.exists(os.path.join(pPath, fname)):
                return True
        if os.path.exists(os.path.join(pPath, "_cache", "bin_features_cache.npz")):
            return True
        return False
    if os.path.isfile(pPath):
        base = os.path.basename(pPath).lower()
        return base in ("bin_features.xlsx", "pair_features.xlsx", "feature_tiers.csv")
    return False


def resolve_feature_pack_dir(pPath: str) -> str:
    """Return the directory containing the feature pack files."""
    if os.path.isfile(pPath):
        return os.path.dirname(os.path.abspath(pPath))
    return os.path.abspath(pPath)


def _col_letters_to_idx(cell_ref: str) -> int:
    idx = 0
    for ch in cell_ref:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return idx - 1


def _list_xlsx_sheets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    """Return list of (sheet_name, zip_member_path) in workbook order."""
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {}
    for rel in rels_root:
        r_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if r_id and target:
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            rel_map[r_id] = target

    sheets = []
    for sh in wb_root.findall(f".//{_NS_MAIN}sheet"):
        name = sh.attrib.get("name", "")
        r_id = sh.attrib.get(f"{_NS_REL}id", "")
        target = rel_map.get(r_id, "")
        if name and target:
            sheets.append((name, target))
    return sheets


def _parse_bin_sheet_xml(zf: zipfile.ZipFile, member_path: str) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Parse one worksheet of bin_features.xlsx using streaming ET.iterparse.
    Returns (chroms, feat_headers, matrix_bin_indices, feat_matrix_float64).
    """
    tag_row = _NS_MAIN + "row"
    tag_c = _NS_MAIN + "c"
    tag_v = _NS_MAIN + "v"
    tag_is = _NS_MAIN + "is"
    tag_t = _NS_MAIN + "t"

    headers: List[str] = []
    chroms: List[str] = []
    bin_indices: List[int] = []
    rows_data: List[List[float]] = []
    n_cols = 0

    with zf.open(member_path) as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != tag_row:
                continue
            r_idx = int(elem.attrib.get("r", "0"))
            if r_idx == 1:
                col_dict = {}
                for c in elem.findall(tag_c):
                    c_idx = _col_letters_to_idx(c.attrib.get("r", "A1"))
                    val = None
                    v_el = c.find(tag_v)
                    if v_el is not None and v_el.text is not None:
                        val = v_el.text
                    else:
                        is_el = c.find(tag_is)
                        if is_el is not None:
                            t_el = is_el.find(tag_t)
                            if t_el is not None and t_el.text is not None:
                                val = t_el.text
                    col_dict[c_idx] = val or ""
                n_cols = max(col_dict.keys()) + 1 if col_dict else 0
                headers = [col_dict.get(i, "") for i in range(n_cols)]
            else:
                n_feat = max(0, n_cols - 5)
                feat_row = [np.nan] * n_feat
                chrom_val = ""
                m_bin_idx = r_idx - 2
                for c in elem.findall(tag_c):
                    c_idx = _col_letters_to_idx(c.attrib.get("r", "A1"))
                    t_attr = c.attrib.get("t", "n")
                    val_str = None
                    if t_attr == "inlineStr":
                        is_el = c.find(tag_is)
                        if is_el is not None:
                            t_el = is_el.find(tag_t)
                            if t_el is not None:
                                val_str = t_el.text
                    else:
                        v_el = c.find(tag_v)
                        if v_el is not None:
                            val_str = v_el.text
                    if val_str is None or val_str == "":
                        continue
                    if c_idx == 0:
                        chrom_val = str(val_str).strip()
                        if chrom_val.endswith(".0"):
                            chrom_val = chrom_val[:-2]
                        if chrom_val.lower().startswith("chr"):
                            chrom_val = chrom_val[3:]
                    elif c_idx == 4:
                        try:
                            m_bin_idx = int(float(val_str))
                        except ValueError:
                            pass
                    elif c_idx >= 5 and (c_idx - 5) < n_feat:
                        try:
                            feat_row[c_idx - 5] = float(val_str)
                        except ValueError:
                            pass
                chroms.append(chrom_val)
                bin_indices.append(m_bin_idx)
                rows_data.append(feat_row)
            elem.clear()

    feat_headers = headers[5:] if len(headers) > 5 else []
    feat_arr = np.asarray(rows_data, dtype=np.float64)
    return chroms, feat_headers, np.asarray(bin_indices, dtype=np.int32), feat_arr


def _normalize_feature_column(col: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D feature column (with possible NaNs) to [0.0, 1.0].
    Applies log1p first for non-negative skewed count/signal/length columns.
    """
    valid_mask = np.isfinite(col)
    if not np.any(valid_mask):
        return np.zeros_like(col, dtype=np.float32)

    valid_vals = col[valid_mask]
    c_min = float(np.min(valid_vals))
    c_max = float(np.max(valid_vals))

    out = np.where(valid_mask, col, 0.0).astype(np.float64)
    # Apply log1p for non-negative columns whose range exceeds [0, 2]
    if c_min >= 0.0 and c_max > 2.0:
        out = np.log1p(np.maximum(out, 0.0))

    o_min = float(np.min(out))
    o_max = float(np.max(out))
    if o_max > o_min + 1e-12:
        out = (out - o_min) / (o_max - o_min)
    else:
        out = np.zeros_like(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_or_build_bin_features_cache(pPackDir: str) -> Dict[str, np.ndarray]:
    """
    Load cached normalized 1D bin features from <pPackDir>/_cache/bin_features_cache.npz,
    or build the cache from bin_features.xlsx and feature_tiers.csv if not yet cached.
    """
    pack_dir = resolve_feature_pack_dir(pPackDir)
    if pack_dir in _BIN_CACHE_MEM:
        return _BIN_CACHE_MEM[pack_dir]

    cache_dir = os.path.join(pack_dir, "_cache")
    cache_path = os.path.join(cache_dir, "bin_features_cache.npz")
    xlsx_path = os.path.join(pack_dir, "bin_features.xlsx")
    tiers_path = os.path.join(pack_dir, "feature_tiers.csv")

    if os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        xlsx_mtime = os.path.getmtime(xlsx_path) if os.path.exists(xlsx_path) else 0.0
        tiers_mtime = os.path.getmtime(tiers_path) if os.path.exists(tiers_path) else 0.0
        if cache_mtime >= max(xlsx_mtime, tiers_mtime):
            with np.load(cache_path, allow_pickle=False) as data:
                result = {k: data[k] for k in data.files}
            _BIN_CACHE_MEM[pack_dir] = result
            return result

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"bin_features.xlsx not found in feature pack directory: {pack_dir}")

    log.info("Building 1D bin_features cache from %s ...", xlsx_path)
    os.makedirs(cache_dir, exist_ok=True)

    # Read metadata from feature_tiers.csv if available
    tiers_meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    if os.path.exists(tiers_path):
        with open(tiers_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loc = (row.get("location") or row.get("sheet") or "").strip()
                sh = loc.split("/")[-1].strip() if "/" in loc else loc
                col = (row.get("feature") or row.get("column_name") or row.get("column") or "").strip()
                row_type = (row.get("type") or row.get("category") or "").strip().lower()
                if col and row_type != "id":
                    tiers_meta[(sh, col)] = row
                    tiers_meta[("", col)] = row

    all_feat_names: List[str] = []
    all_feat_sheets: List[str] = []
    all_feat_tiers: List[str] = []
    all_feat_rep: List[str] = []
    all_feat_cell: List[str] = []
    raw_feat_blocks: List[np.ndarray] = []

    ref_chroms: Optional[List[str]] = None
    ref_bin_indices: Optional[np.ndarray] = None

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        sheets = _list_xlsx_sheets(zf)
        for sh_name, member_path in sheets:
            chroms, feat_headers, bin_indices, feat_arr = _parse_bin_sheet_xml(zf, member_path)
            if ref_chroms is None:
                ref_chroms = chroms
                ref_bin_indices = bin_indices
            raw_feat_blocks.append(feat_arr)
            for col_name in feat_headers:
                meta = tiers_meta.get((sh_name, col_name)) or tiers_meta.get(("", col_name), {})
                all_feat_names.append(col_name)
                all_feat_sheets.append(sh_name)
                all_feat_tiers.append(str(meta.get("tier", "0")).strip())
                all_feat_rep.append(str(meta.get("representative", "yes")).strip().lower())
                all_feat_cell.append(str(meta.get("cell_line", "none")).strip())

    raw_all = np.concatenate(raw_feat_blocks, axis=1)  # (N_bins, 80)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}

    # Build the 3 missingness/presence binary indicators recommended by READ_ME.md
    n_bins = raw_all.shape[0]
    ind_pc_gene = np.ones((n_bins,), dtype=np.float32)
    if "mean_gene_length" in name_to_idx:
        ind_pc_gene = np.isfinite(raw_all[:, name_to_idx["mean_gene_length"]]).astype(np.float32)
    elif "pc_gene_count" in name_to_idx:
        ind_pc_gene = (np.nan_to_num(raw_all[:, name_to_idx["pc_gene_count"]], nan=0.0) > 0).astype(np.float32)

    ind_mirna = np.ones((n_bins,), dtype=np.float32)
    if "mirna_mean_log2expr" in name_to_idx:
        ind_mirna = np.isfinite(raw_all[:, name_to_idx["mirna_mean_log2expr"]]).astype(np.float32)

    ind_hela = np.ones((n_bins,), dtype=np.float32)
    if "hela_rnapii_chip_signal" in name_to_idx:
        ind_hela = np.isfinite(raw_all[:, name_to_idx["hela_rnapii_chip_signal"]]).astype(np.float32)

    # Normalize each of the 80 feature columns globally to [0, 1]
    norm_cols = [_normalize_feature_column(raw_all[:, j]) for j in range(raw_all.shape[1])]
    norm_cols.extend([ind_pc_gene, ind_mirna, ind_hela])
    features_norm = np.stack(norm_cols, axis=1).astype(np.float32)

    all_feat_names.extend(["ind_has_pc_gene", "ind_has_mirna_expr", "ind_has_hela_data"])
    all_feat_sheets.extend(["indicator", "indicator", "indicator"])
    all_feat_tiers.extend(["0", "0", "0"])
    all_feat_rep.extend(["yes", "yes", "yes"])
    all_feat_cell.extend(["none", "GM12878", "HeLa"])

    result = {
        "chroms": np.asarray(ref_chroms, dtype="<U8"),
        "matrix_bin_index": ref_bin_indices if ref_bin_indices is not None else np.arange(n_bins, dtype=np.int32),
        "feature_names": np.asarray(all_feat_names, dtype="<U64"),
        "feature_sheets": np.asarray(all_feat_sheets, dtype="<U32"),
        "feature_tiers": np.asarray(all_feat_tiers, dtype="<U8"),
        "feature_rep": np.asarray(all_feat_rep, dtype="<U8"),
        "feature_cell_line": np.asarray(all_feat_cell, dtype="<U32"),
        "features_norm": features_norm,
    }
    np.savez_compressed(cache_path, **result)
    _BIN_CACHE_MEM[pack_dir] = result
    log.info("Saved 1D bin_features cache (%d bins, %d features) to %s", n_bins, features_norm.shape[1], cache_path)
    return result


def select_feature_columns(
    pCacheData: Dict[str, np.ndarray],
    pFeatureSet: Union[str, Sequence[str]] = "representative",
    pIncludeIndicators: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Resolve a featureSet preset or custom list of sheet/column names into column indices and names.

    Supported presets:
      - "representative" (default): representative == "yes" across all sheets (24 cols + 3 indicators = 27)
      - "direct": all columns from the 'direct' sheet (13 cols + 2 non-HeLa indicators = 15)
      - "tier1_2": representative == "yes" columns in tiers 0, 1, 2 (16 cols + 3 indicators = 19)
      - "gm12878": representative == "yes" columns excluding HeLa (21 cols + 2 indicators = 23)
      - "hela": representative == "yes" columns excluding GM12878 (13 cols + 2 indicators = 15)
      - "all": all 80 feature columns (+ 3 indicators = 83)
      - Custom comma-separated string or sequence of sheet names or feature column names.
    """
    names = [str(x) for x in pCacheData["feature_names"]]
    sheets = [str(x) for x in pCacheData["feature_sheets"]]
    tiers = [str(x) for x in pCacheData["feature_tiers"]]
    reps = [str(x) for x in pCacheData["feature_rep"]]
    cells = [str(x) for x in pCacheData["feature_cell_line"]]

    if isinstance(pFeatureSet, str):
        preset = pFeatureSet.strip().lower()
    else:
        preset = ""

    selected_idx: List[int] = []

    if preset in ("representative", "rep", "default", ""):
        for i, (sh, rep) in enumerate(zip(sheets, reps)):
            if sh == "indicator":
                if pIncludeIndicators:
                    selected_idx.append(i)
            elif rep == "yes":
                selected_idx.append(i)
    elif preset == "direct":
        for i, (sh, cell) in enumerate(zip(sheets, cells)):
            if sh == "direct":
                selected_idx.append(i)
            elif sh == "indicator" and pIncludeIndicators and "hela" not in cell.lower():
                selected_idx.append(i)
    elif preset in ("tier1_2", "tier12", "tier0_1_2"):
        for i, (sh, rep, tr) in enumerate(zip(sheets, reps, tiers)):
            if sh == "indicator":
                if pIncludeIndicators:
                    selected_idx.append(i)
            elif rep == "yes" and tr.startswith(("0", "1", "2")):
                selected_idx.append(i)
    elif preset == "gm12878":
        for i, (sh, rep, cell) in enumerate(zip(sheets, reps, cells)):
            if "hela" in cell.lower():
                continue
            if sh == "indicator":
                if pIncludeIndicators:
                    selected_idx.append(i)
            elif rep == "yes":
                selected_idx.append(i)
    elif preset == "hela":
        for i, (sh, rep, cell) in enumerate(zip(sheets, reps, cells)):
            if "gm12878" in cell.lower():
                continue
            if sh == "indicator":
                if pIncludeIndicators:
                    selected_idx.append(i)
            elif rep == "yes":
                selected_idx.append(i)
    elif preset == "all":
        for i, sh in enumerate(sheets):
            if sh == "indicator" and not pIncludeIndicators:
                continue
            selected_idx.append(i)
    else:
        # Custom comma-separated tokens (either sheet names or feature column names)
        if isinstance(pFeatureSet, str):
            tokens = [t.strip() for t in pFeatureSet.split(",") if t.strip()]
        else:
            tokens = [str(t).strip() for t in pFeatureSet if str(t).strip()]
        token_set_lower = {t.lower() for t in tokens}
        for i, (nm, sh) in enumerate(zip(names, sheets)):
            if nm.lower() in token_set_lower or sh.lower() in token_set_lower:
                selected_idx.append(i)
        if pIncludeIndicators:
            for i, sh in enumerate(sheets):
                if sh == "indicator" and i not in selected_idx:
                    selected_idx.append(i)

    if not selected_idx:
        raise ValueError(
            f"No features matched featureSet={pFeatureSet!r}. "
            f"Valid presets: 'representative', 'direct', 'tier1_2', 'gm12878', 'hela', 'all', "
            f"or sheet/column names."
        )

    idx_arr = np.asarray(selected_idx, dtype=np.int32)
    sel_names = [names[i] for i in selected_idx]
    return idx_arr, sel_names


def get_chrom_bin_features(
    pPackDir: str,
    pChrom: str,
    pFeatureSet: Union[str, Sequence[str]] = "representative",
    pNumBins: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (features_array, feature_names) for chromosome pChrom.
    features_array has shape (n_bins, n_selected_features) and dtype float32.
    """
    cache_data = load_or_build_bin_features_cache(pPackDir)
    sel_idx, sel_names = select_feature_columns(cache_data, pFeatureSet=pFeatureSet)

    chrom_norm = str(pChrom).strip()
    if chrom_norm.lower().startswith("chr"):
        chrom_norm = chrom_norm[3:]

    chroms = cache_data["chroms"]
    mask = chroms == chrom_norm
    if not np.any(mask):
        raise ValueError(f"Chromosome {pChrom!r} (normalized: {chrom_norm!r}) not found in bin_features.xlsx")

    bin_idx = cache_data["matrix_bin_index"][mask]
    feat_sub = cache_data["features_norm"][mask][:, sel_idx]

    # Sort by matrix_bin_index to guarantee exact 0..N-1 bin alignment
    order = np.argsort(bin_idx, kind="stable")
    feat_sub = feat_sub[order].astype(np.float32)

    if pNumBins is not None and feat_sub.shape[0] != pNumBins:
        if feat_sub.shape[0] > pNumBins:
            feat_sub = feat_sub[:pNumBins]
        else:
            pad = np.zeros((pNumBins - feat_sub.shape[0], feat_sub.shape[1]), dtype=np.float32)
            feat_sub = np.concatenate([feat_sub, pad], axis=0)

    return feat_sub, sel_names


_PAIR_ROW_RE = re.compile(
    r'<row r="\d+"><c [^>]*>(?:<is><t>([^<]+)</t></is>|<v>([^<]+)</v>)</c>'
    r'<c [^>]*><v>(\d+)</v></c><c [^>]*><v>(\d+)</v></c><c [^>]*><v>([^<]+)</v></c></row>'
)


def _build_all_pair_feature_caches(pPackDir: str) -> None:
    """
    Parse pair_features.xlsx once and save per-chromosome symmetric CSR .npz matrices
    in <pPackDir>/_cache/pair_kmer7_chr{chrom}.npz.
    """
    pack_dir = resolve_feature_pack_dir(pPackDir)
    pair_xlsx = os.path.join(pack_dir, "pair_features.xlsx")
    if not os.path.exists(pair_xlsx):
        raise FileNotFoundError(f"pair_features.xlsx not found in feature pack directory: {pack_dir}")

    cache_dir = os.path.join(pack_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Determine per-chromosome bin counts from bin_features cache
    bin_cache = load_or_build_bin_features_cache(pack_dir)
    chroms_arr = bin_cache["chroms"]
    unique_chroms, counts = np.unique(chroms_arr, return_counts=True)
    chrom_n_bins = {str(ch): int(cnt) for ch, cnt in zip(unique_chroms, counts)}

    log.info("Building 2D pair_features (kmer7_revcomp_cosine) CSR cache from %s ...", pair_xlsx)
    rows_by_chrom: Dict[str, List[int]] = {ch: [] for ch in chrom_n_bins}
    cols_by_chrom: Dict[str, List[int]] = {ch: [] for ch in chrom_n_bins}
    vals_by_chrom: Dict[str, List[float]] = {ch: [] for ch in chrom_n_bins}

    with zipfile.ZipFile(pair_xlsx, "r") as zf:
        sheets = _list_xlsx_sheets(zf)
        for _, member_path in sheets:
            raw_xml = zf.read(member_path).decode("utf-8")
            for m in _PAIR_ROW_RE.finditer(raw_xml):
                ch = (m.group(1) or m.group(2)).strip()
                if ch.endswith(".0"):
                    ch = ch[:-2]
                if ch.lower().startswith("chr"):
                    ch = ch[3:]
                bi = int(m.group(3))
                bj = int(m.group(4))
                val = float(m.group(5))
                if ch not in rows_by_chrom:
                    rows_by_chrom[ch] = []
                    cols_by_chrom[ch] = []
                    vals_by_chrom[ch] = []
                rows_by_chrom[ch].append(bi)
                cols_by_chrom[ch].append(bj)
                vals_by_chrom[ch].append(val)

    for ch, r_list in rows_by_chrom.items():
        c_list = cols_by_chrom[ch]
        v_list = vals_by_chrom[ch]
        n_b = chrom_n_bins.get(ch, (max(max(r_list, default=0), max(c_list, default=0)) + 1))
        if r_list:
            r_arr = np.asarray(r_list, dtype=np.int32)
            c_arr = np.asarray(c_list, dtype=np.int32)
            v_arr = np.asarray(v_list, dtype=np.float32)
            # Symmetrize upper-triangle entries (bi != bj)
            off_diag = r_arr != c_arr
            r_sym = np.concatenate([r_arr, c_arr[off_diag]])
            c_sym = np.concatenate([c_arr, r_arr[off_diag]])
            v_sym = np.concatenate([v_arr, v_arr[off_diag]])
            mat = csr_matrix((v_sym, (r_sym, c_sym)), shape=(n_b, n_b), dtype=np.float32)
        else:
            mat = csr_matrix((n_b, n_b), dtype=np.float32)
        out_path = os.path.join(cache_dir, f"pair_kmer7_chr{ch}.npz")
        save_npz(out_path, mat)
        _PAIR_CACHE_MEM[(pack_dir, ch)] = mat

    log.info("Saved 2D pair_features CSR caches for %d chromosomes to %s", len(rows_by_chrom), cache_dir)


def get_chrom_pair_matrix(
    pPackDir: str,
    pChrom: str,
    pNumBins: Optional[int] = None,
) -> csr_matrix:
    """
    Return the symmetric (n_bins, n_bins) CSR sparse matrix of kmer7_revcomp_cosine
    for chromosome pChrom, building the .npz cache on first call if needed.
    """
    pack_dir = resolve_feature_pack_dir(pPackDir)
    chrom_norm = str(pChrom).strip()
    if chrom_norm.lower().startswith("chr"):
        chrom_norm = chrom_norm[3:]

    cache_key = (pack_dir, chrom_norm)
    if cache_key not in _PAIR_CACHE_MEM:
        cache_path = os.path.join(pack_dir, "_cache", f"pair_kmer7_chr{chrom_norm}.npz")
        pair_xlsx = os.path.join(pack_dir, "pair_features.xlsx")
        need_build = not os.path.exists(cache_path)
        if not need_build and os.path.exists(pair_xlsx):
            if os.path.getmtime(cache_path) < os.path.getmtime(pair_xlsx):
                need_build = True
        if need_build:
            _build_all_pair_feature_caches(pack_dir)
        if cache_key not in _PAIR_CACHE_MEM:
            if not os.path.exists(cache_path):
                raise ValueError(f"Pair feature matrix for chromosome {pChrom!r} not found in {pack_dir}")
            _PAIR_CACHE_MEM[cache_key] = load_npz(cache_path).tocsr().astype(np.float32)

    mat = _PAIR_CACHE_MEM[cache_key]
    if pNumBins is not None and mat.shape[0] != pNumBins:
        if mat.shape[0] > pNumBins:
            mat = mat[:pNumBins, :pNumBins].tocsr()
        else:
            mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=(pNumBins, pNumBins), dtype=np.float32)
    return mat
