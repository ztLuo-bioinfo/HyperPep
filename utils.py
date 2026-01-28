"""Feature extraction utilities for HyperPep.

Two interchangeable hypernode instantiation modes:
- PG (physicochemical group + k-mer window): fast, RDKit-free.
- FG (SMARTS functional groups): chemistry-aligned, requires RDKit.

Public entrypoint:
- sequences_geodata_1(): converts one (sequence, label) pair into a PyG Data object
  with fields expected by the training/evaluation pipeline.

"""

# utils.py — HyperPep unified feature extraction (FG / PG)
# Supports mode="fg" (SMARTS functional-group hypergraph) or mode="pg" (physicochemical-group + k-mer motif hypergraph).
# Downstream code expects the same fields:
#   x_h           : [H, d_h] hypernode features
#   h2_edge_index : [2, M]  incidence (hypernode_id, residue_id)
#   h2_edge_attr  : [E, d_e] hyperedge (residue) attributes
#   num_hypernodes, num_hyper2edges
#
# Note: RDKit is required only for mode="fg". mode="pg" does NOT require RDKit.

import math
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional

# -----------------------
# Minimal Data container with batching rules for hypergraph
# -----------------------
class HyperData(Data):
    """Custom PyG Data that batches incidence indices correctly."""
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'h2_edge_index':
            return 1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        # When batching multiple samples, PyG needs to know how to offset indices.
        # For incidence indices [2, M] = (hypernode_id, residue_id), we must offset
        # hypernode_id by num_hypernodes and residue_id by num_hyper2edges.

        if key == 'h2_edge_index':
            Hn = int(getattr(self, 'num_hypernodes', 0))
            He2 = int(getattr(self, 'num_hyper2edges', 0))
            return torch.tensor([[Hn], [He2]], dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

# ======================
# Common: 20 AA mapping
# ======================
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i for i, aa in enumerate(AA20)}

def clean_aa20_sequence(seq: str) -> str:
    """Return a sequence containing only 20 canonical amino acids (uppercase)."""
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq.strip().replace(" ", "").upper()
    # Remove non-canonical residues (X, B, Z, etc.) to keep fixed alphabet.
    return "".join([ch for ch in seq if ch in AA20])

# ==========================================================
# PG mode: physicochemical-group hypernodes
# ==========================================================

AA_GROUPS = {
    "hydrophobic": set("AVLIMFWY"),
    "polar":       set("STNQCH"),
    "positive":    set("KR"),
    "negative":    set("DE"),
    "special":     set("GP"),
}
GROUP_NAMES = list(AA_GROUPS.keys())
GROUP2IDX = {name: i for i, name in enumerate(GROUP_NAMES)}

AA2GROUP = {}
for gname, aas in AA_GROUPS.items():
    gid = GROUP2IDX[gname]
    for aa in aas:
        AA2GROUP[aa] = gid

def build_property_trigram_hyper(sequence: str, k: int = 3):
    """
    Hypernodes:
      1) physicochemical group nodes (5)
      2) k-mer window nodes (default k=3, stride=1)
    Hyperedges:
      residues (positions) 0..L-1
    Features:
      h2_edge_attr: residue one-hot [L, 20]
      x_h: mean of connected residue one-hot [H, 20]
    """
    residues = list(clean_aa20_sequence(sequence))
    L = len(residues)

    H_prop = len(GROUP_NAMES)
    # Number of sliding windows (stride=1). If L < k, no trigram nodes.
    H_tri = max(L - k + 1, 0)
    H_total = H_prop + H_tri

    # Residue features
    h2_edge_attr = torch.zeros((L, len(AA20)), dtype=torch.float32)
    for ridx, aa in enumerate(residues):
        h2_edge_attr[ridx, AA2IDX[aa]] = 1.0

    # Hypernode features (sum then mean)
    x_h = torch.zeros((H_total, len(AA20)), dtype=torch.float32)
    counts = torch.zeros((H_total,), dtype=torch.float32)

    rows, cols = [], []

    # group nodes
    for ridx, aa in enumerate(residues):
        gid = AA2GROUP[aa]
        rows.append(gid); cols.append(ridx)
        x_h[gid] += h2_edge_attr[ridx]
        counts[gid] += 1.0

    # k-mer window nodes
    for t in range(H_tri):
        hidx = H_prop + t
        for ridx in range(t, min(L, t + k)):
            rows.append(hidx); cols.append(ridx)
            x_h[hidx] += h2_edge_attr[ridx]
            counts[hidx] += 1.0

    h2_edge_index = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2, 0), dtype=torch.long)

    nonzero = counts > 0
    if nonzero.any():
        x_h[nonzero] = x_h[nonzero] / counts[nonzero].unsqueeze(-1)

    return x_h, h2_edge_index, h2_edge_attr, int(H_total), int(L)

# ==========================================================
# FG mode: RDKit + HELM + SMARTS functional-group hypernodes
# ==========================================================
def _ensure_rdkit():
    try:
        from rdkit import Chem  # type: ignore
        return Chem
    except Exception as e:
        raise ImportError("mode='fg' requires RDKit. Please install rdkit, or use mode='pg'.") from e

def peptide_to_helm(peptide: str, polymer_id: str = 'PEPTIDE1') -> str:
    # Convert a peptide string into HELM notation understood by RDKit.
    # This allows parsing modified residues like (ac).

    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = ''.join([c + '.' if c.isupper() else c for c in sequence]).rstrip('.')
    return f"{polymer_id}{{{sequence_helm}}}$$$$"

def get_aminoacids_2(peptide: str):
    sequence = peptide.replace("(ac)", "[ac]*").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = ''.join([c + '.' if c.isupper() else c for c in sequence]).rstrip('.')
    tokens = sequence_helm.split('.')
    tokens = [elem.replace('*', '.') for elem in tokens]
    return tokens

def get_sequence_fg(peptide: str) -> str:
    return peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")

SMARTS_PATTERNS = {
    "phenyl":              "c1ccccc1",
    "phenol":              "c1ccc(cc1)O",
    "indole":              "c1cc2cc[nH]c2cc1",
    "imidazole":           "n1cc[nH]c1",
    "guanidinium":         "NC(=N)N",
    "primary_amine":       "[$([NX3H2]),$([NX3H][CX4])]",
    "ammonium":            "[NX4+]",
    "carboxylic_acid":     "[CX3](=O)[OX2H]",
    "carboxylate":         "[CX3](=O)[O-]",
    "amide_sidechain":     "[CX3](=O)[NX3H2]",
    "thiol":               "[SX2H]",
    "thiolate":            "[S-]",
    "thioether":           "[SX2]([#6])[#6]",
    "disulfide":           "[SX2]-[SX2]",
    "aliphatic_alcohol":   "[OX2H][CX4;!$(C=O)]",
    "phosphate_monoester": "[OX2][PX4](=O)(O)O",
    "pyrrolidine":         "[NX3H0]1CCCC1"
}

def _one_hot_len(n: int, i: int):
    v = [0] * n
    if 0 <= i < n:
        v[i] = 1
    return v

def _dedup_sets(list_of_atom_sets):
    seen, uniq = set(), []
    for s in list_of_atom_sets:
        key = frozenset(s)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(list(s))
    return uniq

def build_smarts_hyper(mol, pad_dim: int = None, allow_overlap: bool = True):
    # allow_overlap=True keeps overlapping SMARTS matches; set False to uniquify.

    """
    Build functional-group hypernodes from SMARTS patterns.
    Returns:
        h_edge_index: [2, I] pairs (atom_id, hypernode_id) -- internal use
        h_edge_attr : [H, pad_dim] hypernode features (one-hot(pattern) + [size, aromatic_ratio, hetero_ratio])
    """
    Chem = _ensure_rdkit()
    smarts_mols = {}
    for k, v in SMARTS_PATTERNS.items():
        m = Chem.MolFromSmarts(v)
        if m is not None:
            smarts_mols[k] = m
    pattern_names = list(smarts_mols.keys())
    P = len(pattern_names)

    if pad_dim is None:
        pad_dim = P + 3  # keep 20 dims if P==17

    hyper_sets, hyper_attrs = [], []

    for pid, name in enumerate(pattern_names):
        patt = smarts_mols[name]
        matches = mol.GetSubstructMatches(patt, uniquify=not allow_overlap)
        if not matches:
            continue
        atom_sets = _dedup_sets([list(m) for m in matches])
        for atoms in atom_sets:
            n = len(atoms)
            arom_cnt = sum(int(mol.GetAtomWithIdx(a).GetIsAromatic()) for a in atoms)
            hetero_cnt = sum(int(mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)) for a in atoms)
            arom_ratio = arom_cnt / max(n, 1)
            hetero_ratio = hetero_cnt / max(n, 1)
            feat = _one_hot_len(P, pid) + [float(n), float(arom_ratio), float(hetero_ratio)]
            if len(feat) < pad_dim:
                feat += [0.0] * (pad_dim - len(feat))
            else:
                feat = feat[:pad_dim]
            hyper_sets.append(atoms)
            hyper_attrs.append(feat)

    if not hyper_sets:
        # fallback: one placeholder hypernode covering all atoms
        N = mol.GetNumAtoms()
        if N > 0:
            hyper_sets = [list(range(N))]
            hyper_attrs = [[0.0] * pad_dim]

    pairs = []
    for h, atoms in enumerate(hyper_sets):
        for a in atoms:
            pairs.append([a, h])

    h_edge_index = torch.tensor(np.array(pairs, dtype=np.int64)).T.contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)
    h_edge_attr  = torch.tensor(np.array(hyper_attrs, dtype=np.float32)) if hyper_attrs else torch.empty(0, pad_dim, dtype=torch.float32)
    return h_edge_index, h_edge_attr

def get_edge_indices(molecule):
    Chem = _ensure_rdkit()
    edges = []
    for bond in molecule.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    graph_edges = [[u for (u, v) in edges], [v for (u, v) in edges]]
    edge_index = torch.tensor(graph_edges, dtype=torch.long) if edges else torch.empty(2, 0, dtype=torch.long)
    return edge_index, edges

def get_non_peptide_idx(molecule):
    Chem = _ensure_rdkit()
    edges_nonpep = []
    for bond in molecule.GetBonds():
        a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
        n1 = a1.GetAtomicNum(); n2 = a2.GetAtomicNum()
        neigh1 = [n.GetAtomicNum() for n in a1.GetNeighbors()]
        neigh2 = [n.GetAtomicNum() for n in a2.GetNeighbors()]
        hyb1 = str(a1.GetHybridization()); hyb2 = str(a2.GetHybridization())
        h1 = a1.GetTotalNumHs(); h2 = a2.GetTotalNumHs()
        btype = str(bond.GetBondType()); conj = str(bond.GetIsConjugated())
        # heuristic: peptide bond pattern; if NOT peptide-like, mark as non-peptide
        if not (n1 == 6 and n2 == 7 and 8 in neigh1 and hyb1 == 'SP2' and hyb2 == 'SP2' and h1 == 0 and (h2 in (0, 1)) and conj == 'True' and btype == 'SINGLE'):
            if not (n1 == 7 and n2 == 6 and 8 in neigh2 and hyb1 == 'SP2' and hyb2 == 'SP2' and (h2 in (0, 1)) and h2 == 0 and conj == 'True' and btype == 'SINGLE'):
                edges_nonpep.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return edges_nonpep

def get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule):
    Chem = _ensure_rdkit()
    set_with = set(edges_peptidic); set_without = set(edges_nonpeptidic)
    diffs = list(set_with.symmetric_difference(set_without))
    break_idx = []
    for (u, v) in diffs:
        b = molecule.GetBondBetweenAtoms(u, v)
        if b is not None:
            break_idx.append(b.GetIdx())
    mol_f = Chem.FragmentOnBonds(molecule, break_idx, addDummies=False)
    frags = list(Chem.GetMolFrags(mol_f, asMols=True))
    labels = np.empty(0, dtype=np.int64)
    for i, frag in enumerate(frags):
        n = frag.GetNumAtoms()
        labels = np.concatenate([labels, np.ones(n, dtype=np.int64) * i])
    return torch.tensor(labels.tolist(), dtype=torch.long)

def build_residue_hyper_from_fg(sequence: str, molecule, labels_aminoacid_atoms: torch.Tensor, pad_dim_fg: int = None):
    """
    Build residue-centric hypergraph:
      hypernodes: functional groups from SMARTS
      hyperedges: residues
    Hyperedge features: residue 20AA one-hot [E_res, 20]
    Hypernode features: FG features [H, pad_dim_fg]
    Incidence: (fg_id, residue_id) using atom->residue majority vote.
              Each functional group (set of atoms) is assigned to the residue
              that contains most of its atoms after fragmentation.
    """
    h_edge_index, h_edge_attr = build_smarts_hyper(molecule, pad_dim=pad_dim_fg)
    H = int(h_edge_attr.size(0)) if h_edge_attr.numel() > 0 else 0

    aminoacids = get_aminoacids_2(sequence)
    E_res = len(aminoacids)

    h2_edge_attr = torch.zeros((E_res, 20), dtype=torch.float32)
    for ridx, token in enumerate(aminoacids):
        aa = token[0] if len(token) > 0 else 'X'
        idx = AA2IDX.get(aa, None)
        if idx is not None:
            h2_edge_attr[ridx, idx] = 1.0

    if H == 0 or h_edge_index.numel() == 0:
        h2_edge_index = torch.empty(2, 0, dtype=torch.long)
        return h_edge_attr, h2_edge_index, h2_edge_attr, H, E_res

    fg_ids = h_edge_index[1].tolist()
    atom_ids = h_edge_index[0].tolist()
    fg_to_atoms = [[] for _ in range(H)]
    for a, h in zip(atom_ids, fg_ids):
        fg_to_atoms[h].append(a)

    pairs = []
    lab = labels_aminoacid_atoms.cpu().numpy().astype(int)
    for fg_id, atoms in enumerate(fg_to_atoms):
        if not atoms:
            continue
        res_ids, counts = np.unique(lab[atoms], return_counts=True)
        residue_id = int(res_ids[counts.argmax()])
        residue_id = max(0, min(E_res - 1, residue_id))
        pairs.append([fg_id, residue_id])

    h2_edge_index = torch.tensor(np.array(pairs, dtype=np.int64)).T.contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)
    return h_edge_attr, h2_edge_index, h2_edge_attr, H, E_res

# ======================
# Unified public entry
# ======================
def sequences_geodata_1(cc, sequence, y, device=None,
                       mode: str = "pg",
                       k: int = 3,
                       max_len: Optional[int] = None,
                       # max_len/shrink_mode are kept for backward compatibility;
                       # current PG/FG pipelines do not require length limiting.

                       shrink_mode: str = "even",
                       pad_dim_fg: int = None):
    """
    Convert one sample (sequence, label) into HyperData.

    mode:
      - "pg": physicochemical group + k-mer window hypergraph (fast, no RDKit)
      - "fg": SMARTS functional-group hypergraph (needs RDKit)

    Note: We keep tensors on CPU during preprocessing; training code moves batches to GPU.
    """
    mode = str(mode).lower().strip()
    y_tensor = torch.tensor(np.array([float(y)]), dtype=torch.float32)

    if mode == "pg":
        seq = clean_aa20_sequence(sequence)
        x_h, h2_edge_index, h2_edge_attr, H, E_res = build_property_trigram_hyper(seq, k=k)
        dp = HyperData(
            x_h=x_h,
            h2_edge_index=h2_edge_index,
            h2_edge_attr=h2_edge_attr,
            y=y_tensor,
            cc=cc,
            sequence=seq,
        )
    elif mode == "fg":
        Chem = _ensure_rdkit()
        seq = get_sequence_fg(sequence)
        helm_notation = peptide_to_helm(seq, polymer_id='PEPTIDE1')
        molecule = Chem.MolFromHELM(helm_notation)
        if molecule is None:
            raise RuntimeError("RDKit failed to parse HELM for this sequence. Please check the input, or use mode='pg'.")
        _, edges_peptidic = get_edge_indices(molecule)
        edges_nonpeptidic = get_non_peptide_idx(molecule)
        labels_aminoacid_atoms = get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule)

        x_h, h2_edge_index, h2_edge_attr, H, E_res = build_residue_hyper_from_fg(
            sequence=seq, molecule=molecule, labels_aminoacid_atoms=labels_aminoacid_atoms, pad_dim_fg=pad_dim_fg
        )
        dp = HyperData(
            x_h=x_h,
            h2_edge_index=h2_edge_index,
            h2_edge_attr=h2_edge_attr,
            y=y_tensor,
            cc=cc,
            sequence=seq,
        )
    else:
        raise ValueError(f"Unknown mode={mode}. Choose 'fg' or 'pg'.")

    dp.num_hypernodes = int(dp.x_h.size(0))
    dp.num_hyper2edges = int(dp.h2_edge_attr.size(0))
    dp.x = dp.x_h  # for PyG batch vector; not used as a separate graph
    return dp