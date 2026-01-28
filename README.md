# HyperPep

## Abstract
Accurate identification of T cell receptor (TCR)–peptide pairing is central to immune monitoring and precision immunotherapies, yet existing sequence-based predictors often degrade under distribution shift, unseen TCR/peptide combinations, and screening-like class imbalance. Here we present **HyperPep**, a chemistry-informed hypergraph learning framework that encodes the concatenated TCR and peptide sequences as a unified hypergraph, and models residue semantics through two interchangeable hypernode instantiation strategies. In the functional-group (FG) mode, hypernodes are derived from SMARTS-defined side-chain substructures to capture chemistry-aligned semantics; in the physicochemical-group (PG) mode, residues are aggregated by property-based grouping while incorporating local sequence context to form a complementary, coarser-grained semantic view. HyperPep further couples hypergraph message passing with a residue-chain refinement module, jointly capturing higher-order chemical semantics and sequential dependencies.

We systematically evaluate HyperPep on both single-chain TCRβ–peptide and paired-chain TCRαβ–peptide tasks, spanning standard cross-validation, external transfer, and zero-shot extrapolation, and further stress-test robustness under realistic screening conditions with severe class imbalance and stringent held-out protocols. Across these evaluation regimes and data distributions, HyperPep consistently exhibits stable advantages—outperforming representative baselines in both single- and paired-chain settings—and maintains more reliable generalization and recall when extrapolating to unseen peptides/unseen TCRs under highly imbalanced screening scenarios. Collectively, HyperPep offers a scalable, chemistry-interpretable hypergraph learning solution for deployable TCR–peptide pairing prediction, providing computational support for immune monitoring and the design of personalized immunotherapy strategies.

---

## Environment
- Python 3.10.15  
- numpy == 1.26.4  
- pytorch == 1.13.1  
- torch-scatter == 2.1.0  
- torch-sparse == 0.6.15  
- scipy == 1.14.1  

---

## Usage

### 1) TCRβ–peptide tasks
```bash
python main.py --load_ckpt checkpoints/hyperpep_fg.pt --test_name test_zero-shot.csv
```
```bash
python main.py --load_ckpt checkpoints/hyperpep_fg.pt --test_name test_covid19.csv
```
### 2) TCRαβ–peptide tasks
```bash
python main.py --load_ckpt checkpoints/hyperpep_pg_k3.pt --test_name test_tcr_split.csv
```
```bash
python main.py --load_ckpt checkpoints/hyperpep_pg_k3.pt --test_name test_strict_split.csv
```



