# Kaelion Derivation

**Theoretical Foundation for the Kaelion Correspondence**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

---

## Overview

This repository contains the theoretical derivation of the Kaelion interpolation parameter λ from first principles. 

**The key result:** The relation α(λ) = -0.5 - λ is **derived**, not phenomenologically fitted.

---

## Related Work

This derivation provides theoretical foundation for the Kaelion model:

📦 **Main repository:** [kaelion](https://github.com/AsesorErick/kaelion) - DOI: [10.5281/zenodo.18237393](https://doi.org/10.5281/zenodo.18237393)

If you're new to Kaelion, start there. Come here when you want to understand **why** the model works.

---

## Contents

| File | Description |
|------|-------------|
| `module26_lambda_derivation.py` | Derivation from tensor network coarse-graining |
| `module27_qec_lambda.py` | Derivation from holographic quantum error correction |
| `DERIVATION_PAPER.md` | Complete theoretical paper |
| `figures/` | Visualizations |

---

## The Derivation in Brief

### The Problem

Kaelion proposes: α(λ) = -0.5 - λ

But **why** this specific form?

### The Solution

**Two independent derivations give the same answer:**

### 1. Tensor Networks (Module 26)

```
λ = coarse-graining level in MERA-like tensor network

- Layer 0 (bulk): λ = 0, α = -0.5 (LQG)
- Layer n (boundary): λ = 1, α = -1.5 (CFT)
- Each layer contributes equally: Δα = -1/n per layer
- After fraction λ: α = -0.5 - λ ✓
```

### 2. Holographic QEC (Module 27)

```
λ = accessible bulk information from boundary

- 0% boundary access: λ = 0, α = -0.5
- 100% boundary access: λ = 1, α = -1.5
- Interpolation follows error correction structure
- Result: α = -0.5 - λ ✓
```

### Convergence

Two completely different approaches → Same result

This is strong evidence the relation is **fundamental**.

---

## Verification Results

| Module | Tests | Passed | Key Result |
|--------|-------|--------|------------|
| 26 - Tensor Networks | 6 | 6 (100%) | α(λ) = -0.5 - λ derived |
| 27 - Holographic QEC | 6 | 6 (100%) | Same result independently |

---

## Quick Start

```bash
git clone https://github.com/AsesorErick/kaelion-derivation.git
cd kaelion-derivation

# Run tensor network derivation
python3 module26_lambda_derivation.py

# Run QEC derivation
python3 module27_qec_lambda.py
```

---

## Key References

1. **Swingle (2012)** - Entanglement Renormalization and Holography
2. **Pastawski et al. (2015)** - HaPPY code (holographic QEC)
3. **Harlow (2016)** - Ryu-Takayanagi from QEC
4. **Kaul-Majumdar (2000)** - α = -1/2 from LQG
5. **Sen (2012)** - α = -3/2 from CFT

---

## Citation

```bibtex
@software{perez_kaelion_derivation_2026,
  author = {Pérez Eugenio, Erick Francisco},
  title = {Kaelion Derivation: Theoretical Foundation from Tensor Networks and QEC},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.PENDING}
}
```

---

## License

MIT License

---

## Author

Erick Francisco Pérez Eugenio  
January 2026

---

## Navigation

| Want to... | Go to |
|------------|-------|
| Use the Kaelion model | [kaelion](https://github.com/AsesorErick/kaelion) |
| Understand why it works | **You're here** |
