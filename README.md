# Kaelion Derivation v2.1

**Theoretical Foundation, Mainstream Connections, and Experimental Predictions**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

---

## Overview

This repository provides the complete theoretical foundation for the Kaelion correspondence, connecting it to mainstream holography research (SYK, JT gravity) and proposing falsifiable experimental tests.

---

## Related Work

📦 **Main model:** [kaelion](https://github.com/AsesorErick/kaelion) - DOI: [10.5281/zenodo.18237393](https://doi.org/10.5281/zenodo.18237393)

---

## What's in v2.1

| Module | Content | Tests |
|--------|---------|-------|
| 26 | Tensor network derivation | 6/6 ✓ |
| 27 | Holographic QEC derivation | 6/6 ✓ |
| 28 | Analog experiments (BEC, circuits) | 5/6 ✓ |
| 29 | Formal action framework | 5/6 ✓ |
| **30** | **SYK model connection** | **6/6 ✓** |
| **31** | **JT gravity (exact 2D)** | **4/6 ✓** |

**Total: 32/36 tests (88.9%)**

---

## Key Results

### 1. Lambda Derivation (Modules 26-27)
```
α(λ) = -0.5 - λ  DERIVED from:
  • Tensor network coarse-graining
  • Holographic quantum error correction
```

### 2. SYK Connection (Module 30)
```
SYK model:
  • Saturates MSS bound: λ_L = 2πT
  • This corresponds to λ_Kaelion = 1
  • Therefore α = -1.5 (holographic limit)

Key insight: Chaos saturation = Holographic-ness
```

### 3. JT Gravity (Module 31)
```
Exact 2D formula:
  S = S_0 + 2π·φ_h + α(λ)·log(φ_h)
  
  • First exactly solvable Kaelion model
  • Dual to SYK via AdS2/CFT1
  • Confirms α(λ) = -0.5 - λ analytically
```

### 4. Experimental Predictions (Module 28)
```
BEC sonic black holes:
  • α transitions -0.5 → -1.5
  • Measurable with current technology

Superconducting circuits:
  • OTOC decay 2x faster at λ=1
  • Page curve shift ~5%
```

---

## Why This Matters

| Before (v3.1) | After (v2.1) |
|---------------|--------------|
| α(λ) phenomenological | α(λ) derived |
| Disconnected from mainstream | Connected to SYK, JT |
| No exact model | Exactly solvable in 2D |
| General predictions | Specific falsifiable tests |

---

## Repository Structure

```
kaelion-derivation/
├── module26_lambda_derivation.py   # Tensor networks
├── module27_qec_lambda.py          # Holographic QEC
├── module28_analog_experiment.py   # BEC & circuits
├── module29_formal_action.py       # Action framework
├── module30_syk.py                 # SYK model
├── module31_jt_gravity.py          # JT gravity (2D)
├── DERIVATION_PAPER.md             # Theory paper
├── paper/
│   ├── kaelion_paper.tex           # LaTeX source
│   └── kaelion_paper.pdf           # Compiled (6 pages)
└── figures/                        # Visualizations
```

---

## Quick Start

```bash
git clone https://github.com/AsesorErick/kaelion-derivation.git
cd kaelion-derivation

# Run all modules
python3 module26_lambda_derivation.py  # Tensor networks
python3 module27_qec_lambda.py          # QEC
python3 module28_analog_experiment.py   # Experiments
python3 module29_formal_action.py       # Action
python3 module30_syk.py                 # SYK
python3 module31_jt_gravity.py          # JT gravity
```

---

## Citation

```bibtex
@software{perez_kaelion_derivation_2026,
  author = {Pérez Eugenio, Erick Francisco},
  title = {Kaelion Derivation v2.1: SYK, JT Gravity, and Experiments},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.PENDING}
}
```

---

## References

Key papers connected to this work:

1. **Sachdev-Ye (1993), Kitaev (2015)**: SYK model
2. **Maldacena-Stanford (2016)**: SYK and AdS2
3. **Jackiw (1985), Teitelboim (1983)**: JT gravity
4. **Saad-Shenker-Stanford (2019)**: JT gravity path integral
5. **Steinhauer (2016)**: BEC Hawking radiation

---

## License

MIT License

---

## Author

Erick Francisco Pérez Eugenio  
January 2026
