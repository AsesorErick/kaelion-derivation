# Kaelion Derivation v2.0

**Theoretical Foundation and Experimental Predictions for the Kaelion Correspondence**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

---

## Overview

This repository contains:
1. **Theoretical derivation** of λ from tensor networks and QEC
2. **Formal action framework** connecting λ to gravitational physics
3. **Experimental predictions** for analog gravity systems
4. **Publishable paper** (LaTeX + PDF)

---

## Related Work

📦 **Main model:** [kaelion](https://github.com/AsesorErick/kaelion) - DOI: [10.5281/zenodo.18237393](https://doi.org/10.5281/zenodo.18237393)

---

## What's New in v2.0

| Module | Content | Tests |
|--------|---------|-------|
| 26 | Tensor network derivation | 6/6 ✓ |
| 27 | Holographic QEC derivation | 6/6 ✓ |
| **28** | **Analog experiments (BEC, circuits)** | **5/6 ✓** |
| **29** | **Formal action framework** | **5/6 ✓** |

**New:** LaTeX paper ready for journal submission

---

## Key Results

### The Derivation

```
α(λ) = -0.5 - λ  is DERIVED, not fitted

From tensor networks: λ = coarse-graining level
From QEC: λ = accessible bulk information
From action: λ = regularization parameter
```

### Experimental Predictions

**BEC Sonic Black Holes:**
- α should transition from -0.5 to -1.5
- Measurable via correlation functions
- Timescale: seconds

**Superconducting Circuits:**
- OTOC decay 2x faster at λ=1
- Page curve shifts ~5%
- Testable with current technology

### Falsification Criteria

```
α constant → Kaelion falsified
α non-linear → Kaelion modified  
α: -0.5 → -1.5 → Kaelion supported
```

---

## Contents

```
kaelion-derivation/
├── module26_lambda_derivation.py   # Tensor networks
├── module27_qec_lambda.py          # Holographic QEC
├── module28_analog_experiment.py   # BEC & circuits
├── module29_formal_action.py       # Action framework
├── DERIVATION_PAPER.md             # Markdown paper
├── paper/
│   ├── kaelion_paper.tex           # LaTeX source
│   └── kaelion_paper.pdf           # Compiled paper (6 pages)
└── figures/                        # Visualizations
```

---

## Quick Start

```bash
git clone https://github.com/AsesorErick/kaelion-derivation.git
cd kaelion-derivation

# Run all modules
python3 module26_lambda_derivation.py
python3 module27_qec_lambda.py
python3 module28_analog_experiment.py
python3 module29_formal_action.py
```

---

## Paper

The LaTeX paper (`paper/kaelion_paper.pdf`) is ready for submission to journals like:
- Physical Review D
- Journal of High Energy Physics
- Classical and Quantum Gravity

---

## Citation

```bibtex
@software{perez_kaelion_derivation_2026,
  author = {Pérez Eugenio, Erick Francisco},
  title = {Kaelion Derivation v2.0: Theory and Experimental Predictions},
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
