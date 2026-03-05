# henry-fan-research

> **Henry Fan · San Jose State University CS**  
> Research pipeline for CS PhD applications — 3 papers, fully reproducible, CPU-only.

---

## Papers

### Paper 1 — Classical vs. Deep Learning on Small Datasets
> *A Reproducible Benchmark*

**Research question:** On small tabular datasets (*n* < 1000), does deep learning outperform classical ML — and at what cost?

**Key finding:** SVM and Logistic Regression match or beat MLP on all 4 datasets at 10–100× lower training cost. Friedman test confirms significance on 3/4 datasets.

```bash
cd paper1-small-dataset-benchmark
python run_benchmark.py --fast     # 2 min, skip MNIST
python analysis/visualize.py
python analysis/stats.py
```

📄 `paper1-small-dataset-benchmark/paper/main.tex`  
📊 5 figures · 100-row results CSV · Friedman + Wilcoxon tests  
🎯 Target: **arXiv cs.LG** → **ML Reproducibility Workshop (NeurIPS)**

---

### Paper 2 — Reproducible ML Curriculum for Self-Taught Engineers
> *A Concept-Coverage and Pedagogical-Depth Study*

**Research question:** How do five widely-used ML curricula compare across coverage, scaffolding, reproducibility, and accessibility?

**Key finding:** Project-first curricula score 0.40+ higher than lecture-first approaches. No curricula except this repo cover fairness auditing.

```bash
cd paper2-ml-curriculum
python run_analysis.py
```

📄 `paper2-ml-curriculum/paper/main.tex`  
📊 5 figures · composite score framework · Spearman scaffolding analysis  
🎯 Target: **arXiv cs.CY** → **ICLR Workshop on ML Education**

---

### Paper 3 — Training ML Models Without a GPU
> *A Practical CPU Efficiency Benchmark*

**Research question:** Which ML models are most efficient on CPU-only hardware, and how do parallelism, dataset size, and dimensionality affect training time?

**Key finding:** Logistic Regression achieves 143× higher CPU efficiency than MLP. CPU parallelism (n_jobs) provides no speedup below n=10,000.

```bash
cd paper3-cpu-efficiency
python run_experiments.py
```

📄 `paper3-cpu-efficiency/paper/main.tex`  
📊 4 figures · 71 experiments · parallelism + scaling + efficiency + dimensionality  
🎯 Target: **arXiv cs.LG** → **NeurIPS Workshop on Efficient ML**

---

## Repo Structure

```
henry-fan-research/
│
├── paper1-small-dataset-benchmark/
│   ├── run_benchmark.py
│   ├── data/loaders.py
│   ├── models/registry.py
│   ├── analysis/{visualize,stats}.py
│   ├── results/          ← generated CSVs
│   ├── figures/          ← generated PDFs + PNGs
│   └── paper/{main.tex, references.bib}
│
├── paper2-ml-curriculum/
│   ├── run_analysis.py
│   ├── results/
│   ├── figures/
│   └── paper/{main.tex, references.bib}
│
├── paper3-cpu-efficiency/
│   ├── run_experiments.py
│   ├── results/
│   ├── figures/
│   └── paper/{main.tex, references.bib}
│
├── docs/                 ← GitHub Pages site
│   └── index.html
│
├── requirements.txt
└── README.md             ← this file
```

---

## Quickstart (all 3 papers)

```bash
git clone https://github.com/HenryFan97/henry-fan-research
cd henry-fan-research
pip install -r requirements.txt

# Paper 1 (~5 min)
cd paper1-small-dataset-benchmark && python run_benchmark.py && python analysis/visualize.py && python analysis/stats.py && cd ..

# Paper 2 (~10 sec)
cd paper2-ml-curriculum && python run_analysis.py && cd ..

# Paper 3 (~5 min)
cd paper3-cpu-efficiency && python run_experiments.py && cd ..
```

Total runtime: **< 12 minutes on any laptop CPU. No GPU required.**

---

## Publication Roadmap

| Paper | Status | Next Step | Target Venue |
|-------|--------|-----------|--------------|
| Paper 1 | ✅ Experiments done · Draft complete | Submit to arXiv | NeurIPS ML Reproducibility Workshop |
| Paper 2 | ✅ Experiments done · Draft complete | Submit to arXiv | ICLR Workshop on ML Education |
| Paper 3 | ✅ Experiments done · Draft complete | Submit to arXiv | NeurIPS Efficient ML Workshop |

**Submit Paper 1 to arXiv first** — it's the strongest standalone contribution.

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
scipy>=1.10.0
```

No GPU. No cloud compute. Runs on any laptop made after 2015.

---

## About

Henry Fan · BS Computer Science, San Jose State University  
Building toward CS PhD applications · Focus: ML systems, reproducibility, equitable ML education  
GitHub: [github.com/HenryFan97](https://github.com/HenryFan97)
