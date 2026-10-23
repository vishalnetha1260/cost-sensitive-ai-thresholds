# Cost-Sensitive AI Thresholds for Decision Intelligence

Code and data for the Short Communication **"Cost-Sensitive AI Thresholds for Decision Intelligence: Optimizing Data-Driven Cutoffs Under Asymmetric Business Loss"** by Vishal Netha Devasani (Doctor of Business Administration Program, Midwest University). The manuscript was prepared for *Knowledge-Based Systems* (Elsevier).

One script, `main.py`, regenerates every number, table and figure in the paper from a public dataset.

## Key findings

AI credit models are usually run at accuracy-oriented cutoffs, which treat a wrongful approval and a wrongful decline as equally costly. A knowledge-based decision layer instead turns an explicit business loss ratio *r* = C<sub>FN</sub>/C<sub>FP</sub> into the closed-form cutoff *t\** = 1/(1 + *r*), with no retraining.

| Loss ratio *r* (calibrated GBDT, 20 splits) | 3:1 | 5:1 | 10:1 |
|---|---|---|---|
| Cost reduction vs. accuracy-optimal cutoff | 13.0% | 25.0% | 51.6% |
| 95% CI (Nadeau–Bengio corrected) | 7.9–18.2% | 19.9–30.1% | 48.0–55.2% |

- The closed-form cutoff is statistically indistinguishable from a validation-searched cutoff. It is competitive with cost-weighted retraining, which needs one model per ratio.
- Calibration is the enabling condition. Class-balanced training breaks the rule (51.5% regret at 2:1). A closed-form prior correction or isotonic recalibration brings the regret down to about 0.7%.
- When managers know only an interval (3 ≤ *r* ≤ 10), a minimax-regret cutoff limits worst-case regret to 17.4%. The accuracy-optimal cutoff reaches 109.5%.

## Repository layout

```
main.py                          complete experiment: data, models, decision layer, statistics, figures
requirements.txt                 Python dependencies
data/uci_credit_default.csv.gz   UCI "Default of Credit Card Clients" (CC BY 4.0), verified copy
results/                         outputs written by main.py (committed for transparency)
LICENSE                          MIT (code); data license noted inside
```

## Reproduce

```bash
git clone https://github.com/vishalnetha1260/cost-sensitive-ai-thresholds.git
cd cost-sensitive-ai-thresholds
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # full protocol, 20 repeated splits (about 10 min on 2 CPU cores)
python main.py --repeats 5         # quicker check
```

The paper's numbers were produced with Python 3.13, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1 and matplotlib 3.11.2. Runs are deterministic for a fixed seed. Other library versions can change the last digits.

## Outputs and where they appear in the paper

| File in `results/` | Paper element |
|---|---|
| `table1_cost_results.csv` | Table 1: expected cost of each cutoff policy for six loss ratios |
| `table2_ablation_robustness.csv` | Table 2: calibration ablation (Panel A) and interval knowledge (Panel B) |
| `Figure_1.png` / `.pdf` | Fig. 1: cost curves and savings vs. loss ratio |
| `Figure_2.png` / `.pdf` | Fig. 2: reliability diagram and misspecification regret surface |
| `Graphical_Abstract.png` / `.pdf` | Graphical abstract |
| `per_split_decisions.csv` | Every policy × ratio × split (supporting data) |
| `regret_surface_gbdt.csv` | Data behind Fig. 2(b), plus the accuracy-cutoff regret |
| `run_metadata.json` | Library versions, data hash, protocol and runtimes |

## Method in brief

- **Data.** 30,000 credit-card clients and 6,636 defaults (22.12%). Sex, marital status and age are excluded as a fair-lending precaution. Five behavioral features are added, for 27 inputs in total.
- **Predictive layer.** Logistic regression and histogram gradient-boosted trees, each calibrated by five-fold isotonic regression on the training split.
- **Knowledge-based decision layer.**
  - Cutoff for a known loss ratio: *t\**(*r*) = 1/(1 + *r*).
  - Prior correction for re-weighted models: *p* = *k p<sub>w</sub>* / (*k p<sub>w</sub>* + 1 − *p<sub>w</sub>*), with *k* = *w<sub>0</sub>*/*w<sub>1</sub>*.
  - Minimax-regret cutoff when only an interval [*r<sub>lo</sub>*, *r<sub>hi</sub>*] is known.
- **Baselines.** Default 0.5 cutoff, accuracy-optimal, F1-optimal, validation cost search, and cost-weighted retraining (one model per ratio).
- **Inference.** 20 stratified 60/20/20 splits (seed 2026 + *j*), Nadeau–Bengio corrected resampled t-test and Holm adjustment.

## Data provenance

Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. License: CC BY 4.0.

`data/uci_credit_default.csv.gz` is a lossless CSV conversion of the original `.xls`. The descriptive header row is kept and the target column is renamed to `DEFAULT`. `main.py` checks the row count (30,000), the default count (6,636) and a SHA-256 content hash (`debf24d7…c83b`). If the file is missing, `main.py` downloads the original archive from UCI; reading the `.xls` requires `pip install xlrd`.

## Citation

```bibtex
@unpublished{devasani2026costsensitive,
  author = {Devasani, Vishal Netha},
  title  = {Cost-Sensitive {AI} Thresholds for Decision Intelligence: Optimizing Data-Driven Cutoffs Under Asymmetric Business Loss},
  note   = {Manuscript prepared for submission to Knowledge-Based Systems},
  year   = {2026},
  url    = {https://github.com/vishalnetha1260/cost-sensitive-ai-thresholds}
}
```

## License

The code is released under the MIT License. The dataset is redistributed under its original CC BY 4.0 license, with attribution above.
