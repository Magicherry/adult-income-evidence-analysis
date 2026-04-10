# Adult Income Triangulated Evidence Project

This repo explores a reproducible machine learning study of the UCI Adult (Census Income) dataset, built around an analysis-first workflow. The pipeline runs through a single sequential notebook, with reusable components in `src/`.

## Environment Setup

Use the dedicated Conda environment only.

```powershell
conda env create -f environment.yml
conda activate adult-income-ml
python --version
python -c "import pandas, numpy, sklearn, scipy, matplotlib; print('ok')"
```

Expected verified versions after installation:

- `Python 3.12.13`
- `pandas 3.0.2`
- `numpy 2.4.4`
- `scikit-learn 1.8.0`
- `scipy 1.17.1`
- `matplotlib 3.10.8`
- `ipykernel 7.2.0`
- `nbformat 5.10.4`

If the environment already exists and `requirements.txt` changes:

```powershell
conda activate adult-income-ml
python -m pip install -r requirements.txt
```

## Project Structure

- `data/raw/adult.csv`: raw dataset
- `data/processed/splits/`: saved split manifests
- `src/`: reusable analysis code
- `outputs/`: figures, tables, and metrics
- `report/`: optional hand-written report drafts
- `notebooks/experiments.ipynb`: single sequential notebook that runs the full workflow inline with `src/` helpers

## Run Order

Run the project from the repository root with the Conda environment activated.

```powershell
jupyter lab notebooks/experiments.ipynb
```

Open `notebooks/experiments.ipynb` and run it from top to bottom.

## Reports

- English report: [report/report.md](report/report.md)
- Chinese report: [report/report_zh.md](report/report_zh.md)

## Reproducibility Rules

- The analysis uses a dedicated Python 3.12 Conda environment named `adult-income-ml`.
- The main split uses seed `42`. Robustness reruns use seeds `7`, `42`, and `99`.
- All tuning is done on the training split only.
- The test split is reserved for final evaluation and robustness reruns.
- The notebook saves structured artifacts under `outputs/` and displays tables and figures inline as it runs.
