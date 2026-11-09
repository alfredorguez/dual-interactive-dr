# Dual DiDR

Dual interactive dimensionality reduction (t-SNE) of electromechanical tests on an asynchronous machine. The application displays the **sample space** and the **feature space** side by side, lets you select points in one to condition the other, and uses an elastic-net logistic regression to estimate which features separate the selection.

Authors © 2026:

- Alfredo Rodríguez Magdalena (main author) — a.rodmagdalena@gmail.com
- Ignacio Díaz Blanco — idiaz@uniovi.es
- Jose María Enguita González — j.menguita@uniovi.es

> **Publication status:** the paper describing this work was submitted on 18 November 2026 to [ESANN 2027](https://www.esann.org/) (European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning).

## Requirements

- Python 3.10 (`==3.10.*`)
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- C/C++ compiler with OpenMP if openTSNE has to be built from source (the `Dockerfile` already includes it)
- Data file `data/dataicann.mat` (included in the repository).

## Getting started

All commands must be run **from the repository root** (the paths to `data/` and `src/` are relative).

### With uv

```bash
uv sync
uv run bokeh serve src/dualIDR.py --show
```

### With pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bokeh serve src/dualIDR.py --show
```

The application is then available at <http://localhost:5006/dualIDR>.

### With Docker

```bash
docker build -t dual-didr .
docker run --rm -p 5006:5006 dual-didr
```

The image copies `src/` and `data/` at build time, so `data/dataicann.mat` must be present before running `docker build`. The container serves Bokeh on port 5006 with `--allow-websocket-origin=*`.

## Using the application

1. Adjust `perplexity` and `learning_rate (log)` for each view; t-SNE is optimized live.
2. Select points in a view with the lasso: they are listed in the text boxes (*selected samples/features*).
3. Click **use selected features & samples** to recompute the projection conditioned on that selection; **reset projections** returns to the initial PCA.
4. Color selections with the **selection ➡ color** buttons, or type the name of a single feature/sample to color by its expression.
5. **Discriminate** runs the classification pipeline with whatever is lasso-selected **at that moment** (there is no need to click *use selected…* first): the selected samples against the rest and, if there is a lasso in the feature view, using only those features (the rest get an importance of 0). It shows the test F1 and sizes/colors the features according to the coefficients; the *L1 ratio* slider controls the elastic-net term.
6. **Download CSV** exports the importance of each feature.

### How the F1 Score is computed

The F1 is a **test** score on a temporal hold-out, with no information leakage:

1. In each record a temporal cut is chosen (target: 70 % training / 30 % test), trying to keep both classes (selected windows = 1) on both sides.
2. Windows that cross the cut are excluded, along with a guard of 400 samples after it.
3. The signal and feature normalizers and the elastic-net logistic regression are fitted **only on the training data**; the F1 (class 1) is evaluated on the test data. The coefficients are those of that classifier.
4. If train or test do not have enough windows of each class (e.g. a single-class selection), it is not computed and `F1 Score : n/a (reason)` is shown without changing colors or sizes.

### Tests

```bash
uv run pytest tests
```

## Structure

```
src/
├── dualIDR.py        # assembly script of the Bokeh app (callbacks)
└── dualidr/          # logic: data pipeline (signals -> windows -> normalized features), windows, classification (hold-out), state, t-SNE engine, selection, importance, figures, widgets
tests/                # tests with synthetic data
data/                 # input data (dataicann.mat)
outputs/              # generated figures and CSVs (not versioned)
docs/                 # supporting material
```

## Git workflow

- `main`: stable changes only.
- `develop`: development and integration. Features branch off `develop` (`feature/<description>`) and merge back into it; `main` is updated from `develop` once stable.
- Commits in English following [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).

## Citation

If you use this software or the ideas behind it in your work, please cite our paper. It has been submitted to ESANN 2027 and is not published yet; the reference will be added here once it is available.

### Dataset

The experimental data in `data/dataicann.mat` comes from the Dataicann dataset. If you use it, please also cite:

```bibtex
@misc{diaz_dataicann_2019,
  author = {D{\'i}az Blanco, Ignacio and
            Cuadrado Vega, Abel Alberto and
            Gonz{\'a}lez Mu{\~n}iz, Ana and
            Garc{\'i}a P{\'e}rez, Diego},
  title  = {{Dataicann}: datos de vibraci{\'o}n y corriente de un motor de inducci{\'o}n},
  year   = {2019},
  publisher = {Universidad de Oviedo},
  url    = {https://hdl.handle.net/10651/53461},
  note   = {Dataset}
}
```

## License

- **Code:** released under the [MIT License](LICENSE).
- **Dataset:** `data/dataicann.mat` is part of the *Dataicann* dataset (Díaz Blanco et al., Universidad de Oviedo, 2019, <https://hdl.handle.net/10651/53461>) and is **not** covered by the MIT License. It is distributed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. This means that you must credit the authors (see [Citation](#citation)), you may not use the dataset for commercial purposes, and you may not redistribute modified versions of it.
