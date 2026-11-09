import sys
import warnings

import numpy as np
from bokeh.events import SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import ColumnDataSource, Div
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

sys.path.append("src")

from dualidr.data import get_data_with_meta
from dualidr import colors, dr_engine, feature_importance, figures, selection, widgets
from dualidr.state import AppState

# =============== Obtención de datos ===============

# Matriz de características, etiquetas de muestras, etiquetas de características,
# registros originales y metadatos de ventana (window_id, record_id, start, end)
F, Fy, Fx, raw, window_meta = get_data_with_meta()

encoder_obs = LabelEncoder()
encoder_obs.fit(Fy)
Fy_enc = encoder_obs.transform(Fy)

encoder_var = LabelEncoder()
encoder_var.fit(Fx)
Fx_enc = encoder_var.transform(Fx)

scaler = MinMaxScaler()
encoder = LabelEncoder()

X = F
n_samples = F.shape[0]
n_features = F.shape[1]

ids_muestras = list(map(str, np.arange(n_samples).tolist()))

# muestras
x1 = X.copy()
# armónicos
x2 = X.T.copy()

# 2D map initialization
print("initializing maps (samples, features) ...")
pca = PCA(n_components=2, random_state=1)
y1 = pca.fit_transform(X)
y2 = pca.fit_transform(X.T)

N = x1.shape[0]  # número de muestras
M = x2.shape[0]  # número de features

print(f"{N}")

# selection colors
colors_obs = ["#000000"] * N
colors_var = ["#000000"] * M

dotSize_obs = np.ones(N) * 7
dotSize_var = np.ones(M) * 7
importances_var = np.zeros(M)

# column datas sources
source_obs = ColumnDataSource(
    {"x1": y1[:, 0], "y1": y1[:, 1], "labels": Fy, "colors": colors_obs, "ids": ids_muestras, "size": dotSize_obs}
)
source_var = ColumnDataSource(
    {"x2": y2[:, 0], "y2": y2[:, 1], "labels": Fx, "colors": colors_var, "size": dotSize_var, "importance": importances_var}
)

state = AppState(
    X=X,
    Fx=Fx,
    Fy=Fy,
    ids_muestras=ids_muestras,
    source_obs=source_obs,
    source_var=source_var,
    x1=x1,
    x2=x2,
    y1=y1,
    y2=y2,
    pca=pca,
    N=N,
    M=M,
    idx_obs=np.arange(n_samples),
    idx_var=np.arange(n_features),
    raw=raw,
    window_meta=window_meta,
)

# =============== Widgets y figuras ===============

w = widgets.build_widgets(source_var)

fig1, labels_obs = figures.build_sample_figure(source_obs)
fig2, labels_var = figures.build_feature_figure(source_var)
fig3 = figures.build_importance_figure(source_var)

# compute affinities by first time
dr_engine.update_affinities(state, w.slider_perplexity_1.value, w.slider_perplexity_2.value)


# =============== Callbacks ===============


def _on_select_obs(event):
    w.textinput_colorsel_obs.value = selection.compute_selection_fil(state, source_obs.selected.indices)


def _on_select_var(attr, old, new):
    # Se dispara con el lazo de fig2, con la franja de la gráfica de importancia
    # (resaltado cruzado) y con el reset de fig2 (new == [])
    w.textinput_colorsel_var.value = selection.compute_selection_col(state, list(new))


fig1.on_event(SelectionGeometry, _on_select_obs)
source_var.selected.on_change("indices", _on_select_var)


def _on_conditional_dr():
    text_obs, text_var = selection.update_conditional_dr(
        state, w.textinput_colorsel_var.value, w.slider_perplexity_1.value, w.slider_perplexity_2.value
    )
    w.textinput_colorsel_obs.value = text_obs
    w.textinput_colorsel_var.value = text_var


w.button_conditional_dr.on_click(_on_conditional_dr)


def _on_reset():
    dr_engine.reset_original_state(state)

    # set parameters to their default values
    print("resetting DR parameters to defaults...")
    w.slider_perplexity_1.value = 20
    w.slider_lr_1.value = 1

    # select all samples and features (default)
    w.textinput_colorsel_obs.value = ""
    w.textinput_colorsel_var.value = ""

    _on_conditional_dr()


w.button_reset.on_click(_on_reset)


def _on_button_obs():
    colors.recolor_by_selection_obs(state, state.texto_dr_cond_fil, w.textinput_color_obs.value)
    print(f"update_color_obs_1: {state.texto_dr_cond_fil}, {len(state.idx_obs)} elementos")


w.button_obs.on_click(_on_button_obs)


def _on_button_var():
    colors.recolor_by_selection_var(state, w.textinput_colorsel_var.value, w.textinput_color_var.value)


w.button_var.on_click(_on_button_var)


def _on_color_obs_change(attr, old, new):
    label = colors.apply_expression_colorscale_obs(state, new)
    if label is not None:
        w.textinput_color_obs.value = label
    else:
        colors.recolor_by_selection_obs(state, state.texto_dr_cond_fil, new)
    print(attr, old, new)


w.textinput_color_obs.on_change("value", _on_color_obs_change)


def _on_color_var_change(attr, old, new):
    label = colors.apply_expression_colorscale_var(state, new)
    if label is not None:
        w.textinput_color_var.value = label
    else:
        colors.recolor_by_selection_var(state, w.textinput_colorsel_var.value, new)
    print(attr, old, new)


w.textinput_color_var.on_change("value", _on_color_var_change)


def _on_checkbox(attr):
    labels_obs.visible = 0 in attr
    labels_var.visible = 1 in attr


w.checkbox_group.on_click(_on_checkbox)


def _on_slider_perplexity_1_change(att, old, new):
    dr_engine.update_affinities(state, w.slider_perplexity_1.value, w.slider_perplexity_2.value)


w.slider_perplexity_1.on_change("value", _on_slider_perplexity_1_change)


def _on_slider_perplexity_2_change(att, old, new):
    dr_engine.update_affinities(state, w.slider_perplexity_1.value, w.slider_perplexity_2.value)


w.slider_perplexity_2.on_change("value", _on_slider_perplexity_2_change)


def _on_change_exaggeration():
    if w.button_exaggeration.label == "exaggeration = ON":
        w.button_exaggeration.label = "exaggeration = OFF"
        state.early_exaggeration = 1
    else:
        w.button_exaggeration.label = "exaggeration = ON"
        state.early_exaggeration = 2


w.button_exaggeration.on_click(_on_change_exaggeration)


def _run_feature_importance(l1_ratio):
    # La discriminación usa lo seleccionado en las gráficas en este momento
    sample_idx = list(source_obs.selected.indices)
    feature_idx = list(state.sel_col_idx)
    result = feature_importance.train_feature_importance(
        state, l1_ratio=l1_ratio, sample_idx=sample_idx, feature_idx=feature_idx
    )
    if result.ok:
        n_features = len(feature_idx) if feature_idx else state.M
        w.f1_textbox.value = f"F1 Score : {result.f1:.3f} ({len(sample_idx)} samples, {n_features} features)"
        feature_importance.update_importance_barchart(state, fig3)
    else:
        w.f1_textbox.value = f"F1 Score : n/a ({result.reason})"


def _on_feature_importance():
    _run_feature_importance(l1_ratio=1)


w.button_feature_importance.on_click(_on_feature_importance)


def _on_slider_l1_change(attr, old, new):
    _run_feature_importance(l1_ratio=new)


w.slider_l1.on_change("value", _on_slider_l1_change)


def _periodic_update_tsne():
    dr_engine.update_tsne(state, w.slider_lr_1.value, w.slider_lr_2.value)


curdoc().add_periodic_callback(_periodic_update_tsne, 50)  # callback lenta

# ELEMENTO DE TEXTO
cabecera = Div(
    text="""
	<h1>Dual DiDR of electromechanical tests with an induction machine</h1>
	<p>Author: Alfredo Rodríguez Magdalena</p>""",
    width=1000,
    height=100,
)

curdoc().add_root(
    layout(
        column(
            cabecera,
            row(
                column(fig1, w.slider_perplexity_1, w.slider_lr_1, w.textinput_color_obs, w.textinput_colorsel_obs),
                column(fig2, w.slider_perplexity_2, w.slider_lr_2, w.textinput_color_var, w.textinput_colorsel_var),
                column(
                    w.button_conditional_dr,
                    w.button_reset,
                    w.button_obs,
                    w.button_var,
                    w.checkbox_group,
                    w.button_exaggeration,
                    w.button_feature_importance,
                    w.slider_l1,
                    w.f1_textbox,
                    w.button_download,
                ),
                column(fig3),
            ),
        )
    )
)
