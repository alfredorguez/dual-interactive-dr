"""Importancia de características a partir del pipeline de clasificación y su gráfico de barras."""

from typing import Optional, Sequence

import numpy as np
from bokeh.plotting import figure
from matplotlib.cm import coolwarm
from matplotlib.colors import rgb2hex

from dualidr.classification import (
    ClassificationParams,
    ClassificationResult,
    run_classification_pipeline,
)
from dualidr.state import AppState

# Procedimiento para la importancia de las características:
#   1. Sobre el espacio muestral -> se asigna la clase "1" a las ventanas seleccionadas
#   2. Pipeline de clasificación (hold-out temporal): devuelve F1 de test y coeficientes
#   3. Se determinan las variables que afectan positiva y negativamente (escala de color divergente)
#   4. Se obtiene la importancia de los features (valor absoluto de los coeficientes normalizados)
#   5. Se asigna el tamaño de los puntos en función de la importancia de las features


def train_feature_importance(
    state: AppState,
    l1_ratio: float,
    sample_idx: Sequence[int],
    feature_idx: Optional[Sequence[int]] = None,
) -> ClassificationResult:
    """Ejecuta el pipeline de clasificación con las ventanas seleccionadas en la gráfica
    (``sample_idx``) como clase 1 y, si es válido, actualiza en ``state`` el F1, los
    colores y los tamaños de ``source_var`` según la importancia de cada característica.

    :param state: estado de la aplicación
    :type state: AppState
    :param l1_ratio: ratio L1 del término elasticnet
    :type l1_ratio: float
    :param sample_idx: índices de ventana (``window_id``) seleccionados en el espacio muestral
    :type sample_idx: Sequence[int]
    :param feature_idx: características seleccionadas con las que entrenar; vacío o None usa todas
    :type feature_idx: Optional[Sequence[int]]
    :return: resultado del pipeline; si ``ok`` es False el estado visual no se modifica
    :rtype: ClassificationResult
    """
    if len(sample_idx) == 0:
        return ClassificationResult(ok=False, reason="no samples selected")

    y = np.zeros(len(state.window_meta), dtype=int)
    y[list(sample_idx)] = 1
    columns = np.asarray(feature_idx, dtype=int) if feature_idx else None

    result = run_classification_pipeline(
        state.raw, state.window_meta, y, ClassificationParams(l1_ratio=l1_ratio), columns
    )
    state.classification = result
    if not result.ok:
        return result

    state.f1 = result.f1
    max_abs = np.max(np.abs(result.coef))
    coeffs_normalized = result.coef / max_abs if max_abs > 0 else result.coef

    colors_var = [rgb2hex(coolwarm((c + 1) / 2)) for c in coeffs_normalized]

    imp_var = np.abs(coeffs_normalized)
    state.importances_var = imp_var
    state.source_var.data["size"] = 7 + (imp_var * 30)  # escalado al rango [7, 37]
    state.source_var.data["colors"] = colors_var
    state.source_var.data["importance"] = imp_var

    return result


def update_importance_barchart(state: AppState, fig3: figure) -> None:
    """Actualiza el gráfico de barras de importancia de características,
    ordenando de manera que las variables de tonos azulados queden en la parte
    inferior y las de tonos rojizos en la parte superior, manteniendo las
    etiquetas originales.

    :param state: estado de la aplicación
    :type state: AppState
    :param fig3: figura Bokeh del gráfico de importancia
    :type fig3: figure
    :return: None
    """
    labels = state.source_var.data["labels"]
    importances = state.source_var.data["importance"]
    colors = state.source_var.data["colors"]

    def estimate_coolwarm_value(color: str) -> float:
        """Devuelve un escalar de 0 (azul) a 1 (rojo) para cada color aproximado."""
        hex_cmap = [rgb2hex(coolwarm(i / 255.0)) for i in range(256)]
        if color in hex_cmap:
            return hex_cmap.index(color) / 255.0
        return 0.5  # si no se puede estimar, se asume neutral

    scores = [estimate_coolwarm_value(c) for c in colors]

    sorted_indices = np.argsort(scores)
    labels = [labels[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    fig3.y_range.factors = labels
    fig3.renderers[0].data_source.data = {
        "feature_label": labels,
        "importance": importances,
        "color": colors,
    }
