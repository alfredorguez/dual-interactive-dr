"""Recoloreado de las vistas por selección o por escala de expresión."""

from typing import Optional

import numpy as np
from matplotlib.cm import seismic
from matplotlib.colors import rgb2hex

from dualidr.selection import get_idx_selection
from dualidr.state import AppState


def recolor_by_selection_obs(state: AppState, selection_text: str, color: str) -> None:
    """Colorea las muestras seleccionadas (según ``selection_text``) con ``color``.

    :param state: estado de la aplicación
    :type state: AppState
    :param selection_text: texto de selección de muestras (p.ej. ``state.texto_dr_cond_fil``)
    :type selection_text: str
    :param color: color a asignar
    :type color: str
    :return: None
    """
    idx = get_idx_selection(selection_text, state.ids_muestras, no_empty=True)
    state.source_obs.data["colors"] = [
        color if i in idx else state.source_obs.data["colors"][i] for i in range(state.N)
    ]


def recolor_by_selection_var(state: AppState, selection_text: str, color: str) -> None:
    """Colorea las características seleccionadas (según ``selection_text``) con ``color``.

    :param state: estado de la aplicación
    :type state: AppState
    :param selection_text: texto de selección de características
    :type selection_text: str
    :param color: color a asignar
    :type color: str
    :return: None
    """
    idx = get_idx_selection(selection_text, state.Fx, no_empty=True)
    state.source_var.data["colors"] = [
        color if i in idx else state.source_var.data["colors"][i] for i in range(state.M)
    ]


def apply_expression_colorscale_obs(state: AppState, feature_query: str) -> Optional[str]:
    """Si ``feature_query`` identifica exactamente una característica, colorea
    las muestras según su expresión para esa característica (escala divergente).

    :param state: estado de la aplicación
    :type state: AppState
    :param feature_query: texto introducido (nombre de característica)
    :type feature_query: str
    :return: la etiqueta de la característica encontrada, o None si no aplica
    :rtype: Optional[str]
    """
    idx = get_idx_selection(feature_query, state.Fx)
    if len(idx) == 1:
        state.source_obs.data["colors"] = [
            rgb2hex(seismic(np.interp(x, [-1, 1], [0, 1]))) for x in state.x1[:, idx[0]]
        ]
        return state.Fx[idx[0]]
    return None


def apply_expression_colorscale_var(state: AppState, sample_query: str) -> Optional[str]:
    """Si ``sample_query`` identifica exactamente una muestra, colorea las
    características según su expresión para esa muestra (escala divergente).

    :param state: estado de la aplicación
    :type state: AppState
    :param sample_query: texto introducido (etiqueta de muestra)
    :type sample_query: str
    :return: la etiqueta de la muestra encontrada, o None si no aplica
    :rtype: Optional[str]
    """
    idx = get_idx_selection(sample_query, state.Fy)
    if len(idx) == 1:
        state.source_var.data["colors"] = [
            rgb2hex(seismic(np.interp(x, [-1, 1], [0, 1]))) for x in state.x2[:, idx[0]]
        ]
        return state.Fy[idx[0]]
    return None
