"""Selección de muestras/características y disparo de la reducción condicional."""

from typing import List, Tuple

from dualidr.dr_engine import update_affinities
from dualidr.state import AppState


def get_idx_selection(text: str, lista_labels: List[str], no_empty: bool = True) -> List[int]:
    """Selecciona todos los elementos de ``lista_labels`` que coincidan
    (literalmente) con alguna de las subcadenas de ``text`` separadas por comas.

    :param text: cadena introducida por el usuario con una lista de elementos
    :type text: str
    :param lista_labels: lista de etiquetas de las que se seleccionarán los índices
    :type lista_labels: List[str]
    :param no_empty: si es True, se selecciona todo cuando ``text`` está vacío
    :type no_empty: bool
    :return: índices seleccionados
    :rtype: List[int]
    """
    idx_sel: List[int] = []
    if len(text) > 0:
        items = text.replace(" ", "").split(",")
        aux = list(map(lambda y: [i for i, X in enumerate(lista_labels) if y == X], items))
        idx_sel = list(set().union(*aux))
    elif no_empty:
        idx_sel = [i for i, v in enumerate(lista_labels)]

    return idx_sel


def compute_selection_fil(state: AppState, indices: List[int]) -> str:
    """Registra la selección de muestras (lasso en el espacio muestral).

    :param state: estado de la aplicación
    :type state: AppState
    :param indices: índices seleccionados en ``source_obs``
    :type indices: List[int]
    :return: texto a mostrar en el textinput de muestras seleccionadas
    :rtype: str
    """
    state.sel_fil = [state.ids_muestras[i] for i in indices]
    state.sel_fil_idx = indices
    state.texto_dr_cond_fil = ",".join([state.ids_muestras[i] for i in state.sel_fil_idx])
    return ",".join([state.Fy[i] for i in state.sel_fil_idx])


def compute_selection_col(state: AppState, indices: List[int]) -> str:
    """Registra la selección de características (lasso en el espacio de características).

    :param state: estado de la aplicación
    :type state: AppState
    :param indices: índices seleccionados en ``source_var``
    :type indices: List[int]
    :return: texto a mostrar en el textinput de características seleccionadas
    :rtype: str
    """
    state.sel_col = [state.Fx[i] for i in indices]
    state.sel_col_idx = indices
    return ",".join([state.Fx[i] for i in state.sel_col_idx])


def update_conditional_dr(
    state: AppState, texto_var: str, perplexity_1: float, perplexity_2: float
) -> Tuple[str, str]:
    """Aplica la selección de muestras/características actual como condición
    para las proyecciones y recalcula las afinidades.

    :param state: estado de la aplicación
    :type state: AppState
    :param texto_var: texto interpretado con las características seleccionadas
    :type texto_var: str
    :param perplexity_1: perplexity para el espacio muestral
    :type perplexity_1: float
    :param perplexity_2: perplexity para el espacio de características
    :type perplexity_2: float
    :return: textos interpretados (muestras, características) para las textbox
    :rtype: Tuple[str, str]
    """
    state.idx_obs = get_idx_selection(state.texto_dr_cond_fil, state.ids_muestras, no_empty=True)
    state.idx_var = get_idx_selection(texto_var, state.Fx, no_empty=True)

    text_obs = ",".join([state.Fy[i] for i in state.idx_obs])
    text_var = ",".join([state.Fx[i] for i in state.idx_var])

    update_affinities(state, perplexity_1, perplexity_2)

    return text_obs, text_var
