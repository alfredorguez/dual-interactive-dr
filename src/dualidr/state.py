"""Contenedor del estado mutable de la app Dual DiDR."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from bokeh.models import ColumnDataSource
from sklearn.linear_model import LogisticRegression

from dualidr.classification import ClassificationResult
from dualidr.data import RawRecords, WindowMeta


@dataclass
class AppState:
    """Estado mutable compartido por el motor de reducción dimensional,
    la selección de muestras/características y los modelos de importancia.

    :param X: matriz de características (muestras x features), ya normalizada
    :type X: np.ndarray
    :param Fx: nombres de las características
    :type Fx: List[str]
    :param Fy: etiquetas de las muestras
    :type Fy: List[str]
    :param ids_muestras: identificadores de muestras (como strings)
    :type ids_muestras: List[str]
    """

    X: np.ndarray
    Fx: List[str]
    Fy: List[str]
    ids_muestras: List[str]

    source_obs: ColumnDataSource
    source_var: ColumnDataSource

    x1: np.ndarray
    x2: np.ndarray
    y1: np.ndarray
    y2: np.ndarray

    pca: Any

    N: int  # número de muestras
    M: int  # número de características

    idx_obs: np.ndarray = field(default_factory=lambda: np.array([]))
    idx_var: np.ndarray = field(default_factory=lambda: np.array([]))

    sel_fil: List[str] = field(default_factory=list)
    sel_fil_idx: List[int] = field(default_factory=list)
    sel_col: List[str] = field(default_factory=list)
    sel_col_idx: List[int] = field(default_factory=list)
    texto_dr_cond_fil: str = ""

    affinities_1: Any = None
    affinities_2: Any = None
    embedding_1: Any = None
    embedding_2: Any = None
    early_exaggeration: int = 1

    log_obs: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(
            n_jobs=-1, solver="saga", penalty="elasticnet", l1_ratio=1, C=0.5
        )
    )
    log_var: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(
            n_jobs=-1, solver="saga", penalty="elasticnet", l1_ratio=1, C=0.5
        )
    )
    f1: float = 0.0
    importances_var: Optional[np.ndarray] = None

    raw: Optional[RawRecords] = None
    window_meta: Optional[WindowMeta] = None
    classification: Optional[ClassificationResult] = None
