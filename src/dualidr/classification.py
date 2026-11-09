"""Pipeline de clasificación/validación: hold-out temporal sin fuga de información.

Dado un etiquetado de ventanas (seleccionadas = 1), se corta cada registro en el
tiempo en un tramo de entrenamiento y otro de test, se excluyen las ventanas que
cruzan el corte (más una guarda adicional) y se ajustan **solo con entrenamiento**
el normalizador de señales, el de características y el clasificador.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from dualidr.data import (
    RawRecords,
    WindowMeta,
    apply_signal_scaler,
    extract_features,
    fit_signal_scaler,
)


@dataclass
class ClassificationParams:
    """Parámetros del pipeline.

    :param l1_ratio: ratio L1 del término elasticnet
    :param train_fraction: fracción objetivo de cada registro dedicada a entrenamiento
    :param gap_samples: separación mínima (en muestras) entre el corte y la primera ventana de test
    :param min_per_class: ventanas mínimas de cada clase en entrenamiento y en test
    :param C: inversa de la regularización de la regresión logística
    """

    l1_ratio: float = 1.0
    train_fraction: float = 0.7
    gap_samples: int = 400
    min_per_class: int = 5
    C: float = 0.5


@dataclass
class ClassificationResult:
    """Resultado del pipeline: métrica de test y coeficientes del clasificador.

    ``coef`` sigue el orden de las columnas de la matriz de características.
    """

    ok: bool
    reason: str = ""
    f1: float = 0.0
    coef: Optional[np.ndarray] = None
    n_train: int = 0
    n_test: int = 0
    n_excluded: int = 0
    cuts: Dict[int, int] = field(default_factory=dict)


def choose_cut(
    starts: np.ndarray, ends: np.ndarray, y: np.ndarray, length: int, params: ClassificationParams
) -> Optional[int]:
    """Elige el corte temporal de un registro.

    Los candidatos son los finales de ventana. Se prefiere el más cercano a
    ``train_fraction``; si el registro contiene ambas clases, se priorizan los
    cortes que dejan ambas clases a los dos lados.

    :param starts: inicios de las ventanas del registro
    :type starts: np.ndarray
    :param ends: finales (exclusivos) de las ventanas del registro
    :type ends: np.ndarray
    :param y: etiquetas de las ventanas del registro
    :type y: np.ndarray
    :param length: muestras del registro
    :type length: int
    :param params: parámetros del pipeline
    :type params: ClassificationParams
    :return: muestra de corte, o None si no hay ninguno con ventanas a ambos lados
    :rtype: Optional[int]
    """
    mixed = len(np.unique(y)) > 1
    scored = []
    for cut in np.unique(ends):
        train, test = ends <= cut, starts >= cut + params.gap_samples
        if not train.any() or not test.any():
            continue
        balanced = not mixed or (len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2)
        scored.append((not balanced, abs(cut / length - params.train_fraction), int(cut)))
    return min(scored)[2] if scored else None


def split_windows(
    meta: WindowMeta, cuts: Dict[int, int], gap_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Máscaras de entrenamiento y test; las ventanas que cruzan el corte o caen en la guarda quedan fuera.

    :param meta: metadatos de todas las ventanas
    :type meta: WindowMeta
    :param cuts: corte de cada registro (``record_id`` -> muestra)
    :type cuts: Dict[int, int]
    :param gap_samples: guarda adicional tras el corte
    :type gap_samples: int
    :return: máscaras booleanas (train, test)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    cut = np.array([cuts.get(int(r), -(10**9)) for r in meta.record_id])
    has_cut = np.array([int(r) in cuts for r in meta.record_id])
    return has_cut & (meta.end <= cut), has_cut & (meta.start >= cut + gap_samples)


def _class_counts_ok(y: np.ndarray, min_per_class: int) -> bool:
    return int(np.sum(y == 1)) >= min_per_class and int(np.sum(y == 0)) >= min_per_class


def run_classification_pipeline(
    raw: RawRecords,
    meta: WindowMeta,
    y: np.ndarray,
    params: ClassificationParams,
    feature_idx: Optional[np.ndarray] = None,
) -> ClassificationResult:
    """Ejecuta el hold-out temporal y devuelve el F1 de test y los coeficientes.

    :param raw: registros originales (sin normalizar)
    :type raw: RawRecords
    :param meta: metadatos de todas las ventanas
    :type meta: WindowMeta
    :param y: etiqueta binaria de cada ventana (1 = seleccionada)
    :type y: np.ndarray
    :param params: parámetros del pipeline
    :type params: ClassificationParams
    :param feature_idx: columnas de características con las que se entrena el clasificador;
        None usa todas. Los coeficientes de las demás se devuelven como 0
    :type feature_idx: Optional[np.ndarray]
    :return: resultado; ``ok=False`` (sin entrenar) si falta representación de alguna clase
    :rtype: ClassificationResult
    """
    if feature_idx is not None and len(feature_idx) == 0:
        return ClassificationResult(ok=False, reason="no features selected")
    if len(np.unique(y)) < 2:
        return ClassificationResult(ok=False, reason="selection has a single class")

    cuts: Dict[int, int] = {}
    for record in np.unique(meta.record_id):
        rows = meta.record_id == record
        cut = choose_cut(
            meta.start[rows], meta.end[rows], y[rows], raw.signals[int(record)].shape[0], params
        )
        if cut is not None:
            cuts[int(record)] = cut

    train_mask, test_mask = split_windows(meta, cuts, params.gap_samples)
    n_excluded = len(meta) - int(train_mask.sum()) - int(test_mask.sum())
    if not _class_counts_ok(y[train_mask], params.min_per_class):
        return ClassificationResult(ok=False, reason="train lacks a class", n_excluded=n_excluded, cuts=cuts)
    if not _class_counts_ok(y[test_mask], params.min_per_class):
        return ClassificationResult(ok=False, reason="test lacks a class", n_excluded=n_excluded, cuts=cuts)

    signal_scaler = fit_signal_scaler([raw.signals[r][:cut] for r, cut in cuts.items()])
    signals = apply_signal_scaler(signal_scaler, raw.signals)
    x_train = extract_features(signals, meta.subset(train_mask), fs=raw.fs)
    x_test = extract_features(signals, meta.subset(test_mask), fs=raw.fs)

    n_features = x_train.shape[1]
    columns = np.arange(n_features) if feature_idx is None else np.asarray(feature_idx, dtype=int)
    x_train, x_test = x_train[:, columns], x_test[:, columns]

    feature_scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        solver="saga", penalty="elasticnet", l1_ratio=params.l1_ratio, C=params.C, random_state=0
    ).fit(feature_scaler.transform(x_train), y[train_mask])
    y_pred = model.predict(feature_scaler.transform(x_test))

    coef = np.zeros(n_features)
    coef[columns] = model.coef_.flatten()
    return ClassificationResult(
        ok=True,
        f1=float(f1_score(y[test_mask], y_pred, average="binary", zero_division=0)),
        coef=coef,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        n_excluded=n_excluded,
        cuts=cuts,
    )
