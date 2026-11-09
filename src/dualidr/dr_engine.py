"""Motor de reducción dimensional: afinidades, t-SNE condicional y reinicio a PCA."""

import numpy as np
from openTSNE import TSNEEmbedding, affinity

from dualidr.state import AppState


def update_affinities(state: AppState, perplexity_1: float, perplexity_2: float) -> None:
    """Recalcula las afinidades y reinicia los objetos de embedding t-SNE
    condicionados a la selección actual de features (idx_var) y muestras (idx_obs).

    :param state: estado de la aplicación
    :type state: AppState
    :param perplexity_1: perplexity para el espacio muestral
    :type perplexity_1: float
    :param perplexity_2: perplexity para el espacio de características
    :type perplexity_2: float
    :return: None
    """
    print("computing affinities (samples, features) ...")
    state.affinities_1 = affinity.PerplexityBasedNN(
        state.x1[:, state.idx_var],
        perplexity=perplexity_1,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    state.affinities_2 = affinity.PerplexityBasedNN(
        state.x2[:, state.idx_obs],
        perplexity=perplexity_2,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    print("creating embedding objects (samples, features) ...")
    state.embedding_1 = TSNEEmbedding(
        state.y1,
        state.affinities_1,
        negative_gradient_method="bh",
        n_jobs=8,
        verbose=False,
    )

    state.embedding_2 = TSNEEmbedding(
        state.y2,
        state.affinities_2,
        negative_gradient_method="bh",
        n_jobs=8,
        verbose=False,
    )
    print("... done")


def update_tsne(state: AppState, lr_1: float, lr_2: float) -> None:
    """Ejecuta un paso de optimización t-SNE para ambos embeddings y
    actualiza los ColumnDataSource de Bokeh.

    Notas:
      - la actualización toma pocas épocas por llamada
      - ``state.early_exaggeration`` por defecto es 1 (t-SNE básico sin distorsión)

    :param state: estado de la aplicación
    :type state: AppState
    :param lr_1: log-learning-rate para el espacio muestral
    :type lr_1: float
    :param lr_2: log-learning-rate para el espacio de características
    :type lr_2: float
    :return: None
    """
    state.embedding_1 = state.embedding_1.optimize(
        n_iter=1,
        exaggeration=state.early_exaggeration,
        momentum=0.5,
        learning_rate=np.exp(lr_1),
        verbose=False,
    )
    state.embedding_2 = state.embedding_2.optimize(
        n_iter=1,
        exaggeration=state.early_exaggeration,
        momentum=0.5,
        learning_rate=np.exp(lr_2),
        verbose=False,
    )

    state.y1 = np.array(state.embedding_1)
    state.y2 = np.array(state.embedding_2)

    state.source_obs.data["x1"] = state.y1[:, 0]
    state.source_obs.data["y1"] = state.y1[:, 1]
    state.source_var.data["x2"] = state.y2[:, 0]
    state.source_var.data["y2"] = state.y2[:, 1]


def reset_original_state(state: AppState) -> None:
    """Reentrena las proyecciones iniciales PCA para muestras y características.

    :param state: estado de la aplicación
    :type state: AppState
    :return: None
    """
    print("training PCA (samples and features)...")
    state.y1 = state.pca.fit_transform(state.X)
    state.y2 = state.pca.fit_transform(state.X.T)
