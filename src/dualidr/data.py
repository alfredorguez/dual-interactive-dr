"""Carga de registros, ventanas con trazabilidad y extracción de características espectrales."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 400  # muestras por ventana
WINDOW_STRIDE = 160  # desplazamiento entre ventanas
SAMPLING_RATE = 5000.0  # Hz
BANDS = np.arange(25, 525, 25)  # Hz, centros de las bandas de frecuencia
BAND_HALF_WIDTH = 5.0  # Hz


@dataclass
class RawRecords:
    """Señales originales de los ensayos.

    :param signals: una matriz (muestras x canales) por registro, en el orden del fichero
    :param record_names: nombre del experimento de cada registro
    :param channel_names: nombre de cada canal
    :param fs: frecuencia de muestreo en Hz
    """

    signals: List[np.ndarray]
    record_names: List[str]
    channel_names: List[str]
    fs: float = SAMPLING_RATE


@dataclass
class WindowMeta:
    """Trazabilidad de las ventanas: tupla ``(window_id, record_id, start, end)``.

    ``record_id`` indexa ``RawRecords.signals``; ``end`` es exclusivo
    (``end = start + window_size``). Todos los campos tienen la misma longitud.
    """

    window_id: np.ndarray
    record_id: np.ndarray
    start: np.ndarray
    end: np.ndarray

    def __len__(self) -> int:
        return len(self.window_id)

    def subset(self, mask: np.ndarray) -> "WindowMeta":
        """Devuelve las ventanas indicadas por ``mask`` conservando sus ``window_id``.

        :param mask: máscara booleana o índices de las ventanas a conservar
        :type mask: np.ndarray
        :return: metadatos del subconjunto
        :rtype: WindowMeta
        """
        return WindowMeta(self.window_id[mask], self.record_id[mask], self.start[mask], self.end[mask])


def load_raw_records(datapath: str = "data/dataicann.mat") -> RawRecords:
    """Lee el fichero ``.mat`` con los registros de los ensayos.

    :param datapath: ruta al fichero de datos
    :type datapath: str
    :return: señales, nombres de experimento y de canal
    :rtype: RawRecords
    """
    mat = loadmat(datapath)
    signals = [np.asarray(z, dtype=float) for z in mat["z"][0]]
    record_names = [str(name[0]).split(".")[0] for name in mat["nom"][0]]
    channel_names = [str(name[0]) for name in mat["NombresVariables"][0]]
    return RawRecords(signals=signals, record_names=record_names, channel_names=channel_names)


def make_windows(
    raw: RawRecords, window_size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE
) -> WindowMeta:
    """Genera la rejilla global de ventanas (``start = k * stride``) de todos los registros.

    Los registros se recorren por orden alfabético de experimento, y el último
    tramo incompleto se descarta, igual que en el cálculo original, de modo que
    ``window_id`` coincide con la fila de la matriz de características.

    :param raw: registros originales
    :type raw: RawRecords
    :param window_size: muestras por ventana
    :type window_size: int
    :param stride: desplazamiento entre ventanas
    :type stride: int
    :return: metadatos de todas las ventanas
    :rtype: WindowMeta
    """
    order = sorted(range(len(raw.signals)), key=lambda r: raw.record_names[r])
    starts = [np.arange(0, raw.signals[r].shape[0] - window_size, stride) for r in order]
    record_id = np.concatenate([np.full(len(s), r) for r, s in zip(order, starts)])
    start = np.concatenate(starts)
    return WindowMeta(np.arange(len(start)), record_id, start, start + window_size)


def fit_signal_scaler(segments: Sequence[np.ndarray]) -> StandardScaler:
    """Ajusta el normalizador por canal con los tramos de señal indicados.

    :param segments: tramos (muestras x canales) con los que se ajusta
    :type segments: Sequence[np.ndarray]
    :return: normalizador ajustado
    :rtype: StandardScaler
    """
    return StandardScaler().fit(np.vstack(segments))


def apply_signal_scaler(scaler: StandardScaler, signals: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Aplica el normalizador de señales a cada registro.

    :param scaler: normalizador ajustado
    :type scaler: StandardScaler
    :param signals: registros (muestras x canales)
    :type signals: Sequence[np.ndarray]
    :return: registros normalizados
    :rtype: List[np.ndarray]
    """
    return [scaler.transform(s) for s in signals]


def extract_features(
    signals: Sequence[np.ndarray],
    meta: WindowMeta,
    window_size: int = WINDOW_SIZE,
    fs: float = SAMPLING_RATE,
    bands: np.ndarray = BANDS,
    half_width: float = BAND_HALF_WIDTH,
) -> np.ndarray:
    """Ventana de Hamming + |FFT| + valor RMS en cada banda de frecuencia.

    La columna ``i * n_channels + j`` corresponde a la banda ``i`` del canal ``j``.

    :param signals: registros normalizados, indexados por ``meta.record_id``
    :type signals: Sequence[np.ndarray]
    :param meta: ventanas a procesar
    :type meta: WindowMeta
    :param window_size: muestras por ventana
    :type window_size: int
    :param fs: frecuencia de muestreo en Hz
    :type fs: float
    :param bands: centros de banda en Hz
    :type bands: np.ndarray
    :param half_width: semiancho de banda en Hz
    :type half_width: float
    :return: matriz (ventanas x bandas*canales), sin normalizar
    :rtype: np.ndarray
    """
    n_channels = signals[0].shape[1]
    freqs = np.arange(window_size) * (fs / window_size)
    masks = [(freqs >= b - half_width) & (freqs < b + half_width) for b in bands]
    taper = np.hamming(window_size)[None, :, None]
    offsets = np.arange(window_size)[None, :]

    features = np.zeros((len(meta), len(bands) * n_channels))
    for record in np.unique(meta.record_id):
        rows = np.flatnonzero(meta.record_id == record)
        frames = signals[record][meta.start[rows][:, None] + offsets] * taper  # (w, N, ch)
        spectrum = np.abs(np.fft.fft(frames, axis=1))
        rms = [np.sqrt(np.sum(spectrum[:, m, :] ** 2, axis=1) / m.sum()) for m in masks]
        features[rows] = np.stack(rms, axis=1).reshape(len(rows), -1)
    return features


def feature_names(channel_names: Sequence[str], bands: np.ndarray = BANDS) -> List[str]:
    """Nombres de las características en el orden de las columnas de ``extract_features``.

    :param channel_names: nombres de canal
    :type channel_names: Sequence[str]
    :param bands: centros de banda en Hz
    :type bands: np.ndarray
    :return: lista ``canal@banda Hz``
    :rtype: List[str]
    """
    return [f"{ch}@{int(b)}Hz" for b in bands for ch in channel_names]


def get_data_with_meta(
    datapath: str = "data/dataicann.mat",
) -> Tuple[np.ndarray, List[str], List[str], RawRecords, WindowMeta]:
    """Lee los registros, los enventana y calcula la matriz de características.

    Las señales se normalizan con todos los datos, se extraen las características
    (Hamming + FFT, RMS en bandas de 25 a 500 Hz) y la matriz resultante se normaliza.
    Cada experimento se enventana por separado.

    :param datapath: ruta al fichero de datos
    :type datapath: str
    :return: matriz de características, etiqueta (experimento) y nombres de
        características, registros originales y metadatos ``(window_id, record_id, start, end)``
    :rtype: Tuple[np.ndarray, List[str], List[str], RawRecords, WindowMeta]
    """
    raw = load_raw_records(datapath)
    meta = make_windows(raw)

    signal_scaler = fit_signal_scaler(raw.signals)
    features = extract_features(apply_signal_scaler(signal_scaler, raw.signals), meta)
    features_norm = StandardScaler().fit_transform(features)

    labels = [raw.record_names[r] for r in meta.record_id]
    return features_norm, labels, feature_names(raw.channel_names), raw, meta
