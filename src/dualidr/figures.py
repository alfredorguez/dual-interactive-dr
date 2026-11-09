"""Construcción de las figuras Bokeh: espacio muestral, de características e importancia."""

from typing import Tuple

from bokeh.models import BoxSelectTool, ColumnDataSource, LabelSet
from bokeh.plotting import figure


def build_sample_figure(source_obs: ColumnDataSource) -> Tuple[figure, LabelSet]:
    """Construye la figura del espacio muestral ("sample view").

    :param source_obs: ColumnDataSource con las proyecciones de muestras
    :type source_obs: ColumnDataSource
    :return: figura y su LabelSet de etiquetas
    :rtype: Tuple[figure, LabelSet]
    """
    fig1 = figure(
        width=700,
        height=700,
        title="Sample space",
        tools="crosshair,lasso_select,pan,reset,wheel_zoom",
        tooltips=[("sample:", "@labels")],
        match_aspect=True,
        output_backend="webgl",
    )
    fig1.circle(x="x1", y="y1", color="colors", source=source_obs, size="size")
    fig1.title.text_font_size = "16px"

    labels_obs = LabelSet(x="x1", y="y1", text="labels", source=source_obs, y_offset=5)
    labels_obs.text_alpha = 1
    labels_obs.text_align = "center"
    labels_obs.text_font_size = {"value": "10px"}
    fig1.add_layout(labels_obs)

    return fig1, labels_obs


def build_feature_figure(source_var: ColumnDataSource) -> Tuple[figure, LabelSet]:
    """Construye la figura del espacio de características ("feature view").

    :param source_var: ColumnDataSource con las proyecciones de características
    :type source_var: ColumnDataSource
    :return: figura y su LabelSet de etiquetas
    :rtype: Tuple[figure, LabelSet]
    """
    fig2 = figure(
        width=700,
        height=700,
        title="Feature space",
        tools="crosshair,lasso_select,pan,reset,wheel_zoom",
        tooltips=[("feature:", "@labels")],
        match_aspect=True,
        output_backend="webgl",
    )
    fig2.circle(x="x2", y="y2", color="colors", source=source_var, size="size")
    fig2.title.text_font_size = "16px"

    labels_var = LabelSet(x="x2", y="y2", text="labels", source=source_var)
    labels_var.text_alpha = 1
    labels_var.text_align = "center"
    labels_var.text_font_size = {"value": "10px"}
    fig2.add_layout(labels_var)

    return fig2, labels_var


def build_importance_figure(source_var: ColumnDataSource) -> figure:
    """Construye el gráfico de barras de importancia de características, con
    resaltado cruzado hacia ``source_var`` de las barras seleccionadas arrastrando
    en vertical (franja de ancho completo).

    :param source_var: ColumnDataSource con las características (labels, colors)
    :type source_var: ColumnDataSource
    :return: figura del gráfico de importancia
    :rtype: figure
    """
    labels = source_var.data["labels"]

    fig3 = figure(
        y_range=labels,
        plot_height=max(400, 11 * len(labels)),
        plot_width=500,
        title="Discriminant Weights",
        tools="crosshair,pan,reset,wheel_zoom",
        tooltips=[("feature:", "@feature_label"), ("importance:", "@importance")],
    )

    # Selección por franjas: al arrastrar en vertical el recuadro ocupa todo el ancho
    band_select = BoxSelectTool(dimensions="height")
    fig3.add_tools(band_select)
    fig3.toolbar.active_drag = band_select

    fig3.hbar(
        y="feature_label",
        right="importance",
        height=0.8,
        color="color",
        source=ColumnDataSource(
            data={
                "feature_label": labels,
                "importance": source_var.data["importance"],
                "color": source_var.data["colors"],
            }
        ),
    )

    fig3.ygrid.grid_line_color = None
    fig3.xaxis.axis_label = "Importance"
    fig3.yaxis.group_label_orientation = "vertical"
    fig3.yaxis.major_label_text_font_size = "8pt"
    fig3.x_range.start = 0

    def select_callback(attr, old, new):
        # Las barras están reordenadas respecto a source_var: se cruza por etiqueta
        bar_labels = fig3.renderers[0].data_source.data["feature_label"]
        selected_labels = {bar_labels[i] for i in new}
        source_var.selected.indices = [
            i for i, label in enumerate(source_var.data["labels"]) if label in selected_labels
        ]

    fig3.renderers[0].data_source.selected.on_change("indices", select_callback)

    return fig3
