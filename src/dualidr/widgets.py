"""Construcción de los widgets Bokeh (sliders, botones, textinputs, checkbox)."""

from types import SimpleNamespace

from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Slider,
    TextAreaInput,
    TextInput,
)
from bokeh.models.callbacks import CustomJS

_DOWNLOAD_CSV_JS = """
    const data = source.data;
    const keys = Object.keys(data);
    const nrows = data[keys[0]].length;

    let csv = keys.join(",") + "\\n";

    for (let i = 0; i < nrows; i++) {
        const row = keys.map(k => data[k][i]);
        csv += row.join(",") + "\\n";
    }

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = "source_var.csv";
    a.click();
    URL.revokeObjectURL(url);
"""


def build_widgets(source_var: ColumnDataSource) -> SimpleNamespace:
    """Construye todos los widgets de control de la aplicación.

    Los callbacks de negocio (``on_change``/``on_click``) se conectan fuera
    de esta función, en el script de ensamblaje, salvo el de descarga de CSV
    que es autocontenido (JavaScript puro sobre ``source_var``).

    :param source_var: ColumnDataSource de características, usado por el botón de descarga
    :type source_var: ColumnDataSource
    :return: espacio de nombres con todos los widgets construidos
    :rtype: SimpleNamespace
    """
    slider_perplexity_1 = Slider(start=2, end=200, value=130, step=1, title="perplexity")
    slider_perplexity_2 = Slider(start=2, end=200, value=33, step=1, title="perplexity")
    slider_lr_1 = Slider(start=-5, end=5, value=2, step=0.1, title="learning_rate (log)")
    slider_lr_2 = Slider(start=-5, end=5, value=2, step=0.1, title="learning_rate (log)")
    slider_l1 = Slider(start=0, end=1, value=1, step=0.05, title="L1 ratio")

    f1_textbox = TextInput(width=300, disabled=True, value="F1 Score : 0.000", align="center")

    button_conditional_dr = Button(label="use selected features & samples")
    button_obs = Button(label="selection (sample) ➡ color")
    button_var = Button(label="selection (feature) ➡ color")
    button_reset = Button(label="reset projections")
    button_exaggeration = Button(label="exaggeration = OFF")
    button_feature_importance = Button(label="Discriminate")

    textinput_colorsel_obs = TextAreaInput(
        value="", title="selected samples", width=500, height=145, max_length=10000
    )
    textinput_colorsel_var = TextAreaInput(
        value="", title="selected features", width=500, height=145, max_length=10000
    )

    textinput_color_obs = TextInput(title="color for samples", width=145, value="black")
    textinput_color_var = TextInput(title="color for features", width=145, value="black")

    checkbox_group = CheckboxGroup(labels=["sample labels", "feature labels"], active=[0, 1])

    button_download = Button(label="Download CSV", button_type="success")
    button_download.js_on_click(
        CustomJS(args=dict(source=source_var), code=_DOWNLOAD_CSV_JS)
    )

    return SimpleNamespace(
        slider_perplexity_1=slider_perplexity_1,
        slider_perplexity_2=slider_perplexity_2,
        slider_lr_1=slider_lr_1,
        slider_lr_2=slider_lr_2,
        slider_l1=slider_l1,
        f1_textbox=f1_textbox,
        button_conditional_dr=button_conditional_dr,
        button_obs=button_obs,
        button_var=button_var,
        button_reset=button_reset,
        button_exaggeration=button_exaggeration,
        button_feature_importance=button_feature_importance,
        textinput_colorsel_obs=textinput_colorsel_obs,
        textinput_colorsel_var=textinput_colorsel_var,
        textinput_color_obs=textinput_color_obs,
        textinput_color_var=textinput_color_var,
        checkbox_group=checkbox_group,
        button_download=button_download,
    )
