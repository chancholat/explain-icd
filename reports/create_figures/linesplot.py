import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union, Optional, List

ArrayLike = Union[Sequence[float], Sequence[int]]

def draw_lines(
    series: Union[Tuple[ArrayLike, ArrayLike], List[Tuple[ArrayLike, ArrayLike]]],
    *,
    colors: Optional[Union[str, Sequence[str]]] = None,
    labels: Optional[Union[str, Sequence[str]]] = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
    marker: Optional[str] = None,
    alpha: float = 1.0,
    grid: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    # Font sizes
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    # Legend
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_fontsize: int = 10,
    # Figure options
    figsize: Tuple[int, int] = (7, 4),
    tight_layout: bool = True,
    # Save option
    save_path: Optional[str] = None,
    dpi: int = 300,
    linestyle_list: Optional[List[str]] = None,
):
    """
    Plot one or more (x, y) line graphs with customization options.
    Optionally saves the figure to disk.
    """
    # Ensure multiple lines
    if isinstance(series, tuple):
        series_list = [series]
    else:
        series_list = list(series)

    n = len(series_list)

    # Helper to normalize inputs
    def to_list(v, n):
        if v is None:
            return [None] * n
        if isinstance(v, (str, bytes)):
            return [v] * n
        return list(v)

    colors_list = to_list(colors, n)
    labels_list = to_list(labels, n)

    fig, ax = plt.subplots(figsize=figsize)

    if linestyle_list is None:
        linestyle_list = [linestyle] * n

    for (x, y), c, lab, ls in zip(series_list, colors_list, labels_list, linestyle_list):
        ax.plot(
            x, y,
            color=c,
            label=lab,
            linewidth=linewidth,
            linestyle=ls,
            marker=marker,
            alpha=alpha,
        )

    # Axis labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(grid, linestyle="--", alpha=0.6)

    if show_legend and any(l is not None for l in labels_list):
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    if tight_layout:
        plt.tight_layout()

    # Save the image
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Figure saved to: {save_path}")

    return fig, ax


############################################
linestyle_list = None

# Explanation threshold scale 
x = [0.01, 0.02, 0.03, 0.07]
# explanation db scale vs f1_micro, f1_macro
# y1 = [0.591685712, 0.594176352, 0.593621194, 0.589517593]
# y2 = [0.261785448, 0.249844581, 0.24750942, 0.230254486]

# y1 = [round(x * 100, 2) for x in y1]  # Convert to percentage
# y2 = [round(x * 100, 2) for x in y2]

# # unsupervised baseline f1 micro
# y3 = [0.581475794, 0.581475794, 0.581475794, 0.581475794]
# y4 = [0.24423635, 0.24423635, 0.24423635, 0.24423635]

# # unsupervised baseline f1 macro
# y3 = [round(x * 100, 2) for x in y3]  # Convert to percentage
# y4 = [round(x * 100, 2) for x in y4]
# labels = ["F1_micro", "F1_macro", "Unsupervised F1_micro", "Unsupervised F1_macro"]
# linestyle_list = ["solid", "solid", "dashed", "dashed"]


# Precision, recall vs threshold micro
# y1 = [0.608270168, 0.634443045, 0.609475791, 0.610132575]
# y2 = [0.575981557, 0.55871588, 0.578570545, 0.570250154]
# y1 = [round(x * 100, 2) for x in y1]  # Convert to percentage
# y2 = [round(x * 100, 2) for x in y2]
# labels = ["Precision micro", "Recall micro"]

# Precision, recall vs threshold macro
y1 = [0.272392899, 0.272619724, 0.257578313, 0.245888487]
y2 = [0.291879833, 0.263731033, 0.271857142, 0.247253269]
y1 = [round(x * 100, 2) for x in y1]  # Convert to percentage
y2 = [round(x * 100, 2) for x in y2]
labels = ["Precision macro", "Recall macro"]


# # mAP
# y1 = [0.636758864, 0.638372838, 0.637095809, 0.631905794]
# y1 = [round(x * 100, 2) for x in y1]
# # AUC micro 
# y2 = [0.986142709, 0.987243591, 0.987582291, 0.986667621]
# y2 = [round(x * 100, 2) for x in y2]
# labels = ["mAP", "AUC"]


# #############################################


# Window stride effect 
# x = [3, 10, 20]
# # f1_micro
# y1 = [ 0.591974795, 0.594176352, 0.591940403]
# y1 = [round(x * 100, 2) for x in y1]
# # f1_macro
# y2 = [0.243103519, 0.249844581, 0.252251327]
# y2 = [round(x * 100, 2) for x in y2]
# labels = ["F1_micro", "F1_macro"]


# # # Precision 
# y1 = [0.635771096, 0.634443045, 0.634706557]
# y1 = [round(x * 100, 2) for x in y1]
# # Recall
# y2 = [0.55382365, 0.55871588, 0.554573536]
# y2 = [round(x * 100, 2) for x in y2]
# labels = ["Precision", "Recall"]

# # Map
# y1 = [0.637016416, 0.638372838, 0.634840786]
# y1 = [round(x * 100, 2) for x in y1]
# # AUC micro
# y2 = [0.986428651, 0.987243591, 0.986452056]
# y2 = [round(x * 100, 2) for x in y2]
# labels = ["mAP", "AUC"]


draw_lines(
    [(x, y1), (x, y2)],          # ⬅️ multiple (x, y) pairs
    colors=["blue", "red"],
    labels=labels,
    xlabel="Explanation threshold scale",
    ylabel="",
    title="",
    marker=None,
    legend_loc="best",
    title_fontsize=16,
    label_fontsize=13,
    save_path=f"{labels}.png",       # ⬅️ image will be saved here
    figsize=(7, 3),
    linestyle_list=linestyle_list
)
plt.show()
