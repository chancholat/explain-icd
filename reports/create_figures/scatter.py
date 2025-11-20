import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union, Optional, List

ArrayLike = Union[Sequence[float], Sequence[int]]

def draw_points(
    series: Union[Tuple[ArrayLike, ArrayLike], List[Tuple[ArrayLike, ArrayLike]]],
    *,
    colors: Optional[Union[str, Sequence[str]]] = None,
    labels: Optional[Union[str, Sequence[str]]] = None,
    markers: Optional[Union[str, Sequence[str]]] = "o",
    sizes: Union[float, Sequence[float]] = 60,
    alpha: float = 0.9,
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
):
    """
    Draw a scatter/point plot for one or multiple (x, y) series.
    """
    # Ensure list of tuples
    if isinstance(series, tuple):
        series_list = [series]
    else:
        series_list = list(series)

    n = len(series_list)

    def to_list(v, n):
        if v is None:
            return [None] * n
        if isinstance(v, (str, bytes)):
            return [v] * n
        return list(v)

    colors_list = to_list(colors, n)
    labels_list = to_list(labels, n)
    markers_list = to_list(markers, n)
    if isinstance(sizes, (int, float)):
        sizes_list = [sizes] * n
    else:
        sizes_list = list(sizes)

    fig, ax = plt.subplots(figsize=figsize)

    for (x, y), c, lab, m, s in zip(series_list, colors_list, labels_list, markers_list, sizes_list):
        ax.scatter(
            x, y,
            color=c,
            label=lab,
            s=s,
            marker=m,
            alpha=alpha,
        )

    # Labels and style
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

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Point plot saved to: {save_path}")

    return fig, ax

# supervised
x1 = [0.4256047955420451 * 100]
y1 = [0.591685712 * 100]

# un supervised
x2 = [0.38990598713008573 * 100]
y2 = [0.594557285 * 100]    

# PGD
x3 = [0.3849977406186013 * 100]
y3 = [0.59192729 * 100]


draw_points(
    [(x1, y1), (x2, y2), (x3, y3)],
    colors=["blue", "red", "green"],
    labels=["Supervised", "Unsupervised", "PGD"],
    markers=["o", "o", "o"],
    sizes=80,
    xlabel="Explanation F1",
    ylabel="Prediction F1 micro",
    title="",
    figsize=(7, 3),
    save_path="ExpVsPred.png",
)
plt.show()