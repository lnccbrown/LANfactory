# `lanfactory.network_inspectors`

The public inspection namespace loads a trained Torch LAN, compares its
likelihood with simulation-based KDE estimates, and plots likelihood manifolds.
The configuration dataclasses keep model metadata, evaluation grids, and plot
defaults explicit.

::: lanfactory.network_inspectors.get_torch_mlp

::: lanfactory.network_inspectors.kde_vs_lan_likelihoods

::: lanfactory.network_inspectors.lan_manifold

::: lanfactory.network_inspectors.ModelSpec

::: lanfactory.network_inspectors.PlotConfig

::: lanfactory.network_inspectors.GridSpec

## Styling the KDE-vs-LAN figure

`kde_vs_lan_likelihoods` returns the Matplotlib figure, and `PlotConfig` sets
its titles, font sizes, layout and output file. The defaults draw the same
figure as before. For a compact figure that goes into a paper:

```python
from lanfactory.network_inspectors import PlotConfig, kde_vs_lan_likelihoods

fig = kde_vs_lan_likelihoods(
    parameter_df=parameter_df,  # two rows that differ in v
    model="angle",
    torch_mlp_predict=predictor,
    plot=PlotConfig(
        cols=2,
        figsize=(10, 4),
        layout="constrained",  # fit titles and labels into this size
        title=None,  # the caption names the figure
        panel_titles=["v = 0", "v = 1.5"],  # or "params" for each row's values
        panel_title_fontsize=14,
        xlabel_fontsize=14,
        ylabel_fontsize=14,
        font_scale=1.0,  # tick labels
        show=False,  # return the figure open
        save=True,
        save_dir="figures",
        filename="network_check.png",  # a .pdf name saves a PDF
        dpi=300,
    ),
)
```

With `show=False` the figure stays open, so Matplotlib's own tools still apply
to it (`fig.axes`, `fig.savefig`); close it with `plt.close(fig)` when you are
done. With `show=True`, the default, it is displayed and closed, and still
returned. In a notebook, assign the result (`fig = ...`) or the figure appears a
second time as the cell's value.

The function styles its own figure only: Matplotlib's global style is the same
after the call as before it.
