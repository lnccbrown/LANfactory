from ssms.basic_simulators.simulator import simulator
import numpy as np

# plotting
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import cm

# import pymc as pm
import pandas as pd
import seaborn as sns

try:
    # print('HDDM: Trying import of pytorch related classes.')
    from lanfactory.trainers import LoadTorchMLPInfer
except:
    print(
        "It seems that you do not have pytorch installed."
        + " You cannot use the network_inspector module."
    )

from sklearn.neighbors import KernelDensity
import os

from ssms.config import ModelConfigBuilder
from ssms.support_utils.kde_class import LogKDE

# NETWORK LOADERS -------------------------------------------------------------------------


def get_torch_mlp(model_file_path, network_config, input_dim):
    """Returns the torch network which is the basis of the TORCH_MLP likelihoods

    :Arguments:
        model_file_path: str
        Path to the saved PyTorch model state dict (.pt file).
        network_config: dict or str
        Network configuration dict or path to a config pickle file.
        input_dim: int
        Input dimension of the network (n_params + 2 for rt and choice).

    Returns:
        Returns a function that gives you access to a forward pass through the MLP.
        This in turn expects as input a 2d np.array of datatype np.float32. Each row is filled with
        model parameters trailed by a reaction time and a choice.
        (e.g. input dims for a ddm MLP could be (3, 6), 3 datapoints and 4 parameters + reaction time and choice).
        Predict on batch then returns for each row of the input the log likelihood of the respective parameter vector and datapoint.

    :Example:
        >>> forward = lanfactory.network_inspectors.get_torch_mlp(
        ...     model_file_path='path/to/model.pt',
        ...     network_config='path/to/config.pickle',
        ...     input_dim=6,
        ... )
        >>> data = np.array([[0.5, 1.5, 0.5, 0.5, 1.0, -1.0], [0.5, 1.5, 0.5, 0.5, 1.0, -1.0]], dtype = np.float32)
        >>> forward(data)
    """
    network = LoadTorchMLPInfer(
        model_file_path=model_file_path,
        network_config=network_config,
        input_dim=input_dim,
    )
    return network.predict_on_batch


# PLOTTNG -------------------------------------------------------------------------------------
def kde_vs_lan_likelihoods(  # ax_titles = [],
    parameter_df=None,
    model=None,
    torch_mlp_predict=None,
    n_samples=10,
    n_reps=10,
    alpha=0.1,
    cols=3,
    save=False,
    show=True,
    font_scale=1.5,
    figsize=(10, 10),
):
    """Function creates a plot that compares kernel density estimates from simulation data with mlp output.

    :Arguments:
        parameter_df: pandas.core.frame.DataFrame
            DataFrame hold a parameter vector in each row. (Parameter vector has to be compatible with the model string supplied to the
            'model' argument)
        model: str <default=None>
            String that specifies the model which should be used for the graph (find allowed models listed under ssms.config).
        torch_mlp_predict: callable <default=None>
            A predict_on_batch function, as returned by ``get_torch_mlp``.
        n_samples: int <default=10>
            How many model samples to base kernel density estimates on.
        n_reps: int <default=10>
            How many kernel density estimates to include in a given subplot.
        cols: int <default=3>
            How many columns to use when creating subplots.
        save: bool <default=False>
            Whether to save the plot.
        show: bool <default=True>
            Wheter to show the plot.
        font_scale: float <default=1.5>
            Seaborn setting, exposed here to be adjusted by user, since it is not always
            obvious which value is best.

    :Returns:
        empty
    """

    # Get predictions from simulations /kde

    # mpl.rcParams['text.usetex'] = True
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['svg.fonttype'] = 'none'

    config = ModelConfigBuilder.from_model(model)

    # Initialize rows and graph parameters
    rows = int(np.ceil(parameter_df.shape[0] / cols))
    sns.set(style="white", palette="muted", color_codes=True, font_scale=2)

    fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=False)

    fig.suptitle(
        "Likelihoods KDE vs. LAN" + ": " + model.upper().replace("_", "-"), fontsize=30
    )
    sns.despine(right=True)

    # Data template
    if len(config["choices"]) == 2:
        plot_data = np.zeros((4000, 2))
        plot_data[:, 0] = np.concatenate(
            (
                [i * 0.0025 for i in range(2000, 0, -1)],
                [i * 0.0025 for i in range(1, 2001, 1)],
            )
        )
        plot_data[:, 1] = np.concatenate((np.repeat(-1, 2000), np.repeat(1, 2000)))
    else:
        plot_data = np.zeros((len(config["choices"]) * 1000, 2))
        plot_data[:, 0] = np.concatenate(
            [
                [i * 0.01 for i in range(1, 1001, 1)]
                for j in range(len(config["choices"]))
            ]
        )
        plot_data[:, 1] = np.concatenate(
            [np.repeat(i, 1000) for i in range(len(config["choices"]))]
        )

    # Load Keras model and initialize batch container
    input_batch = np.zeros((4000, parameter_df.shape[1] + 2))
    input_batch[:, parameter_df.shape[1] :] = plot_data

    # n_subplot = 0
    for i in range(parameter_df.shape[0]):
        print(str(i + 1) + " of " + str(parameter_df.shape[0]))

        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)

        # Get predictions from keras model
        input_batch[:, : parameter_df.shape[1]] = parameter_df.iloc[i, :].values
        # input_batch = input_batch.astype(np.float32)
        ll_out_keras = torch_mlp_predict(input_batch.astype(np.float32))

        for j in range(n_reps):
            out = simulator(
                theta=parameter_df.iloc[i, :].values,
                model=model,
                n_samples=n_samples,
                max_t=20,
                delta_t=0.001,
            )

            mykde = LogKDE(out)
            ll_out_gt = mykde.kde_eval(
                {"rts": plot_data[:, 0], "choices": plot_data[:, 1]}
            )

            # Plot kde predictions
            if j == 0:
                label = "KDE"
            else:
                label = None

            if len(config["choices"]) == 2:
                sns.lineplot(
                    x=plot_data[:, 0] * plot_data[:, 1],
                    y=np.exp(ll_out_gt),
                    color="black",
                    alpha=alpha,
                    label=label,
                    ax=ax[row_tmp, col_tmp],
                )

            else:
                for k in range(len(config["choices"])):
                    if k > 0:
                        label = None
                    sns.lineplot(
                        x=plot_data[1000 * k : 1000 * (k + 1), 0],
                        y=np.exp(ll_out_gt[1000 * k : 1000 * (k + 1)]),
                        color="black",
                        alpha=alpha,
                        label=label,
                        ax=ax[row_tmp, col_tmp],
                    )

            if j == 0:
                # Plot keras predictions
                if len(config["choices"]) == 2:
                    sns.lineplot(
                        x=plot_data[:, 0] * plot_data[:, 1],
                        y=np.exp(ll_out_keras[:, 0]),
                        color="green",
                        label="MLP",
                        alpha=1,
                        ax=ax[row_tmp, col_tmp],
                    )
                else:
                    for k in range(len(config["choices"])):
                        if k == 0:
                            label = "MLP"
                        else:
                            label = None

                        sns.lineplot(
                            x=plot_data[1000 * k : 1000 * (k + 1), 0],
                            y=np.exp(ll_out_keras[1000 * k : 1000 * (k + 1), 0]),
                            color="green",
                            label=label,
                            alpha=1,
                            ax=ax[row_tmp, col_tmp],
                        )

        # Legend adjustments
        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(
                loc="upper left", fancybox=True, shadow=True, fontsize=12
            )
        else:
            ax[row_tmp, col_tmp].legend().set_visible(False)

        if row_tmp == rows - 1:
            ax[row_tmp, col_tmp].set_xlabel("rt", fontsize=24)
        else:
            ax[row_tmp, col_tmp].tick_params(color="white")

        if col_tmp == 0:
            ax[row_tmp, col_tmp].set_ylabel("likelihood", fontsize=20)

        # tmp title
        ax[row_tmp, col_tmp].set_title(str(i), fontsize=20)  # ax_titles[i],
        ax[row_tmp, col_tmp].tick_params(axis="y", size=14)
        ax[row_tmp, col_tmp].tick_params(axis="x", size=14)

    for i in range(parameter_df.shape[0], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis("off")

    plt.subplots_adjust(top=0.9)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if save == True:
        if os.path.isdir("figures/"):
            pass
        else:
            os.mkdir("figures/")

        plt.savefig(
            "figures/" + "kde_vs_mlp_plot" + ".png", format="png", transparent=False
        )
        # transparent = True)

    if show:
        plt.show()

    plt.close()

    return


# Predict
def lan_manifold(
    parameter_df=None,
    vary_dict={"v": [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]},
    model="ddm",
    torch_mlp_predict=None,
    n_rt_steps=200,
    max_rt=5,
    fig_scale=1.0,
    save=False,
    show=True,
):
    """Plots lan likelihoods in a 3d-plot.

    :Arguments:
        parameter_df: pandas.core.frame.DataFrame <default=None>
            DataFrame that holds a parameter vector and has parameter names as keys.
        vary_dict: dict <default={'v': [-1.0, -0.75, -.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]}>
            Dictionary where key is a valid parameter name, and value is either a list of numpy.ndarray() of values
            of the respective parameter that you want to plot.
        model: str <default='ddm'>
            String that specifies the model to be used to plotting. (The plot loads the corresponding LAN)
        torch_mlp_predict: callable <default=None>
            A predict_on_batch function, as returned by ``get_torch_mlp``.
        n_rt_steps: int <default=200>
            Numer of rt steps to include (x-axis)
        max_rt: numeric <default=5.0>
            The n_rt_steps argument splits the reaction time axis in to n_rt_step from 0 to max_rt.
        fig_scale: numeric <default=1.0>
            Basic handle to scale the figure.
        save: bool <default=False>
            Whether to save the plot.
        show: bool <default=True>
            Whether to show the plot.

    :Returns:
        empty
    """

    # mpl.rcParams.update(mpl.rcParamsDefault)
    # mpl.rcParams['text.usetex'] = True
    # #matplotlib.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['svg.fonttype'] = 'none'

    config = ModelConfigBuilder.from_model(model)

    assert (
        len(config["choices"]) == 2
    ), "This plot works only for 2-choice models at the moment. Improvements coming!"

    if parameter_df.shape[0] > 0:
        parameters = parameter_df.iloc[0, :]
        print("Using only the first row of the supplied parameter array !")

    if type(parameter_df) == pd.core.frame.DataFrame:
        parameters = np.squeeze(
            parameters[config["params"]].values.astype(np.float32)
        )
    else:
        parameters = parameter_df

    # Prepare data structures

    # Data template
    plot_data = np.zeros((n_rt_steps * 2, 2))
    plot_data[:, 0] = np.concatenate(
        (
            [(i * (max_rt / n_rt_steps)) for i in range(n_rt_steps, 0, -1)],
            [(i * (max_rt / n_rt_steps)) for i in range(1, n_rt_steps + 1, 1)],
        )
    )
    plot_data[:, 1] = np.concatenate(
        (np.repeat(-1, n_rt_steps), np.repeat(1, n_rt_steps))
    )

    n_params = len(config["params"])
    n_levels = vary_dict[list(vary_dict.keys())[0]].shape[0]
    data_var = np.zeros(((n_rt_steps * 2) * n_levels, n_params + 3))

    cnt = 0
    vary_param_name = list(vary_dict.keys())[0]

    for par_tmp in vary_dict[vary_param_name]:
        tmp_begin = (n_rt_steps * 2) * cnt
        tmp_end = (n_rt_steps * 2) * (cnt + 1)
        parameters[config["params"].index(vary_param_name)] = par_tmp

        data_var[tmp_begin:tmp_end, :n_params] = parameters
        data_var[tmp_begin:tmp_end, n_params : (n_params + 2)] = plot_data
        data_var[tmp_begin:tmp_end, (n_params + 2)] = np.squeeze(
            np.exp(torch_mlp_predict(data_var[tmp_begin:tmp_end, :-1].astype(np.float32)))
        )

        cnt += 1

    fig = plt.figure(figsize=(8 * fig_scale, 5.5 * fig_scale))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        data_var[:, -2] * data_var[:, -3],
        data_var[:, config["params"].index(vary_param_name)],
        data_var[:, -1],
        linewidth=0.5,
        alpha=1.0,
        cmap=cm.coolwarm,
    )

    ax.set_ylabel(vary_param_name.upper().replace("_", "-"), fontsize=16, labelpad=20)

    ax.set_xlabel("RT", fontsize=16, labelpad=20)

    ax.set_zlabel("Likelihood", fontsize=16, labelpad=20)

    ax.set_zticks(
        np.round(np.linspace(min(data_var[:, -1]), max(data_var[:, -1]), 5), 1)
    )

    ax.set_yticks(
        np.round(
            np.linspace(
                min(data_var[:, config["params"].index(vary_param_name)]),
                max(data_var[:, config["params"].index(vary_param_name)]),
                5,
            ),
            1,
        )
    )

    ax.set_xticks(
        np.round(
            np.linspace(
                min(data_var[:, -2] * data_var[:, -3]),
                max(data_var[:, -2] * data_var[:, -3]),
                5,
            ),
            1,
        )
    )

    ax.tick_params(labelsize=16)
    ax.set_title(
        model.upper().replace("_", "-") + " - MLP: Manifold", fontsize=20, pad=20
    )

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Save plot
    if save:
        if os.path.isdir("figures/"):
            pass
        else:
            os.mkdir("figures/")

        plt.savefig("figures/mlp_manifold_" + model + ".png", format="png")

    if show:
        return plt.show()

    plt.close()
    return
