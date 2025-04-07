from collections.abc import Iterable
import drjit as dr
import mitsuba as mi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional

FIGURE_LINEWIDTH        = 237.13594/71.959
FIGURE_WIDTH_ONE_COLUMN = 2*FIGURE_LINEWIDTH
FIGURE_WIDTH_TWO_COLUMN = FIGURE_LINEWIDTH

def disable_ticks(ax):
    """ Disable ticks around plot (useful for displaying images) """
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    return ax

def set_siggraph_font():
    font = {'family': 'sans-serif',
        'sans-serif': 'Linux Biolinum'}
    mpl.rc('font', **font)
    mpl.rc('text', **{'usetex': False})
    mpl.rc('mathtext', fontset='custom', rm='Linux Biolinum', it='Linux Biolinum:italic', bf='Linux Biolinum:bold')

def generate_figure(integrators: List[str], data: dict, output_path: Path, grad_projection: str='red', square_r_setting_3: bool = True, quantile: float = 0.89, labels: Optional[List[str]] = None):
    grad_projection_fn = None
    if grad_projection == 'red':
        grad_projection_fn = lambda grad: grad[...,0]
    elif grad_projection == 'mean':
        grad_projection_fn = lambda grad: dr.mean(grad, axis=-1)
    else:
        raise RuntimeError(f"Unknown gradient projection {grad_projection}")

    num_settings = len(list(data.keys()))

    n_rows = num_settings
    n_cols = 2 + len(integrators)

    aspect = (n_rows / n_cols)
    fig = plt.figure(1, figsize=(FIGURE_WIDTH_ONE_COLUMN, aspect * FIGURE_WIDTH_ONE_COLUMN), constrained_layout=False)
    gs  = fig.add_gridspec(n_rows, n_cols, wspace=0.05, hspace=0.05)
    r = None
    for i, setting in enumerate(data.keys()):
        setting_data = data[setting]
        q = quantile[i] if isinstance(quantile, Iterable) else quantile
        for j, integrator in enumerate(integrators):
            assert setting_data[j][0] == integrator
            if j == 0:
                # Show primal image
                img = setting_data[j][1]
                ax_img = disable_ticks(fig.add_subplot(gs[i, j]))
                ax_img.imshow(mi.Bitmap(img).convert(srgb_gamma=True))
                
                # Show FD of reference primal image
                grad_fd = grad_projection_fn(setting_data[j][2])
                ax_fd = disable_ticks(fig.add_subplot(gs[i, j + 1]))
                # init range
                r = np.quantile(np.abs(grad_fd), q)
                # last setting gets a different range
                if square_r_setting_3 and (i == 2):
                    r = np.quantile(np.abs(grad_fd), q)*2
                #r = np.maximum(r, 1)
                ax_fd.imshow(grad_fd, cmap='coolwarm', vmin=-r, vmax=r)

                if i == num_settings - 1:
                    ax_img.set_xlabel("Image", fontsize=12)
                    ax_fd.set_xlabel("FD", fontsize=12)
                
                ax_img.set_ylabel(setting, fontsize=12)

            # Show forward-mode gradient
            grad_fw = grad_projection_fn(setting_data[j][3])
            ax_fw = disable_ticks(fig.add_subplot(gs[i, j + 2]))      
            ax_fw.imshow(grad_fw, cmap='coolwarm', vmin=-r, vmax=r)

            if i == num_settings - 1:
                integrator_label = labels[j] if labels is not None else f"{integrator}"
                ax_fw.set_xlabel(integrator_label, fontsize=12)

    plt.show()

    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, facecolor='white', bbox_inches='tight', dpi=300)