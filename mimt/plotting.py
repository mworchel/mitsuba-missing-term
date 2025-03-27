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
    import matplotlib as mpl
    font = {'family': 'sans-serif',
        'sans-serif': 'Linux Biolinum'}
    mpl.rc('font', **font)
    mpl.rc('text', **{'usetex': False})
    mpl.rc('mathtext', fontset='custom', rm='Linux Biolinum', it='Linux Biolinum:italic', bf='Linux Biolinum:bold')