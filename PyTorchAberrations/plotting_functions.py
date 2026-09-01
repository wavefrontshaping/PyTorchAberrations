import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import matplotlib.colors as colors
from matplotlib.collections import LineCollection


def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = np.transpose(c, (1,2,0))  
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
    

def logplotTM(array, 
            fig,
            ax, 
            degenerate_mask=None, 
            min_val=1e-2, 
            pola_quadrant=True, 
            lw=1,
            c='r',
            cmap='inferno',
            shrink_cb=1):
    array /= np.max(array)
    pcm = ax.matshow(array,
                   norm=colors.LogNorm(vmin=min_val, vmax=1),
                   cmap=cmap)
    ax.axis('off')
    fig.colorbar(pcm, ax=ax, shrink=shrink_cb)#, extend='max')
    if pola_quadrant:
        ax.axvline(array.shape[1]//2-.5, c=c, lw=lw)
        ax.axhline(array.shape[0]//2-.5, c=c, lw=lw)
    if degenerate_mask is not None:
        plot_outlines(np.tile(degenerate_mask.T,(2,2)), ax=ax, lw=lw, color=c)



def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)

def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)



def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


ZERNIKE_NAMES = [
    'dilat',
    'ft_tilt_H', 'ft_tilt_V',
    'ft_astigm_H', 'ft_defoc', 'ft_astigm_V',
    'ft_tref_V', 'ft_coma_V', 'ft_coma_H', 'ft_tref_H',
    'tilt_H', 'tilt_V',
    'astigm_H', 'defoc', 'astigm_V',
    'tref_V', 'coma_V', 'coma_H', 'tref_H',
    'quad_H', 'sec_astigm_H', 'spherical', 'sec_astigm_V', 'quad_V',
]


def getZernikeCoefs(states):
    '''Get the list of Zernike coefficients from a model state_dict.'''
    import torch
    return [torch.Tensor.cpu(states[name]).numpy()[0] for name in states.keys()]


def showZernikeCoefs(
    zernike_coefs_list,
    labels=None,
    emphasis=False,
    thresh=10,
    title=None,
    names=None,
    n_ft=9,
    n_direct=14,
    ax=None,
    **kwargs
):
    '''Display the amplitude of every Zernike coefficient of a fitted model.

    `zernike_coefs_list` may be a single sequence of coefficients or a list of such
    sequences, in which case they are overlaid.
    '''
    names = list(names if names is not None else ZERNIKE_NAMES)
    n = len(names)

    # accept either one set of coefficients or a list of sets
    if len(zernike_coefs_list) and np.isscalar(zernike_coefs_list[0]):
        zernike_coefs_list = [zernike_coefs_list]

    if labels is not None and len(labels) != len(zernike_coefs_list):
        raise ValueError('`labels` must have one entry per set of coefficients')
    for coefs in zernike_coefs_list:
        if len(coefs) != n:
            raise ValueError(
                f'got {len(coefs)} coefficients but {n} names; pass `names=` if the '
                'model does not use the default 9 + 14 polynomials')

    important = []
    if emphasis:
        important = [i for i in range(n) if np.abs(zernike_coefs_list[0][i]) > thresh]

    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(111)
    else:
        ax1, fig = ax, ax.figure

    x = np.arange(n)
    for ind, coefs in enumerate(zernike_coefs_list):
        ax1.plot(x, coefs, 'o', label=labels[ind] if labels else None, **kwargs)
    if labels:
        ax1.legend()

    # set_xticks before set_xticklabels: recent matplotlib warns otherwise and the
    # labels can end up on the wrong ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=40, ha='right')
    ax1.grid(axis='x', ls=':')
    ax1.set_xlabel('Name of correction function')
    ax1.set_ylabel('Amplitude of correction')

    ylims, xlims = ax1.get_ylim(), ax1.get_xlim()
    if important:
        ax1.vlines(important, ymin=ylims[0], ymax=ylims[1], ls='dashed')
    ax1.hlines([-thresh, thresh], xmin=xlims[0], xmax=xlims[1], ls='dotted')
    ax1.set_ylim(*ylims)
    ax1.set_xlim(*xlims)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(['S'] + list(range(2, n_ft + 2)) + list(range(2, n_direct + 2)))
    ax2.set_xlabel('Index of correction function')

    for i in important:
        ax1.get_xticklabels()[i].set_color('red')
        ax2.get_xticklabels()[i].set_color('red')

    ax1.set_title(title or 'Zernike Coefficients values', pad=38)
    fig.tight_layout()
    return fig, ax1, important
