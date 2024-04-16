from astropy.table import Table
import numpy as np

def table_in_coord_bounds(t, ramin=0., ramax=360., decmin=-90., decmax=90., inclusive=True):
    """
    filter an astropy table on RA and DEC columns

    Parameters
    ----------
    ramin
    ramax
    decmin
    decmax

    Returns
    -------
    table inside specified RA/DEC bounds
    """
    if inclusive:
        t = t[np.where((t['RA'] <= ramax) & (t['RA'] >= ramin) & (t['DEC'] <= decmax) & (t['DEC'] >= decmin))]
    else:
        t = t[np.where((t['RA'] < ramax) & (t['RA'] > ramin) & (t['DEC'] < decmax) & (t['DEC'] > decmin))]
    return t

def filter_table_property(t, prop, minval=None, maxval=None, inclusive=True):
    """
    Filter astropy table on a property, given the column name and min/max bounds
    Parameters
    ----------
    t
    prop
    min
    max
    inclusive

    Returns
    -------

    """
    if inclusive:
        if minval is not None:
            t = t[np.where(t[prop] >= minval)]
        if maxval is not None:
            t = t[np.where(t[prop] <= maxval)]
    else:
        if minval is not None:
            t = t[np.where(t[prop] > minval)]
        if maxval is not None:
            t = t[np.where(t[prop] < maxval)]
    return t

def plot_catalog_skydist(cat, nsides=32):
    from . import healpixhelper
    import healpy as hp
    dens = healpixhelper.healpix_density_map(cat, nsides=32, deg2=True)
    mollmap = hp.mollview(dens)
    return mollmap