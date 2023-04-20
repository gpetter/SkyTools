from astropy.table import Table
import numpy as np

def cut_table(cat, prop, minval=-np.inf, maxval=np.inf, inclusive=True):
    if inclusive:
        cat = cat[np.where((cat[prop] >= minval) & (cat[prop] <= maxval))]
    else:
        cat = cat[np.where((cat[prop] > minval) & (cat[prop] < maxval))]

    return cat
