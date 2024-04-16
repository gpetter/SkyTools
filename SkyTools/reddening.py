from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import astropy.units as u
import numpy as np
import healpy as hp
from astropy.table import Table
from . import healpixhelper

av_over_ebv = {'decam_u': 3.995, 'decam_g': 3.214, 'decam_r': 2.165, 'decam_i': 1.592, 'decam_z': 1.211, 'decam_y': 1.064}


def ebv_at_coords(cat_or_coords):
    """
    Get SFD Galacric reddening at sky positions
    Parameters
    ----------
    cat_or_coords: astropy table with RA, DEC columns OR tuple of (ras, decs)

    Returns
    -------

    """
    if type(cat_or_coords) == Table:
        ras, decs = np.array(cat_or_coords['RA']), np.array(cat_or_coords['DEC'])
    else:
        ras, decs = cat_or_coords

    coords = SkyCoord(np.array(ras)*u.deg, np.array(decs)*u.deg, frame='icrs')
    return SFDQuery()(coords)


def get_healpix_ebv(nside):
    """
    Make a healpix map of Galactic reddening at NSIDE resolution
    Parameters
    ----------
    nside

    Returns
    -------

    """
    # query EBV values at high resolution
    ra, dec = hp.pix2ang(4096, np.arange(hp.nside2npix(nside=4096)), lonlat=True)
    ebv = ebv_at_coords((ra, dec))
    # then take median of child pixels
    return 10**healpixhelper.ud_grade_median(np.log10(ebv), nside_out=nside)


def dered_mags(mags, ebvs, filter):
    av = ebvs * av_over_ebv[filter]
    return mags - av


