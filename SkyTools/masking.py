import healpy as hp
import numpy as np
from SkyTools import coordhelper
from SkyTools import healpixhelper as myhp
from astropy.coordinates import SkyCoord
import astropy.units as u


def galactic_mask(nside, b_cut, gal_center_cut, galcoords=True):
    """
    Mask healpixels within specified distances of the galactic plane and galactic center
    Parameters
    ----------
    nside: int, resolution of output mask
    b_cut: float in degrees, galactic latitude cut, mask regions abs(b) < b_but
    gal_center_cut: float in degrees, angular distance from galactic center to mask
    galcoords: make mask in galactic coordinates if True, otherwise equatorial

    Returns
    -------
    mask: np.array, mask is 1 where pixels are specified distances from galactic plane and from the galactic center

    """
    mask = np.ones(hp.nside2npix(nside))
    lons, lats = myhp.healpixels2lon_lat(mask)
    if galcoords:
        ls, bs = lons, lats
    else:
        ls, bs = coordhelper.equatorial_to_galactic(lons, lats)
    mask[np.where(np.abs(bs) < b_cut)] = 0

    center_dists = SkyCoord(0 * u.deg, 0 * u.deg).separation(SkyCoord(ls * u.deg, bs * u.deg)).value
    mask[np.where(center_dists < gal_center_cut)] = 0
    return mask




def mask_from_randoms(nside_out, randlons, randlats, random_density, sn_thresh=6.):
    """
    Create a mask based on arrays of random coordinates characterizing the footprint
    Useful if you already have randoms and want to make a mask for e.g. cross correlation with CMB lensing
    This becomes more accurate the more randoms you have
    Parameters
    ----------
    nside_out: int, the nside resolution of the output mask desired
    randlons: array, random longitudes
    randlats: array, random latitudes
    random_density: float, the sky density of randoms in 1/deg^2
    sn_thresh: float, the Poission S/N threshold of counts per pixel required

    Returns
    -------

    """
    # given a random density, determine the resolution at which the Poisson noise isn't terrible,
    # i.e. at which a small fraction of pixels are expected to have zero randoms due to random fluctuation
    num_per_pix = []
    orders = np.arange(3, 13)
    for order in orders:
        pixarea = hp.nside2pixarea(nside=2**order, degrees=True)
        num_per_pix.append(random_density * pixarea)
    poisson_sn = np.sqrt(num_per_pix)
    highest_order = orders[np.argmin(poisson_sn > sn_thresh)]
    nside = 2 ** highest_order


    mask = np.zeros(hp.nside2npix(nside))
    dens_map = myhp.healpix_density_map(lons=randlons, lats=randlats, nsides=nside, deg2=True)
    mask[np.where(dens_map > 0)] = 1

    frac_area = dens_map / random_density
    frac_area[np.where(frac_area > 1.)] = 1.

    mask = hp.ud_grade(mask, nside_out=nside_out)
    frac_area = hp.ud_grade(frac_area, nside_out=nside_out)

    return mask, frac_area