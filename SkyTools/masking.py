import healpy as hp
import numpy as np
from SkyTools import coordhelper
from SkyTools import healpixhelper as myhp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


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


def cat_in_moc(cat, mocfile):
    """
    Pass an astropy table and a MOC file, return entries in the catalog which are in the MOC footprint
    Parameters
    ----------
    cat
    mocfile

    Returns
    -------

    """
    cat = cat[mocfile.contains(np.array(cat['RA']) * u.deg, np.array(cat['DEC']) * u.deg)]
    return cat


def mask_from_randoms(nside_out, randlons, randlats, random_density=None, sn_thresh=6., dens_map=None):
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
    dens_map: array, optionally pass the density map already computed, use this if random catalog needs to be split
    into multiple chunks to avoid memory overflow

    Returns
    -------

    """
    # given a random density, determine the resolution at which the Poisson noise isn't terrible,
    # i.e. at which a small fraction of pixels are expected to have zero randoms due to random fluctuation

    if random_density is None:
        foo = myhp.healpix_density_map(lons=randlons, lats=randlats, nsides=64, deg2=True)
        foo[np.where(foo < 1)] = np.nan
        random_density = np.nanmedian(foo)

    num_per_pix = []
    orders = np.arange(3, 13)
    for order in orders:
        pixarea = hp.nside2pixarea(nside=2**order, degrees=True)
        num_per_pix.append(random_density * pixarea)
    poisson_sn = np.sqrt(num_per_pix)
    highest_order = orders[np.argmin(poisson_sn > sn_thresh)]
    nside = 2 ** highest_order


    mask = np.zeros(hp.nside2npix(nside))

    if dens_map is None:
        dens_map = myhp.healpix_density_map(lons=randlons, lats=randlats, nsides=nside, deg2=True)
    mask[np.where(dens_map > 0)] = 1

    frac_area = dens_map / random_density
    frac_area[np.where(frac_area > 1.)] = 1.

    mask = hp.ud_grade(mask, nside_out=nside_out)
    frac_area = hp.ud_grade(frac_area, nside_out=nside_out)

    return mask, frac_area

def desi_systmap(syst_label):
    """
        Read in angular systematics map produced by DESI (Myers+23)
        :param syst_label:
        :return:
    """
    pixweight = Table.read('/home/graysonpetter/ssd/Dartmouth/data/desi_targets/syst_maps/pixweight-1-dark.fits')
    systmap = np.empty(hp.nside2npix(256))
    systmap[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['%s' % syst_label]
    if syst_label == 'PSFDEPTH_W2':
        systmap = 22.5 - 2.5 * np.log10(5 / np.sqrt(systmap)) - 3.339
    elif (syst_label == 'PSFDEPTH_Z') | (syst_label == 'PSFDEPTH_R') | (syst_label == 'PSFDEPTH_G'):
        systmap = 22.5 - 2.5 * np.log10(5 / np.sqrt(systmap))
    return systmap

def in_eboss(ras, decs, northonly=False):
    import pymangle
    """
    Filter coordinates in eBOSS quasar footprint
    :param ras:
    :param decs:
    :param northonly:
    :return:
    """

    ebossmoc = pymangle.Mangle('../data/footprints/eBOSS/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
    good_idxs = ebossmoc.contains(ras, decs)
    if northonly:
        innorth = (ras > 90) & (ras < 290)
        good_idxs = good_idxs & innorth
    return good_idxs



def in_ls_dr8(ra, dec):
    from mocpy import MOC
    """
    Filter coordinates inside Legacy Survey DR8 footprint, since we are using photo-z
    :param ra:
    :param dec:
    :return:
    """
    northfoot = MOC.load('/home/graysonpetter/ssd/Dartmouth/data/footprints/legacySurvey/dr8_photoz_duncan/desi_lis_dr8_pzn.rcf.moc.fits')
    southfoot = MOC.load('/home/graysonpetter/ssd/Dartmouth/data/footprints/legacySurvey/dr8_photoz_duncan/desi_lis_dr8_pzs.rcf.moc.fits')
    # for some reason there are patches where Duncan doesn't provide photo zs, identified by hand
    inbad = ((ra < 243.4) & (ra > 242) & (dec < 35.3) & (dec > 34.2)) | \
            ((ra < 147) & (ra > 145.7) & (dec < 57.5) & (dec > 56.85)) | \
            ((ra < 175.4) & (ra > 174.5) & (dec < 43.45) & (dec > 42.75)) | \
            ((ra < 150.66) & (ra > 150) & (dec < 33) & (dec > 32.6)) | \
            ((ra < 150.16) & (ra > 149.5) & (dec < 33.75) & (dec > 33.1)) | \
            ((ra < 166) & (ra > 165.5) & (dec < 61.8) & (dec > 61.5))
    infoot = (northfoot.contains(ra * u.deg, dec * u.deg) |
                            southfoot.contains(ra * u.deg, dec * u.deg)) & np.logical_not(inbad)
    return infoot


def outside_galaxy(ras, decs, galcut=0, betacut=90, ebvcut=0.1, stardenscut=2000.):
    """
    Filter coordinates by galactic/ecliptic cuts, or reddening, stellar density cuts
    :param ras:
    :param decs:
    :param galcut:
    :param betacut:
    :param ebvcut:
    :param stardenscut:
    :return:
    """
    stardens = desi_systmap('STARDENS')
    ebv = desi_systmap('EBV')

    densities = stardens[hp.ang2pix(hp.npix2nside(len(stardens)), ras, decs, lonlat=True)]
    ebvs = ebv[hp.ang2pix(hp.npix2nside(len(ebv)), ras, decs, lonlat=True)]

    l, b = coordhelper.equatorial_to_galactic(ras, decs)
    lams, betas = coordhelper.equatorial_to_ecliptic(ras, decs)
    goodbs = np.abs(b) > galcut
    goodbetas = (betas < betacut)

    gooddens = (densities < stardenscut)
    goodebv = (ebvs < ebvcut)
    return goodebv & gooddens & goodbs & goodbetas


def cat_in_eboss(cat):
    return cat[in_eboss(cat['RA'], cat['DEC'])]

def cat_in_ls_dr8(t):
    infoot = in_ls_dr8(t['RA'], t['DEC'])
    return t[infoot]



def cat_outside_galaxy(cat, galcut=0, betacut=90, ebvcut=0.1, stardenscut=2000.):
    cat = cat[outside_galaxy(cat['RA'], cat['DEC'],
                             galcut=galcut, betacut=betacut, ebvcut=ebvcut, stardenscut=stardenscut)]
    return cat