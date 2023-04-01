import numpy as np
import astropy.units as u
import astropy.constants as const
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()

# https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
wise_vega_ab = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}
# https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/scosmos_irac_colDescriptions.html
irac_vega_ab = {'IRAC1': 2.788, 'IRAC2': 3.255, 'IRAC3': 3.743, 'IRAC4': 4.372}

def nmgy2mag(nmgys, ivars=None):
    errs = 1. / np.sqrt(ivars)
    mags = 22.5 - 2.5 * np.log10(nmgys)
    if ivars is not None:
        emags = 2.5 / np.log(10) * errs / nmgys
        return mags, emags
    return mags

def flux2ABmag(flux, fluxunit=u.uJy):
    return -2.5 * np.log10((flux / 3631.) * (fluxunit / u.Jy).value)

def magAB2flux(mag, fluxunit=u.uJy):
    return (3631. * u.Jy * 10. ** (mag / (-2.5))).to(fluxunit).value

def wise_vega_to_ab(vegamags, filter):
    return vegamags + wise_vega_ab[filter]

def wise_ab_to_vega(abmags, filter):
    return abmags - wise_vega_ab[filter]

def irac_vega_to_ab(vegamags, filter):
    return vegamags + irac_vega_ab[filter]

def irac_ab_to_vega(abmags, filter):
    return abmags - irac_vega_ab[filter]


def restframe_luminosity(wavelengths, fluxes, z, fluxerrs=None, wavunit=u.micron, fluxunit=u.uJy):
    """
    Given fluxes in observed frame and a redshift, compute luminosities in the rest frame
    at rest wavelengths = obs wavelengths / (1+z) (ie don't attempt to do K-correction)
    Parameters
    ----------
    wavelengths: array: wavelengths of observed fluxes (default in micron)
    fluxes: array: fluxes in spectral flux density units (default microJy)
    z: float: redshift
    fluxerrs: None or array: if passed, errors calculated
    wavunit: astropy unit: wavelength unit
    fluxunit: astropy unit: spectral flux density unit like microJy

    Returns
    -------
    Rest wavelengths in original units, and rest-frame luminosities (nu L_nu) in erg/s
    Optionally uncertainties on luminosity if flux errors given

    """
    wavelengths, fluxes = np.atleast_1d(wavelengths), np.atleast_1d(fluxes)
    # observed frame frequencies
    nus_obs = (const.c / (wavelengths * wavunit)).to(u.Hz)
    # leverage that nu_obs * fnu_obs = nu_rest * fnu_rest
    nufnu_rest = nus_obs * fluxes * fluxunit
    # inverse square law
    nuLnu_rest = ((4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nufnu_rest).to(u.erg/u.s).value
    # rest frame wavelengths
    restwav = wavelengths / (1. + z)
    # if flux errors given, propagate to luminosity
    if fluxerrs is not None:
        fluxerrs = np.atleast_1d(fluxerrs)
        nuLnu_err = fluxerrs / fluxes * nuLnu_rest
        return restwav, nuLnu_rest, nuLnu_err
    return restwav, nuLnu_rest



def extrap_flux(flux, alpha, nu_obs, nu_want):
    """
    Extrapolate flux to a given frequency for a power law source with spectral index alpha
    where S_nu \propto nu ** alpha
    Parameters
    ----------
    flux
    alpha
    nu_obs
    nu_want

    Returns
    -------

    """
    return flux * (nu_want / nu_obs) ** alpha


def flux_at_any_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z):
    """

    Parameters
    ----------
    obsflux
    alpha
    nu_obs
    rest_nu_want
    z

    Returns
    -------

    """
    # frequency in rest frame corresponding to bandpass in observed frame
    nu_emit = (1. + z) * nu_obs
    # k correct to that frequency
    flux_nu_emit = extrap_flux(obsflux, alpha, nu_obs, nu_emit)
    # k correct again from emitted frequency in rest frame corresponding to nu_obs
    # to any other frequency in rest frame
    flux_rest_nu = extrap_flux(flux_nu_emit, alpha, nu_emit, nu_rest_want)
    return flux_rest_nu

def luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit=u.GHz, flux_unit=u.uJy):
    flux_rest_nu = flux_at_any_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z)
    nu_f_nu = nu_rest_want * flux_rest_nu * nu_unit * flux_unit
    nu_L_nu = ((4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nu_f_nu).to(u.erg/u.s).value
    return nu_L_nu

def luminosity_at_rest_lam(obsflux, alpha, lam_obs, lam_rest_want, z, lam_unit=u.micron, flux_unit=u.uJy):
    nu_obs = (const.c / (lam_obs * lam_unit)).to(u.GHz)
    nu_rest_want = (const.c / (lam_rest_want * lam_unit)).to(u.GHz)
    return luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit=u.GHz, flux_unit=flux_unit)