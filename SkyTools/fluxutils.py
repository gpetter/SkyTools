import numpy as np
import scipy.constants as cst
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
    nmgys = np.array(np.atleast_1d(nmgys))
    mags = 22.5 - 2.5 * np.log10(nmgys)
    if ivars is not None:
        errs = 1. / np.sqrt(ivars)
        emags = 2.5 / np.log(10) * errs / nmgys
        return mags, emags
    return mags

def nmgy2flux(nmgys, ivars=None, flux_unit=u.uJy):
    nmgys = np.array(np.atleast_1d(nmgys))
    flux = ((3.631 * nmgys * u.uJy).to(flux_unit)).value
    if ivars is not None:
        errs = 1. / np.sqrt(ivars)
        eflux = ((3.631 * errs * u.uJy).to(flux_unit)).value
        return flux, eflux
    return flux

def flux2ABmag(flux, fluxunit=u.uJy):
    return -2.5 * np.log10((flux / 3631.) * (fluxunit / u.Jy).value)

def magAB2flux(mag, fluxunit=u.uJy):
    mag = np.array(np.atleast_1d(mag))
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




def luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit=u.GHz, flux_unit=u.uJy, energy=True):
    """
    Calculate rest frame luminosity at any given nu_rest_want, given an observed flux at nu_obs from source at z with
    spectral index alpha
    Parameters
    ----------
    obsflux
    alpha
    nu_obs
    nu_rest_want
    z
    nu_unit
    flux_unit
    energy: bool, True: nuLnu in erg/s, False: L_nu in W/Hz

    Returns
    -------

    """
    # frequency in observed frame corresponding to rest frame frequency we want to probe
    nu_obs_want = nu_rest_want / (1. + z)
    # extrapolate observed flux to above freq
    # 1+z accounts for bandpass squeezing
    flux_rest_nu = extrap_flux(obsflux, alpha, nu_obs, nu_obs_want) / (1. + z)
    nu_f_nu = flux_rest_nu * flux_unit
    # if you want nu L_nu in erg/s
    if energy:
        return (nu_rest_want * nu_unit *
                (4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nu_f_nu).to('erg/s').value
    # otherwise L_nu in W/Hz
    else:
        return ((4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nu_f_nu).to('W/Hz').value

def rest_lum(obsflux, alpha, z, flux_unit=u.uJy):
    return (obsflux * flux_unit * 4 * np.pi * apcosmo.luminosity_distance(z) ** 2 /
            ((1 + z) ** (1 + alpha))).to('W/Hz').value



def flux_at_obsnu_from_rest_lum(l_rest, alpha, nu_rest, nu_obs_want, z, nu_unit=u.GHz, outflux_unit=u.uJy, energy=True):
    """
    The inverse of luminosity_at_rest_nu
    Given a luminosity at rest frame frequency nu_rest, calculate the predicted observed flux at a given nu_obs

    energy: bool, True: input luminosity in erg/s, False: in W/Hz

    Returns
    -------

    """
    if energy:
        # if s_nu propto nu^alpha, then L_nu also propto nu^alpha
        l_nu_rest = ((l_rest * u.erg/u.s) / (nu_rest * nu_unit)).to('W/Hz').value
    else:
        l_nu_rest = l_rest

    # extrapolate l_nu rest to the rest frequency corresponding to the observed bandpass
    l_nu_bandpass_rest = extrap_flux(l_nu_rest, alpha, nu_obs=nu_rest, nu_want=(1+z)*nu_obs_want) * (1. + z)
    f_nu_obs = (l_nu_bandpass_rest * (u.W / u.Hz)) / (4 * np.pi * apcosmo.luminosity_distance(z) ** 2)
    return f_nu_obs.to(outflux_unit).value




def luminosity_at_rest_lam(obsflux_or_mag, alpha, lam_obs, lam_rest_want, z,
                           lam_unit=u.micron, flux_unit=u.uJy, mag=False):
    """
    The same as luminosity_at_rest_nu but with wavelengths
    Parameters
    ----------
    obsflux
    alpha
    lam_obs
    lam_rest_want
    z
    lam_unit
    flux_unit

    Returns
    -------

    """
    obsflux = obsflux_or_mag
    if mag:
        obsflux = magAB2flux(obsflux_or_mag, fluxunit=flux_unit)

    nu_obs = (const.c / (lam_obs * lam_unit)).to(u.GHz).value
    nu_rest_want = (const.c / (lam_rest_want * lam_unit)).to(u.GHz).value
    return luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit=u.GHz, flux_unit=flux_unit)

def radio_sfr_murphy(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit=u.GHz, flux_unit=u.uJy):
    """
    Murphy et al. 2011 SFR Eq. 14
    Returns
    -------

    """
    l_nu = luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z,
                                    nu_unit=nu_unit, flux_unit=flux_unit, energy=False)
    nu_ghz = (nu_rest_want * nu_unit).to('GHz').value
    sfr = 6.64e-22 * (nu_ghz ** -(alpha)) * l_nu
    return sfr

def sfr2radlum_murphy(sfr, alpha, rest_nu, nu_unit=u.GHz, energy=True):
    """
    inversion of radio_sfr
    Returns
    -------

    """
    nu_ghz = (rest_nu * nu_unit).to(u.GHz).value
    nu_hz = (rest_nu * nu_unit).to(u.Hz).value
    l_nu = sfr * 1. / (6.64e-22) * (nu_ghz) ** (alpha)
    if energy:
        return 1e7 * nu_hz * l_nu
    else:
        return l_nu




# courtesy X-cigale Guang 2020
# converts X-ray flux in erg/s/cm^2 to mJy
def convt_Fx_to_Fnu(flux, flux_err, Elo, Eup):
    nu_1keV = 1e3 * cst.eV / cst.h
    '''
    Convert X-ray flux to flux density
    Input:
        flux, flux of an X-ray band (untis: erg/s/cm2)
              array-like objects
        flux_err, the uncertainty of flux
        Elo, Eup: observed-frame energy range of flux_xray (units: keV)
    Output:
        Fnu, X-ray flux density (units: uJy)
        Fnu_err, the error of Fnu
    '''
    Fnu = 1000. * np.array(flux) / (nu_1keV * (Eup-Elo) * 1e-26)
    Fnu_err = 1000. * np.array(flux_err) / (nu_1keV * (Eup-Elo) * 1e-26)

    return Fnu, Fnu_err


