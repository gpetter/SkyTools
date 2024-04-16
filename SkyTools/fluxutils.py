import numpy as np
import scipy.constants as cst
import astropy.units as u
import astropy.constants as const
from colossus.cosmology import cosmology
import pyphot
from pyphot import unit as photunit
plib = pyphot.get_library()
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()


# https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
wise_vega_ab = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}
# https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/scosmos_irac_colDescriptions.html
irac_vega_ab = {'IRAC1': 2.788, 'IRAC2': 3.255, 'IRAC3': 3.743, 'IRAC4': 4.372}


filter_dict = {'u': 'SDSS_u', 'g': 'SDSS_g', 'r':'SDSS_r', 'i':'SDSS_i', 'z':'SDSS_z', 'y':'PS1_y', 'J':'2MASS_J',
               'H':'2MASS_H', 'Ks':'2MASS_Ks', 'W1':'WISE_RSR_W1', 'W2':'WISE_RSR_W2', 'W3':'WISE_RSR_W3',
               'W4':'WISE_RSR_W4'}

def upsample_sed(sed, upsamp_factor=5):
    """
    If an SED is too low resolution to properly integrate, upsample the wavelength resolution
    by interpolating in log space
    Parameters
    ----------
    sed: tuple (wavelength, flux)
    upsamp_factor: factor more points in wavelength grid than original

    Returns
    -------

    """
    lam, f = np.log(sed[0]), np.log(sed[1])
    newlspace = np.linspace(np.min(lam), np.max(lam), upsamp_factor*len(lam))
    return np.exp(np.interp(newlspace, lam, f))


def convert_units(quantity, in_unit, out_unit):
    try:
        return (np.array(quantity) * u.Unit(in_unit)).to(out_unit).value
    except:
        print('Units not convertible')

def convert_spec_unit(nu_or_lam, in_unit='GHz', out_unit='um'):
    """
    Convert a frequency convertible to Hz to a corresponding wavelength
    Parameters
    ----------
    nu
    nu_unit
    lam_unit

    Returns
    -------

    """
    return (np.array(nu_or_lam) * u.Unit(in_unit)).to(out_unit, equivalencies=u.spectral())



def convert_spectral_flux(sed, nu_or_lam_unit='um',
                          fluxunit='uJy', output_unit='erg/s/AA/cm^2'):
    """
    Convert between flux density definitions
    Transform from f_nu to f_lambda

    Parameters
    ----------
    sed: tuple (wavelength/frequency, flux density (either f_nu or f_lambda))
    nu_or_lam_unit: unit of observing frequency or wavelength
    input_unit: unit of observed f_nu flux
    output_unit: unit of f_lambda to convert to

    Returns
    -------

    """
    nu_or_lam, f_nu_or_lam = np.array(sed[0]), np.array(sed[1])
    return (f_nu_or_lam * u.Unit(fluxunit)).to(output_unit,
                                          equivalencies=u.spectral_density(nu_or_lam * u.Unit(nu_or_lam_unit)))



def flux_density_2_flux(sed, nu_or_lam_unit, fluxunit, output_unit='erg/s/cm^2'):
    """
    Convert flux density (either f_lambda or f_nu) observed at frequencies or wavelengths
    to flux units (energy/time/area)
    Leverage fact that nu*f_nu = lambda*f_lambda
    Parameters
    ----------
    sed: tuple (wavelength/frequency, flux density (either f_nu or f_lambda))
    fluxunit: input flux density unit
    nu_or_lam_unit: input wavelength/frequency unit
    output_unit

    Returns
    -------

    """
    f_nu = convert_spectral_flux(sed=sed, nu_or_lam_unit=nu_or_lam_unit,
                                 fluxunit=fluxunit, output_unit='uJy')

    nu = convert_spec_unit(nu_or_lam=sed[0], in_unit=nu_or_lam_unit, out_unit='Hz')

    return (nu * f_nu).to(output_unit)


def r90_assef(w1, w2):
    """
    Test WISE sources if pass Assef et al. 2018 90% reliable AGN selection criterion
    Parameters
    ----------
    w1
    w2

    Returns
    -------

    """
    w1 = np.array(w1)
    w2 = np.array(w2)
    alpha, beta, gamma = 0.65, 0.153, 13.86
    return ((w1 - w2 > alpha) & (w2 <= gamma)) | (w1 - w2 > alpha * np.exp(beta * np.square(w2 - gamma)))


def r75_assef(w1, w2):
    """
    Test WISE sources if pass Assef et al. 2018 75% reliable AGN selection criterion
    Parameters
    ----------
    w1
    w2

    Returns
    -------

    """
    w1 = np.array(w1)
    w2 = np.array(w2)
    alpha, beta, gamma = 0.486, 0.092, 13.07
    return ((w1 - w2 > alpha) & (w2 <= gamma)) | (w1 - w2 > alpha * np.exp(beta * np.square(w2 - gamma)))


def nmgy2mag(nmgys, ivars=None):
    """
    Convert flux in nanomaggies to AB magnitudes
    Parameters
    ----------
    nmgys
    ivars

    Returns
    -------

    """
    nmgys = np.array(np.atleast_1d(nmgys))
    mags = 22.5 - 2.5 * np.log10(nmgys)
    if ivars is not None:
        errs = 1. / np.sqrt(ivars)
        emags = 2.5 / np.log(10) * errs / nmgys
        return mags, emags
    return mags

def nmgy2flux(nmgys, ivars=None, flux_unit=u.uJy):
    """
    Convert flux in nanomaggies to flux density (default microjansky)
    Parameters
    ----------
    nmgys
    ivars
    flux_unit

    Returns
    -------

    """
    nmgys = np.array(np.atleast_1d(nmgys))
    flux = ((3.631 * nmgys * u.uJy).to(flux_unit)).value
    if ivars is not None:
        errs = 1. / np.sqrt(ivars)
        eflux = ((3.631 * errs * u.uJy).to(flux_unit)).value
        return flux, eflux
    return flux

def flux2ABmag(flux, fluxunit='uJy'):
    """
    Convert flux to AB magnitude
    Parameters
    ----------
    flux
    fluxunit

    Returns
    -------

    """
    return -2.5 * np.log10((flux * u.Unit(fluxunit) / (3631. * u.Jy)).value)

def magAB2flux(mag, fluxunit='uJy'):
    """
    Convert AB magnitude to flux density
    Parameters
    ----------
    mag
    fluxunit

    Returns
    -------

    """
    mag = np.array(np.atleast_1d(mag))
    return (3631. * u.Unit(fluxunit) * 10. ** (mag / (-2.5))).to(fluxunit).value




def wise_vega_to_ab(vegamags, filter):
    return vegamags + wise_vega_ab[filter]

def wise_ab_to_vega(abmags, filter):
    return abmags - wise_vega_ab[filter]

def irac_vega_to_ab(vegamags, filter):
    return vegamags + irac_vega_ab[filter]

def irac_ab_to_vega(abmags, filter):
    return abmags - irac_vega_ab[filter]

def spectral_index(f1, f2, nu1, nu2):
    """
    Calculate spectral index alpha for two fluxes measured at two frequencies
    under definition of S \propto \nu ^ {alpha} (ie alpha is negative for a typical synchrotron spectrum)
    Parameters
    ----------
    f1: flux 1
    f2: flux 2
    nu1: frequency 1
    nu2: frequency 2

    Returns
    -------
    alpha: spectral index

    """
    return np.log(f1 / f2) / np.log(nu1 / nu2)


def restframe_luminosity(wavelengths, f_nu, z, fluxerrs=None, lamunit='um', fluxunit='uJy'):
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
    wavelengths, f_nu = np.atleast_1d(wavelengths), np.atleast_1d(f_nu)
    f_nu = np.array(f_nu, dtype='float64')
    # observed frame frequencies
    nus_obs = convert_spec_unit(nu_or_lam=wavelengths, in_unit=lamunit, out_unit='Hz')



    # leverage that nu_obs * fnu_obs = nu_rest * fnu_rest
    nufnu_rest = nus_obs * f_nu * u.Unit(fluxunit)
    print(apcosmo.luminosity_distance(z) ** 2)
    # inverse square law
    nuLnu_rest = ((4 * np.pi * (apcosmo.luminosity_distance(z) ** 2)) * nufnu_rest).to(u.erg/u.s).value
    # rest frame wavelengths
    restwav = wavelengths / (1. + z)
    # if flux errors given, propagate to luminosity
    if fluxerrs is not None:
        fluxerrs = np.atleast_1d(fluxerrs)
        nuLnu_err = fluxerrs / f_nu * nuLnu_rest
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



def luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit='GHz', flux_unit='uJy', energy=True):
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
    nu_f_nu = flux_rest_nu * u.Unit(flux_unit)
    # if you want nu L_nu in erg/s
    if energy:
        return (nu_rest_want * u.Unit(nu_unit) *
                (4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nu_f_nu).to('erg/s').value
    # otherwise L_nu in W/Hz
    else:
        return ((4 * np.pi * apcosmo.luminosity_distance(z) ** 2) * nu_f_nu).to('W/Hz').value





def flux_at_obsnu_from_rest_lum(l_rest, alpha, nu_rest, nu_obs_want, z, nu_unit='GHz', outflux_unit='uJy', energy=True):
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

def radio_sfr_murphy(obsflux, alpha, nu_obs, nu_rest_want, z, nu_unit='GHz', flux_unit='mJy'):
    """
    Murphy et al. 2011 SFR Eq. 14
    Returns
    -------

    """
    l_nu = luminosity_at_rest_nu(obsflux, alpha, nu_obs, nu_rest_want, z,
                                    nu_unit=nu_unit, flux_unit=flux_unit, energy=False)
    nu_ghz = (nu_rest_want * u.Unit(nu_unit)).to('GHz').value
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

def best23_sfr2lum150(sfr):
    return 22.24 + 1.08 * np.log10(sfr)

def best23_lum150_2sfr(loglum150):
    return 10 ** ((loglum150 - 22.24) / 1.08)

def radio_sfr_best(obsflux, alpha, nu_obs_mhz, z, flux_unit='mJy'):
    nu_unit = 'MHz'
    l_nu = luminosity_at_rest_nu(obsflux=obsflux, alpha=alpha, nu_obs=nu_obs_mhz,
                                 nu_rest_want=150, z=z, nu_unit=nu_unit, flux_unit=flux_unit, energy=False)
    return best23_lum150_2sfr(np.log10(l_nu))


# courtesy X-cigale Guang 2020
# converts X-ray flux in erg/s/cm^2 to mJy
def convt_Fx_to_Fnu(flux, flux_err, Elo, Eup):
    """
        Convert X-ray flux to flux density
        Input:
            flux, flux of an X-ray band (untis: erg/s/cm2)
                  array-like objects
            flux_err, the uncertainty of flux
            Elo, Eup: observed-frame energy range of flux_xray (units: keV)
        Output:
            Fnu, X-ray flux density (units: mJy)
            Fnu_err, the error of Fnu
    """
    nu_1keV = 1e3 * cst.eV / cst.h

    Fnu = 1000. * np.array(flux) / (nu_1keV * (Eup-Elo) * 1e-26)
    Fnu_err = 1000. * np.array(flux_err) / (nu_1keV * (Eup-Elo) * 1e-26)

    return Fnu, Fnu_err

def rayleigh_jeans(nu_or_lam, T, nu_or_lam_unit='um', output_unit='W/sr/m^2/Hz'):
    """
    Astropy doesn't seem to have a Rayleigh-jeans intensity function, so implementing here
    Parameters
    ----------
    nu_or_lam: frequency or wavelengths
    T: temperature in K
    nu_or_lam_unit: unit of frequency or wavelength
    output_unit: intensity unit

    Returns
    -------

    """
    nu = convert_spec_unit(nu_or_lam=nu_or_lam, in_unit=nu_or_lam_unit, out_unit='Hz')

    return (2 * nu**2 * const.k_B * (T*u.K) / u.sr / (const.c ** 2)).to(output_unit).value

def extrap_flux_rayleigh_jeans(f_nu, obs_nu_or_lam, nu_or_lam_want, frequencies=True):
    """
    Version of extrap_flux which assumes a Rayleigh-Jeans spectrum, which has a constant alpha=2
    Useful for K-correcting far-infrared observations, as beyond ~100 micron the dust SED of SFGs is approx RJ

    Parameters
    ----------
    f_nu
    obs_nu_or_lam
    nu_or_lam_want
    frequencies: bool, True if passing frequencies, false if passing wavelengths

    Returns
    -------

    """
    if frequencies:
        return extrap_flux(flux=f_nu, alpha=2, nu_obs=obs_nu_or_lam, nu_want=nu_or_lam_want)
    else:
        # in wavelength space, alpha=-2
        return extrap_flux(flux=f_nu, alpha=-2, nu_obs=obs_nu_or_lam, nu_want=nu_or_lam_want)


def integrate_filter(sed, filtername, input_lamunit, input_fluxunit, output_fluxunit='mJy'):
    """
    Integrate a spectrum or SED over a filter curve (wrapper for PyPhot)

    Parameters
    ----------
    fluxdensity: flux density per wavelength or frequency
    lam: wavelengths corresponding to SED
    filtername:
    input_fluxunit
    input_lamunit
    output_fluxunit

    Returns
    -------

    """

    f_lam = convert_spectral_flux(sed=sed, nu_or_lam_unit=input_lamunit,
                         fluxunit=input_fluxunit, output_unit='erg/s/cm^2/AA').value
    f_lam *= photunit['erg/s/cm**2/AA']
    lam = convert_units(sed[0], in_unit=input_lamunit, out_unit='AA') * photunit['AA']
    filt = plib[filter_dict[filtername]]
    flam_int = filt.get_flux(lam, f_lam).value


    return convert_spectral_flux((filt.lpivot.to('AA').value, flam_int), nu_or_lam_unit='AA',
                                 fluxunit='erg/s/cm^2/AA', output_unit=output_fluxunit).value


def syn_color(sed, bluefilter, redfilter, input_fluxunit, input_lamunit, z=0, vega=False):
    """
    Calculate a synthetic color given an SED and two passbands, optionally observed at a given redshift
    Parameters
    ----------
    sed: tuple (wavelength, flux)
    bluefilter
    redfilter
    input_fluxunit
    input_lamunit
    z
    vega

    Returns
    -------

    """

    lam = sed[0] * (1 + z)
    f = sed[1]

    filt1 = plib[filter_dict[bluefilter]]
    filt2 = plib[filter_dict[redfilter]]


    f1 = integrate_filter(sed=(lam, f), filtername=bluefilter,
                          input_fluxunit=input_fluxunit, input_lamunit=input_lamunit, output_fluxunit='uJy')
    f2 = integrate_filter(sed=(lam, f), filtername=redfilter,
                          input_fluxunit=input_fluxunit, input_lamunit=input_lamunit, output_fluxunit='uJy')


    abcolor = flux2ABmag(f1, fluxunit='uJy') - flux2ABmag(f2, fluxunit='uJy')
    if vega:
        return abcolor + (filt2.Vega_zero_mag - filt2.AB_zero_mag) - (filt1.Vega_zero_mag - filt1.AB_zero_mag)
    else:
        return abcolor

def extinct_spec(sed, lam_unit, ebv):
    from dust_extinction.averages import G03_SMCBar
    ext = G03_SMCBar()
    wav, f = np.array(sed[0])*u.Unit(lam_unit), np.array(sed[1])
    # extinction curve is only defined at lambda > 0.1 micron
    goodidx = np.where(wav.to('um').value > 0.1)
    wav, f = wav[goodidx], f[goodidx]

    # extinction curve is only defined at lambda < 3.3 micron. beyond this, extinction is negligible
    # thus only apply extinction to non-IR part of spectrum, then recombine
    non_ir_idx = np.where(wav.to('um').value < 3.3)
    ir_idx = np.where(wav.to('um').value > 3.3)

    non_ir_sed = f[non_ir_idx] * ext.extinguish(x=wav[non_ir_idx], Ebv=ebv)
    ir_sed = f[ir_idx]
    f = np.concatenate((non_ir_sed, ir_sed))


    return wav.value, f