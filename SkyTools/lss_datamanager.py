from astropy.table import Table, vstack
from halomodelpy import hubbleunits
from halomodelpy import cosmo
from . import table_tools
from . import coordhelper
cosmo = cosmo.apcosmo
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

datadir = '/home/graysonpetter/ssd/Dartmouth/data'
rawdir = datadir + '/lss/raw/'

def reduce_desiEDR_qso(update_zs_wu=True):
    """
    Prepare raw DESI EDR Quasar catalogs for clustering measurements.
    Optionally update redshifts with systemic estimates from Wu+23


    """
    thisdir = rawdir + 'desi_edr/'
    outdir = datadir + '/lss/desiQSO_edr/'
    qso_n = Table.read(thisdir + 'QSO_N_clustering.dat.fits')
    qso_s = Table.read(thisdir + 'QSO_S_clustering.dat.fits')
    rand_n = Table.read(thisdir + 'QSO_N_0_clustering.ran.fits')
    rand_s = Table.read(thisdir + 'QSO_S_0_clustering.ran.fits')

    qso = vstack((qso_n, qso_s))
    rand = vstack((rand_n, rand_s))

    qso['RA'], qso['DEC'] = np.array(qso['RA']), np.array(qso['DEC'])
    rand['RA'], rand['DEC'] = np.array(rand['RA']), np.array(rand['DEC'])

    if update_zs_wu:
        wu23 = Table.read(thisdir + 'DESI_EDR_Aug29_redshift_only.fits')
        # if multiple spectra, take recent one, take dark-time spectrum
        wu23 = wu23[np.where(wu23['SURVEY'] == 'sv3')]
        wu23 = wu23[np.where(wu23['PROGRAM'] == 'dark')]
        # check if unique
        print(np.min(np.unique(wu23['TARGETID'], return_counts=True)[1]))
        print('It is unique')

        inwu = np.where(np.in1d(qso['TARGETID'], wu23['TARGETID']))[0]
        # qso['TARGETID'][inwu]

        foo, idx1, idx2 = np.intersect1d(qso['TARGETID'][inwu], wu23['TARGETID'], assume_unique=True, return_indices=True)
        matchidx1 = inwu[idx1]

        qso['Zerr'] = np.zeros(len(qso))
        qso['Z'][matchidx1] = wu23['Z_SYS'][idx2]
        qso['Zerr'][matchidx1] = wu23['Z_SYS_ERR'][idx2]
        # remove sources with catastrophic redshift errors outside DESI range
        qso = table_tools.filter_table_property(qso, 'Z', 0.6, 3.5)


    qso['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(qso['Z']))
    rand['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(rand['Z']))

    qso['weight'] = qso['WEIGHT'] * qso['WEIGHT_FKP']
    rand['weight'] = rand['WEIGHT'] * rand['WEIGHT_FKP']


    qso.write(outdir + 'desiQSO_edr.fits', overwrite=True)
    rand.write(outdir + 'desiQSO_edr_randoms.fits', overwrite=True)


def reduce_desiEDR_lrg(main=True):
    """
    Prepare DESI EDR LRG catalogs

    """
    thisdir = rawdir + 'desi_edr/'
    outdir = datadir + '/lss/desiLRG_edr/'
    if main:
        mainkey = '_main'
    else:
        mainkey = ''
    lrg_n = Table.read(thisdir + 'LRG%s_N_clustering.dat.fits' % mainkey)
    lrg_s = Table.read(thisdir + 'LRG%s_S_clustering.dat.fits' % mainkey)
    lrg_rand_n = Table.read(thisdir + 'LRG%s_N_0_clustering.ran.fits' % mainkey)
    lrg_rand_s = Table.read(thisdir + 'LRG%s_S_0_clustering.ran.fits' % mainkey)

    lrg_n['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(lrg_n['Z']))
    lrg_s['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(lrg_s['Z']))
    lrg_rand_n['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(lrg_rand_n['Z']))
    lrg_rand_s['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(lrg_rand_s['Z']))

    lrg_n['weight'] = lrg_n['WEIGHT'] * lrg_n['WEIGHT_FKP']
    lrg_s['weight'] = lrg_s['WEIGHT'] * lrg_s['WEIGHT_FKP']
    lrg_rand_n['weight'] = lrg_rand_n['WEIGHT'] * lrg_rand_n['WEIGHT_FKP']
    lrg_rand_s['weight'] = lrg_rand_s['WEIGHT'] * lrg_rand_s['WEIGHT_FKP']

    lrg = vstack((lrg_n, lrg_s))
    lrg_rand = vstack((lrg_rand_n, lrg_rand_s))


    lrg.write(outdir + 'desiLRG_edr.fits', overwrite=True)
    lrg_rand.write(outdir + 'desiLRG_edr_randoms.fits', overwrite=True)

def reduce_ebossQSO(rezaie=True, combine=True, nrandratio=20):
    thisdir = rawdir + 'eBOSS/QSO/'
    outdir = datadir + '/lss/eBOSS_QSO/'
    if rezaie:
        ngcdat = Table.read(thisdir + 'rezaie/eBOSS_QSO_NGC_v7_2.dat.fits.gz')
        sgcdat = Table.read(thisdir + 'rezaie/eBOSS_QSO_SGC_v7_2.dat.fits.gz')
        ngcrand = Table.read(thisdir + 'rezaie/eBOSS_QSO_NGC_v7_2.ran.fits.gz')
        sgcrand = Table.read(thisdir + 'rezaie/eBOSS_QSO_SGC_v7_2.ran.fits.gz')
    else:
        ngcdat = Table.read(thisdir + 'ross/eBOSS_QSO_clustering_data-NGC-vDR16.fits')
        sgcdat = Table.read(thisdir + 'ross/eBOSS_QSO_clustering_data-SGC-vDR16.fits')
        ngcrand = Table.read(thisdir + 'ross/eBOSS_QSO_clustering_random-NGC-vDR16.fits')
        sgcrand = Table.read(thisdir + 'ross/eBOSS_QSO_clustering_random-SGC-vDR16.fits')
    dr16_prop_wu = Table.read(datadir + '/QSO_cats/dr16q_prop_Oct23_2022.fits.gz', hdu=1)
    dr16_lyke = Table.read(datadir + '/QSO_cats/DR16Q_v4.fits')

    # replace redshifts with systemic zs from Wu+2022
    ngccoord = SkyCoord(np.array(ngcdat['RA']) * u.deg, np.array(ngcdat['DEC']) * u.deg)
    wucoord = SkyCoord(dr16_prop_wu['RA'] * u.deg, dr16_prop_wu['DEC'] * u.deg)
    lykecoord = SkyCoord(dr16_lyke['RA'] * u.deg, dr16_lyke['DEC'] * u.deg)
    ngcidx, wuidx, d2d, d3d = wucoord.search_around_sky(ngccoord, 1 * u.arcsec)
    ngcdat['Z'][ngcidx] = dr16_prop_wu['Z_SYS'][wuidx]
    ngcdat['sigZ'] = np.zeros(len(ngcdat))
    ngcdat['sigZ'][ngcidx] = dr16_prop_wu['Z_SYS_ERR'][wuidx]
    ngcdat['MBH'] = np.zeros(len(ngcdat))
    ngcdat['MBH'][ngcidx] = dr16_prop_wu['LOGMBH'][wuidx]
    ngcdat['Lbol'] = np.zeros(len(ngcdat))
    ngcdat['Lbol'][ngcidx] = dr16_prop_wu['LOGLBOL'][wuidx]
    ngcdat['BAL_PROB'] = np.full(len(ngcdat), np.nan)

    ngcdat['FIRST_select'] = np.zeros(len(ngcdat))
    ngcidx, lykeidx, d2d, d3d = lykecoord.search_around_sky(ngccoord, 1 * u.arcsec)
    ngcbossbit = dr16_lyke['BOSS_TARGET1']

    # correct few sources which have no bolometric luminosity measurement by using tight correlation between Lbol and M_I
    lbols = ngcdat['Lbol'][ngcidx]
    m_i = dr16_lyke['M_I'][lykeidx]

    ngcdat['gmag'] = np.zeros(len(ngcdat))
    ngcdat['gmag'][ngcidx] = dr16_lyke['PSFMAG'][:, 1][lykeidx] - dr16_lyke['EXTINCTION'][:, 1][lykeidx]

    ngcdat['imag'] = np.zeros(len(ngcdat))
    ngcdat['imag'][ngcidx] = dr16_lyke['PSFMAG'][:, 3][lykeidx] - dr16_lyke['EXTINCTION'][:, 3][lykeidx]
    bad_lbols = np.where(lbols < 43)
    lbols[bad_lbols] = -1. / 2.3 * m_i[bad_lbols] + 35.

    ngcdat['Lbol'][ngcidx] = lbols
    ngcdat['BAL_PROB'][ngcidx] = dr16_lyke['BAL_PROB'][lykeidx]

    sgccoord = SkyCoord(np.array(sgcdat['RA']) * u.deg, np.array(sgcdat['DEC']) * u.deg)
    sgcidx, wuidx, d2d, d3d = wucoord.search_around_sky(sgccoord, 1 * u.arcsec)
    sgcdat['Z'][sgcidx] = dr16_prop_wu['Z_SYS'][wuidx]
    sgcdat['sigZ'] = np.zeros(len(sgcdat))
    sgcdat['sigZ'][sgcidx] = dr16_prop_wu['Z_SYS_ERR'][wuidx]
    sgcdat['MBH'] = np.zeros(len(sgcdat))
    sgcdat['MBH'][sgcidx] = dr16_prop_wu['LOGMBH'][wuidx]
    sgcdat['Lbol'] = np.zeros(len(sgcdat))
    sgcdat['Lbol'][sgcidx] = dr16_prop_wu['LOGLBOL'][wuidx]

    sgcdat['FIRST_select'] = np.zeros(len(sgcdat))
    sgcidx, lykeidx, d2d, d3d = lykecoord.search_around_sky(sgccoord, 1 * u.arcsec)
    sgcbossbit = dr16_lyke['BOSS_TARGET1']

    # correct few sources which have no bolometric luminosity measurement by using tight correlation between Lbol and M_I
    lbols = sgcdat['Lbol'][sgcidx]
    m_i = dr16_lyke['M_I'][lykeidx]

    sgcdat['gmag'] = np.zeros(len(sgcdat))
    sgcdat['gmag'][sgcidx] = dr16_lyke['PSFMAG'][:, 1][lykeidx] - dr16_lyke['EXTINCTION'][:, 1][lykeidx]
    sgcdat['imag'] = np.zeros(len(sgcdat))
    sgcdat['imag'][sgcidx] = dr16_lyke['PSFMAG'][:, 3][lykeidx] - dr16_lyke['EXTINCTION'][:, 3][lykeidx]
    bad_lbols = np.where(lbols < 43)
    lbols[bad_lbols] = -1. / 2.3 * m_i[bad_lbols] + 35.
    sgcdat['Lbol'][sgcidx] = lbols
    sgcdat['BAL_PROB'] = np.full(len(sgcdat), np.nan)
    sgcdat['BAL_PROB'][sgcidx] = dr16_lyke['BAL_PROB'][lykeidx]

    # a couple DR16 redshifts were catastrophically wrong, outside eBOSS range of 0.8-3.5
    ngcdat = ngcdat[np.where((ngcdat['Z'] >= 0.8) & (ngcdat['Z'] <= 3.5))]
    sgcdat = sgcdat[np.where((sgcdat['Z'] >= 0.8) & (sgcdat['Z'] <= 3.5))]
    ngcdat = ngcdat[np.where((ngcdat['imag'] > 0) & (ngcdat['Lbol'] > 30))]
    sgcdat = sgcdat[np.where((sgcdat['imag'] > 0) & (sgcdat['Lbol'] > 30))]

    ngcdat['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(ngcdat['Z']))
    sgcdat['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(sgcdat['Z']))

    ngcdat['weight'] = ngcdat['WEIGHT_SYSTOT'] * ngcdat['WEIGHT_CP'] * ngcdat['WEIGHT_NOZ']
    sgcdat['weight'] = sgcdat['WEIGHT_SYSTOT'] * sgcdat['WEIGHT_CP'] * sgcdat['WEIGHT_NOZ']
    ngcdat = ngcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'MBH', 'Lbol', 'gmag', 'imag', 'BAL_PROB', 'sigZ']
    sgcdat = sgcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'MBH', 'Lbol', 'gmag', 'imag', 'BAL_PROB', 'sigZ']
    if combine:
        totdat = vstack((ngcdat, sgcdat))
    else:
        totdat = ngcdat
    totdat = totdat[np.where(totdat['sigZ'] != -1)]
    totdat.write(outdir + 'eBOSS_QSO.fits', overwrite=True)

    ngcrand = Table(np.random.permutation(ngcrand))
    sgcrand = Table(np.random.permutation(sgcrand))
    ngcrand = ngcrand[:(nrandratio * len(ngcdat))]
    sgcrand = sgcrand[:(nrandratio * len(sgcdat))]
    ngcrand['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(ngcrand['Z']))
    sgcrand['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(sgcrand['Z']))

    ngcrand['weight'] = ngcrand['WEIGHT_SYSTOT'] * ngcrand['WEIGHT_CP'] * ngcrand['WEIGHT_NOZ']
    sgcrand['weight'] = sgcrand['WEIGHT_SYSTOT'] * sgcrand['WEIGHT_CP'] * sgcrand['WEIGHT_NOZ']
    ngcrand = ngcrand['RA', 'DEC', 'Z', 'CHI', 'weight']
    sgcrand = sgcrand['RA', 'DEC', 'Z', 'CHI', 'weight']

    if combine:
        totrand = Table(np.random.permutation(vstack((ngcrand, sgcrand))))
    else:
        totrand = ngcrand
    totrand.write(outdir + 'eBOSS_QSO_randoms.fits', overwrite=True)

def reduce_quaia():
    thisdir = rawdir + 'quaia/'
    outdir = datadir + '/lss/quaia/'

    maglims = ['20_5', '20']
    for mag in maglims:
        qso = Table.read(thisdir + 'quaia_G%s.fits' % mag)
        qso.rename_column('redshift_quaia', 'Z')
        qso.rename_column('redshift_quaia_err', 'Zerr')

        qso.rename_column('ra', 'RA')
        qso.rename_column('dec', 'DEC')
        qso['RA'] = np.array(qso['RA'])
        qso['DEC'] = np.array(qso['DEC'])

        rand = Table.read(thisdir + 'random_G%s.fits' % mag)
        rand.rename_columns(['ra', 'dec'], ['RA', 'DEC'])
        rand['RA'] = np.array(rand['RA'])
        rand['DEC'] = np.array(rand['DEC'])


        gaiafull = Table.read(datadir + '/QSO_cats/gaia/gaiadr3_qsocandidates.fits')

        fullmatch, qso = coordhelper.match_coords(gaiafull, qso, max_sep=1., symmetric=False)
        # choose objects which are selected on gaia properties, not by matches to outside quasar catalogs
        qso = qso[np.where(
            (fullmatch['ClassDSCC'] == 'quasar') | (fullmatch['ClassDCSSA'] == 'quasar') | (fullmatch['Class'] == 'AGN'))]

        qso.write(outdir + 'quaia%s.fits' % mag, overwrite=True)
        rand.write(outdir + 'quaia_randoms%s.fits' % mag, overwrite=True)




def get_ebossQSO(minz=0.8, maxz=3.5):
    qso = Table.read(datadir + '/lss/eBOSS_QSO/eBOSS_QSO.fits')
    rand = Table.read(datadir + '/lss/eBOSS_QSO/eBOSS_QSO_randoms.fits')
    return qso, rand

def get_desiLRG(minz=0.8, maxz=3.5):
    lrg = Table.read(datadir + '/lss/desiLRG_edr/desiLRG_edr.fits')
    rand = Table.read(datadir + '/lss/desiLRG_edr/desiLRG_edr_randoms.fits')
    return lrg, rand

def desiQSO_ebosslike():
    from mocpy import MOC
    qso = Table.read(datadir + '/lss/desiQSO_edr/desiQSO_edr.fits')
    xdqso = Table.read(datadir + '/QSO_cats/xdqso-z-cat.fits')
    foo, qso = coordhelper.match_coords(xdqso, qso, symmetric=False, max_sep=1.)
    qso = table_tools.filter_table_property(qso, 'Z', 0.8)
    rand = Table.read(datadir + '/lss/desiQSO_edr/desiQSO_edr_randoms.fits')
    rand = table_tools.filter_table_property(rand, 'Z', 0.8)

    sdss_u = MOC.from_fits('../../footprints/sdss/sdss9_u.fits')
    qso = qso[sdss_u.contains(qso['RA']*u.deg, qso['DEC']*u.deg)]
    rand = rand[sdss_u.contains(rand['RA']*u.deg, rand['DEC']*u.deg)]


    return qso, rand

def get_quaia(maglim='20'):
    """
    Get quaia quasar catalog Storey-Fisher+23
    Parameters
    ----------
    maglim: str, '20' or '20_5'

    Returns
    -------

    """
    qso = Table.read(datadir + '/lss/quaia/quaia%s.fits' % maglim)
    rand = Table.read(datadir + '/lss/quaia/quaia_randoms%s.fits' % maglim)
    return qso, rand