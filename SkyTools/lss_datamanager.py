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
    import glob
    thisdir = rawdir + 'desi_edr/'
    outdir = datadir + '/lss/desiLRG_edr/'
    if main:
        mainkey = '_main'
    else:
        mainkey = ''
    lrg_n = Table.read(thisdir + 'LRG%s_N_clustering.dat.fits' % mainkey)
    lrg_s = Table.read(thisdir + 'LRG%s_S_clustering.dat.fits' % mainkey)
    randfiles_n = glob.glob(thisdir + 'LRG%s_N_*_clustering.ran.fits' % mainkey)
    randfiles_s = glob.glob(thisdir + 'LRG%s_S_*_clustering.ran.fits' % mainkey)
    randoms = []
    for j in range(len(randfiles_n)):
        randoms.append(Table.read(randfiles_n[j]))

    lrg_rand_n = vstack(randoms)
    randoms = []
    for j in range(len(randfiles_s)):
        randoms.append(Table.read(randfiles_s[j]))
    lrg_rand_s = vstack(randoms)

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

def reduce_desiEDR_elg(notqso=True):
    """
    Prepare DESI EDR LRG catalogs

    """
    thisdir = rawdir + 'desi_edr/'
    outdir = datadir + '/lss/desiELG_edr/'
    if notqso:
        mainkey = 'notqso'
    else:
        mainkey = ''
    elg_n = Table.read(thisdir + 'ELG%s_N_clustering.dat.fits' % mainkey)
    elg_s = Table.read(thisdir + 'ELG%s_S_clustering.dat.fits' % mainkey)
    elg_rand_n = Table.read(thisdir + 'ELG%s_N_0_clustering.ran.fits' % mainkey)
    elg_rand_s = Table.read(thisdir + 'ELG%s_S_0_clustering.ran.fits' % mainkey)

    elg_n['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(elg_n['Z']))
    elg_s['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(elg_s['Z']))
    elg_rand_n['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(elg_rand_n['Z']))
    elg_rand_s['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(elg_rand_s['Z']))

    elg_n['weight'] = elg_n['WEIGHT'] * elg_n['WEIGHT_FKP']
    elg_s['weight'] = elg_s['WEIGHT'] * elg_s['WEIGHT_FKP']
    elg_rand_n['weight'] = elg_rand_n['WEIGHT'] * elg_rand_n['WEIGHT_FKP']
    elg_rand_s['weight'] = elg_rand_s['WEIGHT'] * elg_rand_s['WEIGHT_FKP']

    lrg = vstack((elg_n, elg_s))
    lrg_rand = vstack((elg_rand_n, elg_rand_s))

    lrg.write(outdir + 'desiELG_edr.fits', overwrite=True)
    lrg_rand.write(outdir + 'desiELG_edr_randoms.fits', overwrite=True)

def reduce_ebossQSO(rezaie=True, combine=True, nrandratio=20):
    """
    Prepare eBOSS quasar LSS catalogs from either Rezaie+21 or Ross+20
    Parameters
    ----------
    rezaie: True if Rezaie, False if Ross
    combine: combine the north and south galactic caps, otherwise just get north
    nrandratio: number of randoms compared to number of data points

    """
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
    ngcdat['gaia_G'] = np.full(len(ngcdat), np.nan)
    ngcdat['gaia_G'][ngcidx] = dr16_lyke['GAIA_G_MAG'][lykeidx]

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
    sgcdat['gaia_G'] = np.full(len(sgcdat), np.nan)
    sgcdat['gaia_G'][sgcidx] = dr16_lyke['GAIA_G_MAG'][lykeidx]
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
    ngcdat = ngcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'MBH', 'Lbol', 'gmag', 'imag', 'BAL_PROB', 'sigZ', 'gaia_G']
    sgcdat = sgcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'MBH', 'Lbol', 'gmag', 'imag', 'BAL_PROB', 'sigZ', 'gaia_G']
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


def reduce_eBOSS_LRGs(cmass=True, nrandratio=20):
    """
    Prepare eBOSS LRG LSS catalogs from Ross+20
    Parameters
    ----------
    cmass: include CMASS galaxies
    nrandratio

    Returns
    -------

    """
    thisdir = rawdir + 'eBOSS/LRG/'
    outdir = datadir + '/lss/eBOSS_LRG/'

    which = ''
    if cmass:
        which = 'pCMASS'

    ngcdat = Table.read(thisdir + 'eBOSS_LRG%s_clustering_data-NGC-vDR16.fits' % which)
    sgcdat = Table.read(thisdir + 'eBOSS_LRG%s_clustering_data-SGC-vDR16.fits' % which)

    # ngcdat['L2'] = np.log10(fluxutils.luminosity_at_rest_lam(ngcdat['zmag'], 0, .9, 2., ngcdat['Z'], mag=True))

    ngcdat['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(ngcdat['Z']))
    sgcdat['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(sgcdat['Z']))

    ngcdat['weight'] = ngcdat['WEIGHT_SYSTOT'] * ngcdat['WEIGHT_CP'] * ngcdat['WEIGHT_NOZ']
    sgcdat['weight'] = sgcdat['WEIGHT_SYSTOT'] * sgcdat['WEIGHT_CP'] * sgcdat['WEIGHT_NOZ']
    ngcdat = ngcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'ISCMASS', 'IN_EBOSS_FOOT']
    sgcdat = sgcdat['RA', 'DEC', 'Z', 'CHI', 'weight', 'ISCMASS', 'IN_EBOSS_FOOT']
    totdat = vstack((ngcdat, sgcdat))
    totdat.write(outdir + 'eBOSS_LRG.fits', overwrite=True)


    ngcrand = Table(np.random.permutation(Table.read('raw/eBOSS_LRG%s_clustering_random-NGC-vDR16.fits' % which)))
    ngcrand = ngcrand[:(nrandratio * len(ngcdat))]
    sgcrand = Table(np.random.permutation(Table.read('raw/eBOSS_LRG%s_clustering_random-SGC-vDR16.fits' % which)))
    sgcrand = sgcrand[:(nrandratio * len(sgcdat))]

    ngcrand['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(ngcrand['Z']))
    sgcrand['CHI'] = hubbleunits.add_h_to_scale(cosmo.comoving_distance(sgcrand['Z']))

    ngcrand['weight'] = ngcrand['WEIGHT_SYSTOT'] * ngcrand['WEIGHT_CP'] * ngcrand['WEIGHT_NOZ']
    sgcrand['weight'] = sgcrand['WEIGHT_SYSTOT'] * sgcrand['WEIGHT_CP'] * sgcrand['WEIGHT_NOZ']
    ngcrand = ngcrand['RA', 'DEC', 'Z', 'CHI', 'weight', 'ISCMASS', 'IN_EBOSS_FOOT']
    sgcrand = sgcrand['RA', 'DEC', 'Z', 'CHI', 'weight', 'ISCMASS', 'IN_EBOSS_FOOT']

    totrand = Table(np.random.permutation(vstack((ngcrand, sgcrand))))
    totrand.write(outdir + 'eBOSS_LRG_randoms.fits', overwrite=True)


def reduce_quaia():
    thisdir = rawdir + 'quaia/'
    outdir = datadir + '/lss/quaia/'

    maglims = ['20_5', '20']
    for mag in maglims:
        qso = Table.read(thisdir + 'quaia_G%s.fits' % mag)
        qso.rename_column('redshift_quaia', 'Z')
        qso.rename_column('redshift_quaia_err', 'Zerr')
        qso.rename_column('phot_g_mean_mag', 'G')
        qso.rename_column('phot_rp_mean_mag', 'RP')
        qso.rename_column('phot_bp_mean_mag', 'BP')
        qso.rename_column('mag_w1_vg', 'W1')
        qso.rename_column('mag_w2_vg', 'W2')

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
        qso['e_G'] = fullmatch['e_Gmag']
        qso['e_RP'] = fullmatch['e_RPmag']
        qso['e_BP'] = fullmatch['e_BPmag']
        # choose objects which are selected on gaia properties, not by matches to outside quasar catalogs
        qso = qso[np.where(
            (fullmatch['ClassDSCC'] == 'quasar') | (fullmatch['ClassDCSSA'] == 'quasar') | (fullmatch['Class'] == 'AGN'))]
        qso = qso['RA', 'DEC', 'Z', 'Zerr', 'G', 'BP', 'RP', 'e_G', 'e_BP', 'e_RP', 'W1', 'W2']
        qso.write(outdir + 'quaia%s.fits' % mag, overwrite=True)
        rand.write(outdir + 'quaia_randoms%s.fits' % mag, overwrite=True)


def reduce_hsc_lbg(dropout='g', deeplim=True):
    import glob
    import pandas as pd
    thisdir = rawdir + 'hsc_lbg/'
    outdir = datadir + '/lss/HSC_LBG/'
    if dropout == 'g':
        dirname = 'gri'
        if deeplim:
            maglim='24.00'
        else:
            maglim = '23.50'
    else:
        dirname = 'riz'
        if deeplim:
            maglim = '24.50'
        else:
            maglim = '24.00'

    datfiles = glob.glob(thisdir + dirname + '/' + '*_%s_*_W_*.cat' % maglim)
    randfiles = glob.glob(thisdir + dirname + '/' + '*_random_W_*.cat')

    dattab = Table()
    randtab = Table()
    for datfile in datfiles:
        datdf = pd.read_csv(datfile, delim_whitespace=True, names=['RA', 'DEC', 'jackknife'])
        dattab = vstack((dattab, Table.from_pandas(datdf)))
    for randfile in randfiles:
        randdf = pd.read_csv(randfile, delim_whitespace=True, names=['RA', 'DEC', 'jackknife'])
        randtab = vstack((randtab, Table.from_pandas(randdf)))

    dattab.write(outdir + 'HSC_LBG_%s.fits' % dropout, overwrite=True)
    randtab.write(outdir + 'HSC_LBG_%s_randoms.fits' % dropout, overwrite=True)


def get_ebossQSO(minz=0.8, maxz=3.5, quaiacut=None):
    qso = Table.read(datadir + '/lss/eBOSS_QSO/eBOSS_QSO.fits')
    rand = Table.read(datadir + '/lss/eBOSS_QSO/eBOSS_QSO_randoms.fits')
    if quaiacut is not None:
        qso = qso[np.where((qso['gaia_G'] < quaiacut) & (qso['gaia_G'] > 0))]
    qso = table_tools.filter_table_property(qso, 'Z', minval=minz, maxval=maxz)
    rand = table_tools.filter_table_property(rand, 'Z', minval=minz, maxval=maxz)
    return qso, rand

def get_desiLRG_edr(minz=None, maxz=None):
    lrg = Table.read(datadir + '/lss/desiLRG_edr/desiLRG_edr.fits')
    rand = Table.read(datadir + '/lss/desiLRG_edr/desiLRG_edr_randoms.fits')
    if minz is not None:
        lrg = lrg[np.where(lrg['Z'] >= minz)]
        rand = rand[np.where(rand['Z'] >= minz)]
    if maxz is not None:
        lrg = lrg[np.where(lrg['Z'] <= maxz)]
        rand = rand[np.where(rand['Z'] <= maxz)]
    return lrg, rand

def get_desiELG_edr(minz=None, maxz=None):
    elg = Table.read(datadir + '/lss/desiELG_edr/desiELG_edr.fits')
    rand = Table.read(datadir + '/lss/desiELG_edr/desiELG_edr_randoms.fits')
    if minz is not None:
        elg = elg[np.where(elg['Z'] >= minz)]
        rand = rand[np.where(rand['Z'] >= minz)]
    if maxz is not None:
        elg = elg[np.where(elg['Z'] <= maxz)]
        rand = rand[np.where(rand['Z'] <= maxz)]
    return elg, rand

def get_desiQSO_edr(ebosslike=False, minz=None, maxz=None):
    """
    Fetch DESI EDR QSOs. optionally find those which would have been observed by eBOSS by matching to XDQSO
    Parameters
    ----------
    ebosslike

    Returns
    -------

    """
    qso = Table.read(datadir + '/lss/desiQSO_edr/desiQSO_edr.fits')
    rand = Table.read(datadir + '/lss/desiQSO_edr/desiQSO_edr_randoms.fits')

    if ebosslike:
        from mocpy import MOC
        qso = table_tools.filter_table_property(qso, 'Z', 0.8)
        rand = table_tools.filter_table_property(rand, 'Z', 0.8)
        sdss_u = MOC.from_fits(datadir + '/footprints/sdss/sdss9_u.fits')
        qso = qso[sdss_u.contains(qso['RA'] * u.deg, qso['DEC'] * u.deg)]
        rand = rand[sdss_u.contains(rand['RA'] * u.deg, rand['DEC'] * u.deg)]

        alldesi = qso.copy()
        alldesirand = rand.copy()

        xdqso = Table.read(datadir + '/QSO_cats/xdqso-z-cat.fits')
        foo, qso = coordhelper.match_coords(xdqso, qso, symmetric=False, max_sep=1.)

        # DESI QSOs targeted by eBOSS have different dndz
        # need to downsample random catalog to match the new dndz
        ntot2nboss = np.sum(alldesi['weight']) / np.sum(qso['weight'])
        ebosshist, binedges = np.histogram(qso['Z'], bins=15, range=(0.8, 3.5), weights=qso['weight'])
        allhist, binedges = np.histogram(alldesi['Z'], bins=15, range=(0.8, 3.5), weights=alldesi['weight'])
        eboss_all_ratio = ntot2nboss * ebosshist / allhist

        # downsampling
        pdraw = eboss_all_ratio[np.digitize(rand['Z'], binedges) - 1]
        pdraw /= np.sum(pdraw)
        rand = rand[np.random.choice(len(rand), int(len(rand) / 1.5), replace=False, p=pdraw)]

        # now need to weight DESIxeBOSS quasars to have same weighted dndz as eBOSS-only catalog

        def weight_dndz1_to_dndz2(cat1, cat2, zbins=15):
            minz1, maxz1 = np.min(cat1['Z']), np.max(cat1['Z'])
            minz2, maxz2 = np.min(cat2['Z']), np.max(cat2['Z'])

            minz = np.min([minz1, minz2]) - 0.01
            maxz = np.max([maxz1, maxz2]) + 0.01

            zhist1, edges = np.histogram(cat1['Z'], bins=zbins, range=(minz, maxz), density=True)
            zhist2, edges = np.histogram(cat2['Z'], bins=zbins, range=(minz, maxz), density=True)

            ratio = zhist2 / zhist1
            cat1['zweight'] = ratio[np.digitize(cat1['Z'], bins=edges) - 1]
            cat1['weight'] *= cat1['zweight']
            return cat1
        ebossqso, ebossrand = get_ebossQSO()
        qso = weight_dndz1_to_dndz2(qso, ebossrand)
        rand = weight_dndz1_to_dndz2(rand, ebossrand)
    if minz is not None:
        qso = qso[np.where(qso['Z'] >= minz)]
        rand = rand[np.where(rand['Z'] >= minz)]
    if maxz is not None:
        qso = qso[np.where(qso['Z'] <= maxz)]
        rand = rand[np.where(rand['Z'] <= maxz)]


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

def get_hsc_lbg(dropout='g'):
    lbg = Table.read(datadir + '/lss/HSC_LBG/HSC_LBG_%s.fits' % dropout)
    rand = Table.read(datadir + '/lss/HSC_LBG/HSC_LBG_%s_randoms.fits' % dropout)
    return lbg, rand


def overlap_weights_1d(prop_arr, prop_range, nbins):
    hists = []
    for j in range(len(prop_arr)):
        hist, binedges = np.histogram(prop_arr[j], bins=nbins, range=prop_range, density=True)
        hists.append(hist)
    hists = np.array(hists)
    min_hist = np.amin(hists, axis=0)
    weights = []
    for j in range(len(prop_arr)):
        dist_ratio = min_hist / hists[j]
        dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0
        weights.append(dist_ratio[np.digitize(prop_arr[j], bins=binedges)-1])
    return weights



def overlap_weights_2d(prop1_arr, prop2_arr, prop1_range, prop2_range, nbins):
    from scipy import stats, interpolate
    hists, bin_locs = [], []

    for j in range(len(prop1_arr)):
        thishist = stats.binned_statistic_2d(prop1_arr[j], prop2_arr[j], None, statistic='count',
                                             bins=[nbins[0], nbins[1]],
                                             range=[[prop1_range[0] - 0.001, prop1_range[1] + 0.001],
                                                    [prop2_range[0] - 0.001, prop2_range[1] + 0.001]],
                                             expand_binnumbers=True)
        normed_hist = thishist[0] / np.sum(thishist[0])
        hists.append(normed_hist)
        bin_locs.append(np.array(thishist[3]) - 1)

    hists = np.array(hists)
    min_hist = np.amin(hists, axis=0)
    weights = []

    for j in range(len(prop1_arr)):
        dist_ratio = min_hist / hists[j]
        dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0
        bin_idxs = bin_locs[j]
        weights.append(dist_ratio[bin_idxs[0], bin_idxs[1]])
    return weights


def match_random_zdist(samplecat, randomcat, nbins=15):
    randminz = np.min(randomcat['Z'])
    randmaxz = np.max(randomcat['Z'])
    zhist, edges = np.histogram(samplecat['Z'], bins=nbins, range=(randminz-1e-3, randmaxz+1e-3), density=True)
    randhist, edges = np.histogram(randomcat['Z'], bins=nbins, range=(randminz-1e-3, randmaxz+1e-3), density=True)
    ratio = zhist/randhist
    sampleweights = ratio[np.digitize(randomcat['Z'], bins=edges) - 1]
    randomcat['weight'] *= sampleweights
    #subrandcat = randomcat[np.random.choice(len(randomcat), size=(10*len(samplecat)), replace=False, p=(sampleweights / np.sum(sampleweights)))]
    return randomcat

def weight_common_dndz(cat1, cat2, zbins=15):
    minz1, maxz1 = np.min(cat1['Z'])-1e-3, np.max(cat1['Z']+1e-3)
    minz2, maxz2 = np.min(cat2['Z'])-1e-3, np.max(cat2['Z']+1e-3)
    minz = np.min([minz1, minz2])
    maxz = np.max([maxz1, maxz2])
    zhist1, edges = np.histogram(cat1['Z'], bins=zbins, range=(minz, maxz), density=True)
    zhist2, edges = np.histogram(cat2['Z'], bins=zbins, range=(minz, maxz), density=True)
    dndz_product = np.sqrt(zhist1 * zhist2)
    ratio1 = zhist1/dndz_product
    ratio2 = zhist2/dndz_product
    cat1['zweight'] = ratio1[np.digitize(cat1['Z'], bins=edges) - 1]
    cat2['zweight'] = ratio2[np.digitize(cat2['Z'], bins=edges) - 1]
    cat1['weight'] *= cat1['zweight']
    cat2['weight'] *= cat2['zweight']
    return cat1, cat2


def rolling_percentile_selection(cat, prop, minpercentile, maxpercentile=100, nzbins=100):
    """
    choose highest nth percentile of e.g. luminosity in bins of redshift
    """
    minz, maxz = np.min(cat['Z']), np.max(cat['Z'])
    zbins = np.linspace(minz, maxz, nzbins)
    idxs = []
    for j in range(len(zbins)-1):
        catinbin = cat[np.where((cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))]
        thresh = np.percentile(catinbin[prop], minpercentile)
        hithresh = np.percentile(catinbin[prop], maxpercentile)
        idxs += list(np.where((cat[prop] > thresh) & (cat[prop] <= hithresh) &
                              (cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))[0])

    newcat = cat[np.array(idxs)]
    return newcat