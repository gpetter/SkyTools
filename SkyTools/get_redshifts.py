from astropy.table import Table, vstack
import glob
import numpy as np
from . import coordhelper
from . import healpixhelper

datadir = '/home/graysonpetter/ssd/Dartmouth/data/'

def match_to_spec_surveys(cat, seplimit, maxphotz=4):

    # set up empty array for redshifts
    cat['Z'] = np.full(len(cat), np.nan)
    # flag for photometric redshifts
    cat['phot_flag'] = np.full(len(cat), np.nan)
    cat['z_source'] = np.full(len(cat), np.nan)
    #tabcoord = SkyCoord(maskedtab['RA'] * u.deg, maskedtab['DEC'] * u.deg)

    crude_mask = healpixhelper.healpix_density_map(cat['RA'], cat['DEC'], nsides=32)
    crude_mask[np.where(crude_mask > 0)] = 1

    # index tracking which catalog the redshift comes from
    k = 0
    # list of strings denoting paths to spectroscopic catalogs
    catlist = []

    # loop through redshift catalogs, matching and copying redshift to WISE AGN
    path = datadir + 'photozs/use/'
    folders = glob.glob(path + '*')

    folders.insert(0, folders.pop(folders.index(path+'chung')))
    folders.insert(1, folders.pop(folders.index(path+'bootes_duncan2')))
    folders.insert(2, folders.pop(folders.index(path+'bootes_duncan1')))
    idxs_w_redshifts = []
    for folder in folders:
        files = glob.glob(folder + '/*.fits')
        for file in files:
            spectab = Table.read(file)

            spectab = spectab[np.where((spectab['Zphot'] > 0.001) & (spectab['Zphot'] < maxphotz))]
            spectab = spectab[healpixhelper.inmask((spectab['RA'], spectab['DEC']), crude_mask)]
            if len(spectab) > 0:

                #idx, d2d, d3d = speccoords.match_to_catalog_sky(tabcoord)
                specidx, catidx = coordhelper.match_coords((spectab['RA'], spectab['DEC']), (cat['RA'], cat['DEC']), max_sep=seplimit, symmetric=False)

                cat['Z'][catidx] = spectab['Zphot'][specidx]
                # if copying photometric redshift, set flag
                cat['phot_flag'][catidx] = 1
                # set integer key to the number corresponding to the redshift catalog
                cat['z_source'][catidx] = k
                k += 1
                # append name of corresponding redshift catalog
                catlist.append(file)
                idxs_w_redshifts += list(catidx)



    path = datadir + 'specsurveys/use/'
    folders = glob.glob(path + '*')
    idxs_w_redshifts = []
    for folder in folders:
        files = glob.glob(folder + '/*.fits')
        for file in files:
            print(file)
            spectab = Table.read(file)
            spectab = spectab[np.where((spectab['Zspec'] > 0.001) & (spectab['Zspec'] < 8))]
            spectab = spectab[healpixhelper.inmask((spectab['RA'], spectab['DEC']), crude_mask)]
            if len(spectab) > 0:
                specidx, catidx = coordhelper.match_coords((spectab['RA'], spectab['DEC']), (cat['RA'], cat['DEC']),
                                                           max_sep=seplimit, symmetric=False)

                cat['Z'][catidx] = spectab['Zspec'][specidx]
                # if copying photometric redshift, set flag
                cat['phot_flag'][catidx] = 0
                # set integer key to the number corresponding to the redshift catalog
                cat['z_source'][catidx] = k
                k += 1
                # append name of corresponding redshift catalog
                catlist.append(file)
                idxs_w_redshifts += list(catidx)



    #with open('catalogs/redshifts/z_source_key.txt', 'w') as f:
    #    for j in range(len(catlist)):
    #        f.write('%s %s\n' % (j, catlist[j]))


    cat['hasz'] = np.zeros(len(cat))
    cat['hasz'][np.where(np.isfinite(cat['Z']))] = 1

    return cat