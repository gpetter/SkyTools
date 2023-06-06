from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.table import Table, vstack, hstack
from . import healpixhelper
import healpy as hp

# convert ras and decs to galactic l, b coordinates
def equatorial_to_galactic(ra, dec):
	"""

	Parameters
	----------
	ra: float or array
	dec: float or array

	Returns
	-------
	tuple of galactic longitudes, latitudes

	"""
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian * u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian * u.rad.to('deg'))
	return ls, bs


# convert ras and decs to galactic l, b coordinates
def galactic_to_equatorial(l, b):
	"""

	Parameters
	----------
	l: float or array
	b: float or array

	Returns
	-------
	tuple of (ra, dec)

	"""
	coords = SkyCoord(l, b, unit='deg', frame='galactic')
	ras = np.array(coords.icrs.ra * u.deg)
	decs = np.array(coords.icrs.dec * u.deg)
	return ras, decs

# convert ras and decs to galactic l, b coordinates
def equatorial_to_ecliptic(ra, dec):
	"""

	Parameters
	----------
	ra: float or array
	dec: float or array

	Returns
	-------
	tuple of (ecliptic longitude, ecliptic latitude)

	"""
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	lons = np.array(ra_decs.barycentricmeanecliptic.lon.radian * u.rad.to('deg'))
	lats = np.array(ra_decs.barycentricmeanecliptic.lat.radian * u.rad.to('deg'))
	return lons, lats

# convert ras and decs to galactic l, b coordinates
def galactic_to_ecliptic(l, b):
	"""

	Parameters
	----------
	l: float or array
	b: float or array

	Returns
	-------
	tuple of (ecliptic longitude, ecliptic latitude)

	"""
	coords = SkyCoord(l, b, unit='deg', frame='galactic')
	lons = np.array(coords.barycentricmeanecliptic.lon.radian * u.rad.to('deg'))
	lats = np.array(coords.barycentricmeanecliptic.lat.radian * u.rad.to('deg'))
	return lons, lats



# find matches between two catalogs of coordinates
def match_coords(catalog_or_coords1, catalog_or_coords2, max_sep=2., find_common_footprint=False, symmetric=True):
	"""

	Parameters
	----------
	catalog_or_coords1: tuple of (lon, lat) or an astropy Table containing columns 'RA', 'DEC'
	catalog_or_coords2: same as 1
	max_sep: float in units of arcsec
	find_common_footprint: bool
	symmetric: bool: if True, symmetric best match a la TOPCAT, otherwise best match for each row in catalog 2

	Returns
	-------

	"""
	# add an astropy unit of arcsec to matching radius
	max_sep *= u.arcsec

	if type(catalog_or_coords1) == Table:
		lons1, lats1 = catalog_or_coords1['RA'], catalog_or_coords1['DEC']
		lons2, lats2 = catalog_or_coords2['RA'], catalog_or_coords2['DEC']
	else:
		# unpack coordinates
		lons1, lats1 = catalog_or_coords1
		lons2, lats2 = catalog_or_coords2
	# keep track of original indices
	original_idxs1 = np.arange(len(lons1))
	original_idxs2 = np.arange(len(lons2))


	# matching finds closest object for every other object, so comparing catalogs without perfect overlap can be
	# inefficient, such as matching an all sky catalog to a field like COSMOS
	# keep only sources in a common footprint of both
	if find_common_footprint:
		# create density maps of each
		dens1 = healpixhelper.healpix_density_map(lons1, lats1, 16)
		dens2 = healpixhelper.healpix_density_map(lons2, lats2, 16)
		# multiply together to find union
		dens = dens1 * dens2
		idxs1_in_footprint = np.where(dens[hp.ang2pix(16, lons1, lats1, lonlat=True)] > 0)[0]
		idxs2_in_footprint = np.where(dens[hp.ang2pix(16, lons2, lats2, lonlat=True)] > 0)[0]
		lons1, lats1 = lons1[idxs1_in_footprint], lats1[idxs1_in_footprint]
		lons2, lats2 = lons2[idxs2_in_footprint], lats2[idxs2_in_footprint]
		original_idxs1, original_idxs2 = idxs1_in_footprint, idxs2_in_footprint

	skycoord1 = SkyCoord(lons1*u.deg, lats1*u.deg, frame='icrs')
	skycoord2 = SkyCoord(lons2*u.deg, lats2*u.deg, frame='icrs')

	idx1, d2d, d3d = skycoord2.match_to_catalog_sky(skycoord1)
	sep_constraint = np.where(d2d < max_sep)[0]
	idx2 = original_idxs2[sep_constraint]
	idx1 = original_idxs1[idx1[sep_constraint]]
	if symmetric:
		newidx2, d2d, d3d = skycoord1.match_to_catalog_sky(skycoord2)
		sep_constraint = np.where(d2d < max_sep)[0]
		newidx1 = original_idxs1[sep_constraint]
		newidx2 = original_idxs2[newidx2[sep_constraint]]
		idx1, idx2 = np.intersect1d(idx1, newidx1), np.intersect1d(idx2, newidx2)


	if type(catalog_or_coords1) == Table:
		return catalog_or_coords1[idx1], catalog_or_coords2[idx2]
	else:
		return idx1, idx2

def match_cats_allfrom1st(cat1, cat2, sep):

	idx1, idx2 = match_coords((cat1['RA'], cat1['DEC']), (cat2['RA'], cat2['DEC']), max_sep=sep)
	cat2.rename_columns(['RA', 'DEC'], ['RA2', 'DEC2'])
	unmatchedidx = np.where(np.logical_not(np.in1d(np.arange(len(cat1)), idx1)))[0]
	matchcat1, unmatchcat = cat1[idx1], cat1[unmatchedidx]
	matchcat = hstack((matchcat1, cat2[idx2]))

	return vstack((matchcat, unmatchcat))

def match2duncan(cat, sep):
	mindec, maxdec = np.min(cat['DEC']), np.max(cat['DEC'])
	duncanpath = '/home/graysonpetter/ssd/Dartmouth/data/photozs/duncan_allsky/'
	import glob
	photfiles = glob.glob(duncanpath + 'dec*')
	outcat = Table()
	for photfile in photfiles:
		photmin = float(photfile.split('dec_')[1].split('_')[0])
		photmax = float(photfile.split('dec_')[1].split('_')[1].split('.')[0])
		if (photmin > maxdec) | (photmax < mindec):
			continue
		else:
			phottab = Table.read(photfile)
			thiscat = cat[np.where((cat['DEC'] >= photmin) & (cat['DEC'] <= photmax))]

			photidx, catidx = match_coords((phottab['ra'], phottab['dec']), (thiscat['RA'], thiscat['DEC']), max_sep=sep, symmetric=False)
			thiscat = thiscat[catidx]
			thiscat['zphot'] = phottab['zphot'][photidx]
			thiscat['zphot_err'] = phottab['zphot_err'][photidx]
			outcat = vstack((outcat, thiscat))
	return outcat


