import healpy as hp
import numpy as np
from scipy import stats
from astropy.table import Table

def filled_map(nside, fillval=0.):
	"""

	Parameters
	----------
	nside
	fillval

	Returns
	-------

	"""
	if fillval == 0:
		return np.zeros(hp.nside2npix(nside=nside))
	else:
		return fillval * np.ones(hp.nside2npix(nside=nside))

def proper_ud_grade_mask(mask, newnside):
	"""
	take a binary mask and properly downgrade it to a lower resolution, expanding the mask to cover all pixels which
	touch bad pixels in the high-res map
	Parameters
	----------
	mask
	newnside

	Returns
	-------

	"""
	mask_lowres_proper = hp.ud_grade(np.copy(mask).astype(float), nside_out=newnside).astype(float)
	mask_lowres_proper = np.where(mask_lowres_proper == 1., True, False).astype(bool)
	return mask_lowres_proper


def rotate_mask(mask, coords=['C', 'G'], conservative=True):
	"""

	https://stackoverflow.com/questions/68010539/healpy-rotate-a-mask-
	together-with-the-map-in-hp-ma-vs-separately-produce-di#

	:param mask: healpix mask, 1/True==observed
	:param coords: G: galactic, C: celestial, E: ecliptic
	:param conservative: only keep good pixels where 100% was observed
	:return:
	"""
	mask = np.array(mask, dtype=bool)
	transform = hp.Rotator(coord=coords)

	m = hp.ma(np.arange(len(mask), dtype=np.float32))
	m.mask = mask

	# if you use a float mask and rotate, healpix interpolates near border
	if conservative:
		rotated_mask = transform.rotate_map_pixel(m.mask)
		# round the interpolated values to 0 or 1
		return np.around(rotated_mask)
	# otherwise, healpix just rotates the binary mask without interpolation, might be unsafe
	else:
		return transform.rotate_map_pixel(m.mask)


def healpix_density_map(cat_or_coords, nsides, weights=None, deg2=False, nest=False):
	"""
	for a given source list with ras and decs, create a healpix map of source density for a given pixel size
	Parameters
	----------
	cat_or_coords: astropy table including RA, DEC columns OR tuple of (lon, lat)
	nsides
	weights
	deg2
	nest

	Returns
	-------

	"""
	if type(cat_or_coords) == Table:
		lons, lats = cat_or_coords['RA'], cat_or_coords['DEC']
	else:
		lons, lats = cat_or_coords
	# convert coordinates to healpix pixels
	pix_of_sources = hp.ang2pix(nsides, lons, lats, lonlat=True, nest=nest)
	# number of pixels for healpix map with nsides
	npix = hp.nside2npix(nsides)
	# count number of sources in each pixel
	density_map = np.bincount(pix_of_sources, minlength=npix, weights=weights)
	if deg2:
		pixarea = hp.nside2pixarea(nsides, degrees=True)
		density_map = np.array(density_map, dtype=float) / pixarea

	return density_map


def healpix_average_in_pixels(cat_or_coords, nsides, values):
	"""
	Make a healpix map by taking average value associated with a list of coordinates
	Parameters
	----------
	cat_or_coords: astropy table including RA, DEC columns OR tuple of (lon, lat)
	nsides
	values

	Returns
	-------

	"""
	if type(cat_or_coords) == Table:
		lons, lats = cat_or_coords['RA'], cat_or_coords['DEC']
	else:
		lons, lats = cat_or_coords
	# convert coordinates to healpix pixels
	pix_of_sources = hp.ang2pix(nsides, lons, lats, lonlat=True)
	# number of pixels for healpix map with nsides
	npix = hp.nside2npix(nsides)
	# average in each pixel is weighted sum divided by total sum
	avg_map = np.bincount(pix_of_sources, weights=values, minlength=npix) / np.bincount(pix_of_sources, minlength=npix)

	return avg_map


def healpix_median_in_pixels(cat_or_coords, nsides, values):
	"""
	Make a healpix map by taking median value associated with a list of coordinates
	Parameters
	----------
	cat_or_coords: astropy table including RA, DEC columns OR tuple of (lon, lat)
	nsides
	values

	Returns
	-------

	"""
	if type(cat_or_coords) == Table:
		lons, lats = cat_or_coords['RA'], cat_or_coords['DEC']
	else:
		lons, lats = cat_or_coords
	# convert coordinates to healpix pixels
	pix_of_sources = hp.ang2pix(nsides, lons, lats, lonlat=True)
	# number of pixels for healpix map with nsides
	npix = hp.nside2npix(nsides)
	medmap = np.array(stats.binned_statistic(x=pix_of_sources, values=values,
							statistic='median', bins=np.linspace(-0.5, npix - 0.5, npix + 1))[0])

	return medmap


def ud_grade_median(m, nside_out, reorder=True):
	"""
	Downgrade a map and take the median of the child pixels, rather than the default mean of healpy.ud_grade
	Parameters
	----------
	m
	nside_out
	reorder

	Returns
	-------

	"""
	if reorder:
		m = hp.reorder(m, 'RING', 'NEST')

	nside_in = hp.npix2nside(len(m))
	npix_in = hp.nside2npix(nside_in)
	npix_out = hp.nside2npix(nside_out)

	rat2 = npix_in // npix_out
	mr = m.reshape(npix_out, rat2)
	map_out = np.nanmedian(mr, axis=1)
	map_out[~np.isfinite(map_out)] = np.nan

	if reorder:
		map_out = hp.reorder(map_out, 'NEST', 'RING')

	return map_out


def healpixels2lon_lat(hpmap):
	"""
	Pass healpix map and get longitude, latitudes of every pixel
	Parameters
	----------
	hpmap

	Returns
	-------

	"""
	return hp.pix2ang(hp.npix2nside(len(hpmap)), np.arange(len(hpmap)), lonlat=True)



def fractional_overdensity_map(hpmap):
	"""

	Parameters
	----------
	hpmap: array, map of counts in cells, is nan where masked

	Returns
	-------
	overdensity delta map
	"""
	mapmean = np.nanmean(hpmap)
	return (hpmap - mapmean) / mapmean

def coords2mapvalues(cat_or_coords, map, nest=False):
	"""
	Get healpix map values at specfied coordinates
	Parameters
	----------
	cat_or_coords: astropy table including RA, DEC columns OR tuple of (lon, lat)
	map: array

	Returns
	-------
	values of map in pixels corresponding to coordinates

	"""
	if type(cat_or_coords) == Table:
		lons, lats = cat_or_coords['RA'], cat_or_coords['DEC']
	else:
		lons, lats = cat_or_coords
	nside = hp.npix2nside(len(map))
	pix = hp.ang2pix(nside, lons, lats, lonlat=True, nest=nest)
	return map[pix]

# take coordinates from 2 surveys with different footprints and return indices of sources within the overlap of both
def match_footprints(testsample, reference_sample, nside=32):
	ras1, decs1 = testsample[0], testsample[1]
	ras2, decs2 = reference_sample[0], reference_sample[1]

	footprint1_pix = hp.ang2pix(nside=nside, theta=ras1, phi=decs1, lonlat=True)
	footprint2_pix = hp.ang2pix(nside=nside, theta=ras2, phi=decs2, lonlat=True)
	commonpix = np.intersect1d(footprint1_pix, footprint2_pix)
	commonfootprint = np.zeros(hp.nside2npix(nside))
	commonfootprint[commonpix] = 1
	idxs = np.where(commonfootprint[footprint1_pix])
	#idxs2 = np.where(commonfootprint[footprint2_pix])

	#sample1out = (ras1[idxs1], decs1[idxs1])
	#sample2out = (ras2[idxs2], decs2[idxs2])

	#return (sample1out, sample2out)

	return idxs


def change_coord(m, coord):
	""" Change coordinates of a HEALPIX map

	Parameters
	----------
	m : map or array of maps
	  map(s) to be rotated
	coord : sequence of two character
	  First character is the coordinate system of m, second character
	  is the coordinate system of the output map. As in HEALPIX, allowed
	  coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

	Example
	-------
	The following rotate m from galactic to equatorial coordinates.
	Notice that m can contain both temperature and polarization.
	"""
	# Basic HEALPix parameters
	npix = m.shape[-1]
	nside = hp.npix2nside(npix)
	ang = hp.pix2ang(nside, np.arange(npix))

	# Select the coordinate transformation
	rot = hp.Rotator(coord=reversed(coord))

	# Convert the coordinates
	new_ang = rot(*ang)
	new_pix = hp.ang2pix(nside, *new_ang)

	return m[..., new_pix]


def masked_smoothing(U, fwhm_arcmin=15):
	"""
	Credit to David at https://stackoverflow.com/questions/50009141/smoothing-without-filling-missing-values-with-zeros
	Smooth a map where a mask is applied, ignoring the masked region
	Parameters
	----------
	U
	fwhm_arcmin

	Returns
	-------

	"""
	V = U.copy()
	maskedidx = (U != U)
	V[maskedidx] = 0
	rad = fwhm_arcmin * np.pi / (180. * 60.)
	VV = hp.smoothing(V, fwhm=rad)
	W = 0 * U.copy() + 1
	W[U != U] = 0
	WW = hp.smoothing(W, fwhm=rad)
	smoothed = VV / WW
	smoothed[maskedidx] = np.nan
	return smoothed



# create a healpix mask from square telescope pointings
# coords (lon, lat) tuple
# pointing_side = length in degrees of pointing on a side
def mask_from_pointings(coords, nside, pointing_radius=None, pointing_side=None, fill_values=None):
	# initialize mask
	mask = np.zeros(hp.nside2npix(nside))



	lons, lats = coords
	lons, lats = np.atleast_1d(lons), np.atleast_1d(lats)
	# can pass values to set mask to, otherwise just set to 1
	if fill_values is None:
		fill_values = np.ones(len(lons))

	if pointing_side is not None:
		half_side = pointing_side / 2.
		# for each coordinate pair
		for j in range(len(lons)):
			# modify ra by cos(delta)
			# wrap around sphere by % 360
			lower_lon = (lons[j] - half_side / np.cos(np.radians(lats[j]))) % 360.
			higher_lon = lons[j] + half_side / np.cos(np.radians(lats[j]))
			lower_lat = lats[j] - half_side
			higher_lat = lats[j] + half_side
			# if close to the north pole, query a strip instead of a polygon
			if lats[j] >= (90. - half_side):
				strippix = hp.query_strip(nside, np.radians(lower_lon), np.radians(higher_lon))
				pixlon, pixlat = hp.pix2ang(nside, strippix, lonlat=True)
				strippix = strippix[np.where(pixlat > (90. - half_side))]
				mask[strippix] = fill_values[j]
				continue
			# same for south pole
			if lats[j] <= (-90. + half_side):
				strippix = hp.query_strip(nside, np.radians(lower_lon), np.radians(higher_lon))
				pixlon, pixlat = hp.pix2ang(nside, strippix, lonlat=True)
				strippix = strippix[np.where(pixlat < (-90. + half_side))]
				mask[strippix] = fill_values[j]
				continue
			# otherwise query a square on the sphere
			lowerleft = hp.ang2vec(lower_lon, lower_lat, lonlat=True)
			lowerright = hp.ang2vec(higher_lon, lower_lat, lonlat=True)
			upperleft = hp.ang2vec(lower_lon, higher_lat, lonlat=True)
			upperright = hp.ang2vec(higher_lon, higher_lat, lonlat=True)
			# set pixels in square = 1
			mask[hp.query_polygon(nside, np.array([lowerleft, upperleft, upperright, lowerright]))] = fill_values[j]
	elif pointing_radius is not None:
		vecs = hp.ang2vec(lons, lats, lonlat=True)
		for j in range(len(lons)):

			mask[hp.query_disc(nside, vecs[j], radius=np.radians(pointing_radius))] = fill_values[j]

	return mask






def query_annulus_coord(nside, coord, inner_rad, outer_rad):
	"""
	Query the pixels in an annulus between r_min and r_max away from a central point
	Parameters
	----------
	nside
	coord
	inner_rad
	outer_rad

	Returns
	indices within the annulus

	"""
	def query_annulus(nside, vec, inner_rad, outer_rad):
		# helper to translate from coordinate tuple to vector
		big_disc = hp.query_disc(nside, vec, np.radians(outer_rad))
		small_disc = hp.query_disc(nside, vec, np.radians(inner_rad), inclusive=True)
		# find indices which are in big_disc but not small_disc
		annulus = np.setdiff1d(big_disc, small_disc, assume_unique=True)
		return annulus
	vec = hp.ang2vec(coord[0], coord[1], lonlat=True)
	return query_annulus(nside, vec, inner_rad, outer_rad)

def query_disc_coord(nside, coord, radius_deg):
	"""
	Default healpy query_disc requires you to pass a 3-vector, this lets you pass a longitude, latitude instead
	Parameters
	----------
	nside
	coord
	radius_deg

	Returns
	-------

	"""
	return hp.query_disc(nside, hp.ang2vec(coord[0], coord[1], lonlat=True), np.radians(radius_deg))

def retract_mask(mask, orders_below=1, nside_out=None, all_neighbors=True):
	"""
	Take a mask defining a survey and retreat the good (observed) area in order to be conservative and not include
	areas unobserved near the edges of the mask

	Works by downgrading resolution, taking average of higher order pixels, upgrading back to higher resolution,
	and rejecting pixels between 0 and 1
	Parameters
	----------
	mask: array
	equal to 1 where good (area observed by survey) and 0 where bad

	orders_below: int
	number of orders to degrade the mask by before upscaling again, bigger number means mask will be retreated further

	nside_out: int
	resolution of map output, can upscale output as much as you want

	all_neighbors: bool
	If True, mask all neighboring pixels for each pixel along the border of the footprint
	Returns
	-------
	mask: array
	mask where good area has been retreated to be conservative with edge effects

	"""

	# original NSIDE and order of input mask
	nside = hp.npix2nside(len(mask))
	original_order = hp.nside2order(nside)
	# order to downgrade mask to
	low_order = original_order - orders_below
	# convert to float so that ud_grade takes an average of the low
	mask = np.array(mask, dtype=float)
	# downgrade resolution taking average, and upgrading back to original resolution
	newmask = hp.ud_grade(hp.ud_grade(mask, 2**low_order), nside)

	# if masking all neighbors to each border pixel
	if all_neighbors:
		# get pixels which are in original footprint, but bordering a bad pixel
		borderpix = np.where((newmask > 0) & (newmask < 1) & (mask == 1.))
		# convert these to coordinates for get_all_neighbors
		ls, bs = hp.pix2ang(nside, borderpix)
		# mask pixels
		for j in range(len(ls)):
			newmask[hp.get_all_neighbours(nside, ls[j], bs[j])] = 0.
	else:
		# mask pixels near boundary of footprint where average in low resolution is 0 < x < 1
		newmask[np.where(newmask < 1)] = 0

	# upscale beyond original NSIDE if desired
	if (nside_out is None) or (nside_out < nside):
		nside_out = nside
	newmask = proper_ud_grade_mask(newmask, nside_out)

	return np.array(newmask, dtype=int)

def inmask(cat_or_coords, mask, return_bool=False):
	"""

	Parameters
	----------
	cat_or_coords:
	mask: array
	binary mask where 1 is good, 0 is bad
	return_bool: bool
	If True, return array of booleans same length as coordinate array, True where inside mask
	Else, return integer indices of coordinates which fall in mask

	Returns
	-------
	indices where coordinates are within the mask, or boolean array for passing the mask

	"""
	if type(cat_or_coords) == Table:
		lons, lats = cat_or_coords['RA'], cat_or_coords['DEC']
	else:
		lons, lats = cat_or_coords

	# convert coordinates to healpixels
	coordpix = hp.ang2pix(hp.npix2nside(len(mask)), lons, lats, lonlat=True)
	if return_bool:
		good = np.array((mask[coordpix] == 1), dtype=bool)
	else:
		# get indices of those passing the mask
		good = np.where(mask[coordpix] == 1)[0]
	return good

def cat_in_mask(cat, mask):
	"""
	Return subset of catalog which lies inside the window
	Parameters
	----------
	cat
	mask

	Returns
	-------

	"""
	good = inmask(cat_or_coords=cat, mask=mask)
	return cat[good]


def subtract_dipole(
		m, dipole, nest=False, bad=hp.UNSEEN, gal_cut=0, fitval=False, copy=True, verbose=True
):
	input_ma = hp.pixelfunc.is_ma(m)
	m = hp.pixelfunc.ma_to_array(m)
	m = np.array(m, copy=copy)
	npix = m.size
	nside = hp.npix2nside(npix)
	if nside > 128:
		bunchsize = npix // 24
	else:
		bunchsize = npix

	for ibunch in range(npix // bunchsize):
		ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
		ipix = ipix[(m.flat[ipix] != bad) & (np.isfinite(m.flat[ipix]))]
		x, y, z = hp.pix2vec(nside, ipix, nest)
		m.flat[ipix] -= dipole[0] * x
		m.flat[ipix] -= dipole[1] * y
		m.flat[ipix] -= dipole[2] * z
	# m.flat[ipix] -= mono

	lon, lat = hp.rotator.vec2dir(dipole, lonlat=True)
	amp = np.sqrt((dipole * dipole).sum())

	if hp.pixelfunc.is_ma:
		m = hp.pixelfunc.ma(m)
	if fitval:
		return m, mono, dipole
	else:
		return m