
from scipy import stats
import healpy as hp
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from . import healpixhelper as myhp



# draw random points uniformly on the sphere
def uniform_sphere(n, density=False, lat_range=(-90., 90.), lon_range=(0, 360.)):
	"""

	Parameters
	----------
	n: int
	Either total number of random points to draw, or density/deg^2
	density: bool
	If True, n is a number density/deg^2,
	Else it's a total number of points
	lat_range: tuple
	lon_range: tuple
	ranges of sphere to populate with randoms in degrees


	Returns
	-------
	A tuple of random longitudes and latitudes evenly dispersed on the sphere

	"""
	min_lat, max_lat = lat_range
	min_lon, max_lon = lon_range


	# calculate total number given the surface density
	if density:
		# number of radians around the equator
		radians_lon_range = np.radians(max_lon - min_lon)
		# total area covered by the ranges
		area_sterad = radians_lon_range * (np.sin(np.radians(max_lat)) - np.sin(np.radians(min_lat)))

		# convert density to total number on the sphere by converting steradian to deg^2, multiplying
		n = int(n * area_sterad * (180. / np.pi) ** 2)
	# uniformly randomly generate points
	randlons = np.random.uniform(min_lon, max_lon, n)
	randlats = np.arcsin(np.random.uniform(np.sin(np.radians(min_lat)), np.sin(np.radians(max_lat)), n)) * 180. / np.pi
	return randlons, randlats



def generate_random_points_from_mangle(manglename, npoints):
	manglemask = pymangle.Mangle('mangle/%s_footprint.ply' % manglename)
	ra_rand, dec_rand = manglemask.genrand(npoints)
	return ra_rand, dec_rand


def gen_rand_points_from_healpix(ras, decs, npoints, nside=32):
	pix_with_sources = hp.ang2pix(nside, ras, decs, lonlat=True)
	hpmap = np.zeros(hp.nside2npix(nside))
	hpmap[pix_with_sources] = 1

	# calculate how many points needed distributed over whole sky to result in npoints over observed region
	npoints_whole_sky = round(npoints * len(hpmap)/len(np.where(hpmap)[0]))

	randras, randdecs = uniform_sphere(npoints_whole_sky)
	# get those which fall inside observed region
	idxs_in_mask = (hpmap[hp.ang2pix(nside, randras, randdecs, lonlat=True)] == 1)
	randras, randdecs = randras[idxs_in_mask], randdecs[idxs_in_mask]

	return randras, randdecs


def random_points_in_hp_mask(mask, npoints):

	masknside = hp.npix2nside(len(mask))
	# find the range of latitude the mask is defined to make random generation cheaper
	pixlon, pixlat = myhp.healpixels2lon_lat(mask)
	lat_range = np.min(pixlat), np.max(pixlat)
	skyfraction = len(np.where(mask > 0)[0])/len(mask)
	randlons, randlats = uniform_sphere(int(1/skyfraction * npoints), lat_range=lat_range)

	inmask = np.where(mask[hp.ang2pix(masknside, randlons, randlats, lonlat=True)] > 0)

	return randlons[inmask], randlats[inmask]




def remove_bright_star_regions(ras, decs):
	bright_star_mask = pymangle.Mangle('footprints/allsky_bright_star_mask_pix.ply')
	in_mask = bright_star_mask.contains(ras, decs)

	ras = ras[np.logical_not(in_mask)]
	decs = decs[np.logical_not((in_mask))]

	return ras, decs

def remove_bad_field_regions(ras, decs):
	bad_field_mask = pymangle.Mangle('footprints/badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply')
	in_mask = bad_field_mask.contains(ras, decs)

	ras = ras[np.logical_not(in_mask)]
	decs = decs[np.logical_not((in_mask))]

	return ras, decs

def eBOSS_randoms(nrandoms):
	eBOSS_mask = pymangle.Mangle('footprints/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')

	ra_rand, dec_rand = eBOSS_mask.genrand(nrandoms)

	centerpost_mask = pymangle.Mangle('footprints/centerpost_mask_eboss_DR16_new.ply')
	in_centerpost_mask = centerpost_mask.contains(ra_rand, dec_rand)

	# remove randoms inside centerpost mask
	ra_rand = ra_rand[np.logical_not(in_centerpost_mask)]
	dec_rand = dec_rand[np.logical_not(in_centerpost_mask)]

	ra_rand, dec_rand = remove_bright_star_regions(ra_rand, dec_rand)

	t = Table()
	t['RA'] = np.array(ra_rand, dtype=np.float)
	t['DEC'] = np.array(dec_rand, dtype=np.float)

	t.write('random_catalogs/ebossrands.fits', format='fits', overwrite=True)

	return ra_rand, dec_rand


def BOSS_randoms(nrandoms, writetable=False):
	BOSS_mask = pymangle.Mangle('footprints/sdss_boss_dipomp.ply')
	ra_rand, dec_rand = BOSS_mask.genrand(nrandoms)

	if writetable:
		t = Table()
		t['RA'] = np.array(ra_rand, dtype=np.float)
		t['DEC'] = np.array(dec_rand, dtype=np.float)

		t.write('random_catalogs/bossrands.fits', format='fits', overwrite=True)

	return ra_rand, dec_rand

def dr8_imaging_randoms(nrandoms, writetable=False):
	dr8_mask = pymangle.Mangle('footprints/sdss_dr8_footprint_dipomp.ply')

	ra_rand, dec_rand = dr8_mask.genrand(nrandoms)

	ra_rand, dec_rand = remove_bright_star_regions(ra_rand, dec_rand)

	ra_rand, dec_rand = remove_bad_field_regions(ra_rand, dec_rand)


	map4stack = hp.read_map('maps/smoothed_masked_planck.fits', dtype=np.single)
	ls, bs = stacking.equatorial_to_galactic(ra_rand, dec_rand)
	outsidemaskbool = (map4stack[hp.ang2pix(2048, ls, bs, lonlat=True)] != hp.UNSEEN)
	planckidxs = np.where(outsidemaskbool)

	ra_rand, dec_rand = ra_rand[planckidxs], dec_rand[planckidxs]


	if writetable:
		t = Table()
		t['RA'] = np.array(ra_rand, dtype=np.float)
		t['DEC'] = np.array(dec_rand, dtype=np.float)

		t.write('random_catalogs/dr8rands.fits', format='fits', overwrite=True)

	return ra_rand, dec_rand


def subsample_random_cat(sample, cap, number):

	cat = fits.open('catalogs/lss/%s/randoms_%s_w_weights.fits' % (sample, cap))[1].data
	cat = cat[np.where(cat['COMP_BOSS'] > 0)]
	inst_weights = 1/(cat['WEIGHT_SYSTOT'] * cat['WEIGHT_CP'] * cat['WEIGHT_NOZ'])
	comp = cat['COMP_BOSS']
	totweights = comp * inst_weights
	norm_weights = totweights/np.sum(totweights)
	bootidxs = np.random.choice(len(cat), int(number), replace=False, p=norm_weights)
	newtab = Table(cat[bootidxs])
	newtab.write('catalogs/lss/%s/randoms_%s_subsampled.fits' % (sample, cap), format='fits', overwrite=True)


def fullsky_eboss_randoms(a):
	ngc_randoms = fits.open('catalogs/lss/eBOSS_randoms_NGC_comov.fits')[1].data
	sgc_randoms = fits.open('catalogs/lss/eBOSS_randoms_SGC_comov.fits')[1].data
	ngc_idxs = np.random.choice(len(ngc_randoms), a*218209, replace=False)
	sgc_idxs = np.random.choice(len(sgc_randoms), a*125499, replace=False)

	ngctab = Table(ngc_randoms[ngc_idxs])
	sgctab = Table(sgc_randoms[sgc_idxs])

	bothtab = vstack((ngctab, sgctab))
	return bothtab





