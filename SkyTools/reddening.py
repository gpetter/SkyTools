from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import astropy.units as u

av_over_ebv = {'decam_u': 3.995, 'decam_g': 3.214, 'decam_r': 2.165, 'decam_i': 1.592, 'decam_z': 1.211, 'decam_y': 1.064}


def ebv_at_coords(ras, decs):

    coords = SkyCoord(ras*u.deg, decs*u.deg, frame='icrs')
    return SFDQuery()(coords)

def dered_mags(mags, ebvs, filter):
    av = ebvs * av_over_ebv[filter]
    return mags - av

