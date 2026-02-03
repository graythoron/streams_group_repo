import numpy as np

import astropy.units as u
import astropy.coordinates as coords
from scipy.stats import binned_statistic

from arviz import hdi, kde

from compute_bound_fracs import compute_iterative_boundness
from plummer_fitting import fit_plummer

def radius_mask(ra, dec, rad):
    mask = np.sqrt((ra)**2+(dec)**2) <= rad
    return mask

def density_median(sample, circular=False):
    sample_kde = kde(sample, circular=circular)
    return sample_kde[0][np.lexsort(sample_kde)[-1]]

def to_SkyCoords(xyz, vels=None, coordsys='spherical'):
    if vels is None:
        vels_ = np.zeros(xyz.shape)*u.km/u.s
    else:
        vels_ = vels
    coordinates=coords.SkyCoord(
        x   =   xyz.T[0], y   =   xyz.T[1], z   =   xyz.T[2],
        v_x = vels_.T[0], v_y = vels_.T[1], v_z = vels_.T[2],
        frame='galactocentric'
    )
    if coordsys == 'ICRS':
        coordinates.transform_to(coords.ICRS)
    if coordsys in ['ICRS', 'spherical']:
        coordinates.representation_type=coords.SphericalRepresentation
        coordinates.differential_type=coords.SphericalDifferential
    return coordinates

def bound_fraction(dark_pos, dark_vel, dark_masses, star_pos, star_vel, star_masses, center_pos=[], center_vel=[], methods=None):
    '''
    Computes the bound fraction of a progenitor via recursive Agama multipole fits. 
    A particle is considered bound if kinetic + potential < 0.

    Parameters
    ----------
    dark_pos : np.array-like with shape (N, 3). 
        Dark matter particle positions, (kpc but Astropy units can be included).
    dark_vel : np.array-like with shape (N, 3). 
        Dark matter particle velocities, (km/s but Astropy units can be included). 
    dark_mass : np.array-like with shape (N, 3). 
        Dark matter particle masses, (M_sun but Astropy units can be included). 
    star_pos : np.array-like with shape (N, 3). 
        Star particle positions, (kpc but Astropy units can be included).
    star_vel : np.array-like with shape (N, 3). 
        Star particle velocities, (km/s but Astropy units can be included). 
    star_mass : np.array-like with shape (N, 3). 
        Star particle masses, (M_sun but Astropy units can be included). 
    center_pos : np.array-like with shape (3,), optional.
        The center of the progenitor (kpc but Astropy units can be included).
        Default is empty list-type
    center_vel : np.array-like with shape (3,), optional.
        The velocity of the progenitor (km/s but Astropy units can be included).
        Default is empty list-type
    methods : list or integer, optional.
        The methods for calculating the position and velocity of the center.
        Default value is None, which supplies four combinations of centering:
            Centering the position based on a KDE of [stars] or [both] stars and dark matter.
            Centering the velocity with [True] and without [False] the KDE.
        Setting methods to an integer selects a method from this list, they are 
            0: ['star', True], 1: ['star', False], 2: ['both', True], 3:['both', False]
    
    Returns
    -------
    results : dict
        'n_part_dm' : int
            The number of dark matter particles
        'n_part' : int
            The number of star particles
        'center_pos' : np.array-like (3,)
            The position of the progenitor using the method that resulted in the highest 
            bound fraction.
        'center_vel' : np.array-like (3,)
            The velocity of the progenitor using the method that resulted in the highest 
            bound fraction.
        'f_bound_part' : float
            The fraction of bound star particles of the progenitor using the method 
            that resulted in the highest bound fraction.
        'f_bound_mass' : float
            The fraction of bound stellar mass of the progenitor using the method 
            that resulted in the highest bound fraction.
        'f_bound_part_dm' : float
            The fraction of bound dark matter particles of the progenitor using the method 
            that resulted in the highest bound fraction.
        'f_bound_mass_dm' : float
            The fraction of bound dark matter mass of the progenitor using the method 
            that resulted in the highest bound fraction.
        'star_bound' : np.array-like (N,)
            Whether a star particle is bound [1] or unbound [0]
        'dm_bound' : np.array-like (N,)
            Whether a dark matter particle is bound [1] or unbound [0]
        'center_pos_{n}' : np.array-like (3,)
            The position of the progenitor using the nth method.
        'center_vel_{n}' : np.array-like (3,)
            The velocity of the progenitor using the nth method.
        'f_bound_part_{n}' : float
            The fraction of bound star particles of the progenitor using the nth method.
        'f_bound_mass_{n}' : float
            The fraction of bound stellar mass of the progenitor using the nth method.
        'f_bound_part_dm_{n}' : float
            The fraction of bound dark matter particles of the progenitor using the 
            nth method.
        'f_bound_mass_dm_{n}' : float
            The fraction of bound dark matter mass of the progenitor using the 
            nth method.
    '''
    assert len(dark_pos) == len(dark_vel)
    assert len(dark_pos) == len(dark_masses)
    assert len(star_pos) == len(star_vel)
    assert len(star_pos) == len(star_masses)

    results = {}
    results['n_part_dm'] = [len(dark_masses)]
    results['n_part'] = [len(star_masses)]

    results_list = []
    # Choosing methods
    if (center_vel == []) &(center_pos == []) & (methods == None):
        methods = [['star', True], ['star', False], ['both', True], ['both', False]]
    elif (center_pos == []) & (methods == None):
        methods = [['star', True], ['both', True]]
    elif (center_vel == []) & (methods == None):
        methods = [['star', True], ['star', False]]
    elif (methods == None):
        methods = [['star', True]]
    elif (type(methods) == int):
        methods = [['star', True], ['star', False], ['both', True], ['both', False]][methods]

    #Several different centering methods are used in order to find the highest bound fraction
    for i in range(len(methods)):
        results0 = compute_iterative_boundness(
                dark_pos, dark_vel, dark_masses,
                star_pos, star_vel, star_masses, 
                center_on=methods[i][0], center_vel_with_KDE=methods[i][1],
                center_position=center_pos, center_velocity=center_vel
        )

        if len(methods)>1:
            results[f'center_pos_{i}'] = results0[1]
            results[f'center_vel_{i}'] = results0[2]
            results[f'f_bound_part_{i}'] = [np.mean(results0[0][1]==1)]
            results[f'f_bound_mass_{i}'] = [np.sum(star_masses[results0[0][1]==1])/np.sum(star_masses)]
            results[f'f_bound_part_dm_{i}'] = [np.mean(results0[0][0]==1)]
            results[f'f_bound_mass_dm_{i}'] = [np.sum(dark_masses[results0[0][0]==1])/np.sum(dark_masses)]
        
        results_list.append(results0)
        
    #Calculating the largest bound fraction of methods used
    m_i = np.argmax([np.mean(i[0][1]==1) for i in results])
    
    results['center_pos'] = results[m_i][1]
    results['center_vel'] = results[m_i][2]  

    results['f_bound_part'] = [np.mean(results[m_i][0][1]==1)]
    results['f_bound_mass'] = [np.sum(star_masses[results[m_i][0][1]==1])/np.sum(star_masses)]
    results['f_bound_part_dm'] = [np.mean(results[m_i][0][0]==1)]
    results['f_bound_mass_dm'] = [np.sum(dark_masses[results[m_i][0][0]==1])/np.sum(dark_masses)]

    results['star_bound'] = results[m_i][0][1]
    results['dm_bound'] = results[m_i][0][0]

    return results

def plummer_profile_fit(
        star_pos, star_vel, star_bound, star_masses, center_pos, star_metals=[], 
        coord_system='spherical', radius=1*u.kpc, num_radii=1, output_file="plummer_fit_out.h5", 
        return_derived_quants = True
    ):
    '''
    Computes the plummer profile of a progenitor projected on to sky coordinates. 
    The plummer profile fitting is split across 5 threads and 500 walkers with 100 
    to each thread with 800 steps and 200 steps of burn-in.

    Parameters
    ----------
    star_pos : np.array-like with shape (N, 3) with astropy units.
        The 3D positions of the star particles [N] with the origin at the GC. (kpc) 
    star_vel : np.array-like with shape (N, 3) with astropy units.
        The 3D velocities of the star particles comoving with the origin at the GC. (km/s)
    star_bound : np.array-like with shape (N,)
        A boolean or [0, 1] list of whether a star particle is bound or not.
    star_masses : np.array-like with shape (N,) with astropy units.
        The mass of the star particles (M_sun)
    center_pos : np.array-like with shape (3,) with astropy units.
        The 3D position of the center of the progenitor. (kpc)
    star_metals : np.array-like with shape (M, N), optional.
        The metal abundances [M] of the star particles [N]
        Default : []
    coord_system : string, optional. 
        The coordinate system in which to project the progenitor.
        Default : 'spherical'
        Options : 'ICRS'
                  'spherical'
    radius : float with astropy units. 
        The characteristic radius of the progenitor
        Default: 1*u.kpc, 
    num_radii : int
        The number of characteristic radii in which to include star particles.
        If num_radii = 0, all bound stars will instead be included.
        Default : 1
    output_file : str
        The output path for the output from the plummer profile fitting proceedure 
        Default : "plummer_fit_out.h5"
    return_derived_quants : bool
        In addition to the fit, the results dictionary will return derived quantities
        such as ang_dif and r_half.
        Default : True

    Returns
    -------
    results : dict
        'ra' : float with astropy units.
            The right ascension of the progenitor. (deg)
        'dec' : float with astropy units.
            The declination of the progenitor. (deg)
        'pm' : np.array-like (2,) with astropy units.
            The proper motion of the masked particles. (mas/yr)
        'rad_vel' : float with astropy units.
            The radial velocity of the masked particles. (km/s)
        'lon' : float with astropy units.
            The longitude origin of the plummer profile with a coordinate system
            centered on the progenitor. (deg)
        'lon_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on the longitude. (deg)
        'lat' : float with astropy units.
            The latitude origin of the plummer profile with a coordinate system
            centered on the progenitor. (deg)
        'lat_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on the latitude. (deg)
        'extension' : float with astropy units.
            The semi-major axis of the plummer profile. (deg)
        'extension_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on the extension. (deg)
        'ellipticity' : float.
            The ellipticity of the plummer profile. 
            0 is a circle (not an ellipse), 
            1 is two paralell lines (so ellipitical it's only an ellipse at infinity)
        'ellipticity_err' : np.array-like (2,)
            The one sigma error [below, above] on the ellipticity.
        'position_angle' :  float with astropy units.
            The angle [N->E] of rotation of the plummer profile on the sky. (deg)
        'position_angle_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on the position angle. (deg) 
        'fitted_metals' : np.array-like (M,)
            The median values of the metalicities of the fitted particles. 
        'r_half' : float with astropy units. 
            The physical azimuthally averaged r_half. (pc)
            Optional with return_derived_quants = False
        'r_half_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on r_half. (deg)
            Optional with return_derived_quants = False
        'ang_dif' : float with astropy units. 
            The angle between the proper motion and the postion angle. (deg)
            Optional with return_derived_quants = False
        'ang_dif_err' : np.array-like (2,) with astropy units.
            The one sigma error [below, above] on ang_dif. (deg)
            Optional with return_derived_quants = False
    '''
    assert len(star_pos) == len(star_vel)
    assert len(star_pos) == len(star_bound)
    assert len(star_pos) == len(star_masses)
    if len(star_metals.T) > 0:
        assert len(star_pos) == len(star_metals)

    #Projecting the center coordinates to skycoord so we can use them later
    halo_skycoords = to_SkyCoords(center_pos, coordsys=coord_system)

    #Projecting Radius
    # given by half mass radius (radius) * multiplication factor (num_radii)
    rad_proj_expand = np.arctan(num_radii*radius/halo_skycoords.distance).to(u.degree).value

    #Projecting the stars on to the sky
    star_skycoords = to_SkyCoords(star_pos, star_vel, coordsys=coord_system)
    #Rotating the coordinate system to originate at the center of the progenitor
    star_skycoords = star_skycoords.transform_to(coords.SkyOffsetFrame(origin=halo_skycoords))
    star_skycoords.representation_type=coords.SphericalRepresentation
    star_skycoords.differential_type=coords.SphericalDifferential
    skyra = star_skycoords.lon.value
    skydec = star_skycoords.lat.value

    #Determining the mask 
    if num_radii == 0:
        #If there's no radius, we assume they're after the bound particles
        star_mask = np.bool(star_bound)
    else:
        star_mask = np.sqrt((skyra)**2+(skydec)**2) <= rad_proj_expand

    if np.any(star_mask): 
        #If there are stars in the mask, it will be fitted
        print('Running MCMC...')

        sampler, __ = fit_plummer(
                skyra[star_mask], skydec[star_mask], 
                nsteps=800, nthreads=5, nwalkers=500,
                outfile=output_file, restart=False
        )

        sample = sampler.get_chain()
        live_samples = sample[0, :, 0] != sample[-1, :, 0]

        if np.any(live_samples):
            #Formatting the samples so we can extract information about each parameter
            samples = sample[:, live_samples, :]
            sampling = samples.reshape(samples.shape[0]*samples.shape[1], 5)

            #Calculating the median
            lon_result = density_median(sampling.T[0])
            lat_result = density_median(sampling.T[1])
            ext_result = density_median(sampling.T[2])
            ell_result = density_median(sampling.T[3])
            ang_result = (density_median(sampling.T[4]/90*np.pi - np.pi, circular=True)+np.pi)/np.pi*90
            kde_results = np.array((lon_result,lat_result, ext_result, ell_result, ang_result))

            #Calculating the error
            result_uncerts0 = hdi(sampling.T[0], hdi_prob=0.68)
            result_uncerts1 = hdi(sampling.T[1], hdi_prob=0.68)
            result_uncerts2 = hdi(sampling.T[2], hdi_prob=0.68)
            result_uncerts3 = hdi(sampling.T[3], hdi_prob=0.68)
            result_uncerts4 = hdi(sampling.T[4]/90*np.pi - np.pi, hdi_prob=0.68, circular=True)*90/np.pi + 90
            result_uncerts = np.array((result_uncerts0, result_uncerts1, result_uncerts2, result_uncerts3, result_uncerts4))
            
            #Output format
            fitting_result = np.array((
                kde_results, 
                np.abs(result_uncerts.T[0]-kde_results), 
                np.abs(result_uncerts.T[1]-kde_results))
            ).T
            #Other useful quantities
            pm = np.array((
                np.nanmean(star_skycoords.pm_lon.value[star_mask]), 
                np.nanmean(star_skycoords.pm_lat.value[star_mask])
            ))
            pm_ra_err = hdi(star_skycoords.pm_lon.value[star_mask & ~np.isnan(star_skycoords.pm_lon)], hdi_prob=0.68) - pm[0]
            pm_dec_err = hdi(star_skycoords.pm_lat.value[star_mask & ~np.isnan(star_skycoords.pm_lat)], hdi_prob=0.68) - pm[1]
            rad_vel = np.nanmean(star_skycoords.radial_velocity[star_mask])
        else:
            print(f'Plummer profile fitting failed')
            fitting_result = np.zeros((5,3))
            pm = np.array((np.nan, np.nan))
            pm_ra_err = np.array((np.nan, np.nan))
            pm_dec_err = np.array((np.nan, np.nan))
            rad_vel = np.nan
    else:
        print(f'No stars within {num_radii} x {radius}')
        fitting_result = np.zeros((5,3))
        pm = np.array((np.nan, np.nan))
        pm_ra_err = np.array((np.nan, np.nan))
        pm_dec_err = np.array((np.nan, np.nan))
        rad_vel = np.nan

    results = {}

    results['ra'] = halo_skycoords.spherical.lon
    results['dec'] = halo_skycoords.spherical.lat
    results['pm'] = pm * u.mas/u.yr
    results['pm_ra_err'] = pm_ra_err * u.mas/u.yr
    results['pm_dec_err'] = pm_dec_err * u.mas/u.yr
    results['rad_vel'] = rad_vel
    results['lon'] = fitting_result[0,0] * u.degree
    results['lon_err'] = fitting_result[0,1:] * u.degree
    results['lat'] = fitting_result[1,1:] * u.degree
    results['lat_err'] = fitting_result[1,1:] * u.degree
    results['extension'] = fitting_result[2,0] * u.degree
    results['extension_err'] = fitting_result[2,1:] * u.degree
    results['ellipticity'] = fitting_result[3,0]
    results['ellipticity_err'] = fitting_result[3,1:]
    results['position_angle'] = fitting_result[4,0] * u.degree
    results['position_angle_err'] = fitting_result[4,1:] * u.degree

    if len(star_metals)>0:
        results['fitted_metals'] = np.nanmedian(star_metals.T[star_mask].T, axis=0)
    
    if return_derived_quants:
        center_dist = np.linalg.norm(center_pos)
        ext_in_rads = fitting_result[2,0] * np.pi/180

        results['r_half'] = center_dist * np.tan(ext_in_rads) * fitting_result[3,0] * 1000 * u.pc
        r_half_fact = center_dist * (np.pi/180) / np.cos(ext_in_rads)**2
        results['r_half_err'] = (
            np.linalg.norm(
                (
                    r_half_fact*fitting_result[2,1],
                    fitting_result[3,1]/(2*np.sqrt(1-fitting_result[3,0]))
                )
            ),
            np.linalg.norm(
                (
                    r_half_fact*fitting_result[2,2],
                    fitting_result[3,2]/(2*np.sqrt(1-fitting_result[3,0])) 
                )
            )
        )

        angle_dif = np.abs(fitting_result[4,0] - np.arctan2(pm[1], pm[0])*180/np.pi)
        results['angle_dif'] = np.abs(angle_dif-180*(angle_dif>90)-180*(angle_dif>270)) * u.degree
        results['ang_dif_err'] = (
            np.linalg.norm((
                fitting_result[4,1], 
                ((pm[0]/pm[1])/(1+(pm[0]/pm[1]))**2)*np.linalg.norm(
                    ((
                        pm_ra_err[0]/pm[0], 
                        pm_dec_err[0]/pm[1]
                    )), 
                    axis=0
                )*180/np.pi)
            ),
            np.linalg.norm((
                fitting_result[4,2], 
                ((pm[0]/pm[1])/(1+(pm[0]/pm[1]))**2)*np.linalg.norm(
                    ((
                        pm_ra_err[1]/pm[0], 
                        pm_dec_err[1]/pm[1]
                    )), 
                    axis=0
                )*180/np.pi)
            )
        ) * u.degree

    return results

def r_half_density(r_half, center_pos, star_pos, dm_pos, star_masses, dm_masses, return_totals=True):
    '''
    Calculates the cumulative distribution of stars and dark matter particles within 50 r_half.
    Parameters
    ----------
    r_half : float.
        The physical azimuthally averaged r_half.
    center_pos : np.array-like with shape (3,).
        The 3D position of the center of the progenitor. (same units as r_half)
    star_pos : np.array-like with shape (N, 3) with astropy units.
        The 3D positions of the star particles [N] with the origin at the GC. (same units as r_half)
    dm_pos : np.array-like with shape (N, 3) with astropy units.
        The 3D positions of the dark matter particles [M] with the origin at the GC. (same units as r_half)
    star_masses : np.array-like with shape (N,).
        The mass of the star particles (M_sun)
    dm_masses : np.array-like with shape (M,).
        The mass of the dark matter particles (M_sun)
    return_totals : bool, optional.
        Returns the total count of stars, dark matter, and the total mass
        Default : True

    Returns
    -------
    results : tuple (4 or 5 (if return_totals),)
        [0] : np.array-like with shape (50,)
            The cumulative count of star particles within 50 r_half with r_half intervals.
        [1] : np.array-like with shape (50,)
            The cumulative mass of star particles within 50 r_half with r_half intervals.
        [2] : np.array-like with shape (50,)
            The cumulative count of dark matter particles within 50 r_half with r_half intervals.
        [3] : np.array-like with shape (50,)
            The cumulative mass of dark matter particles within 50 r_half with r_half intervals.
        [4] : np.array-like with shape (4,), optional with return_totals
            [0] The total number of star particles
            [1] The total mass of all star particles
            [2] The total number of dark matter particles
            [3] The total mass of all dark matter particles
    '''
    r_half_bins = np.linspace(0, 50*r_half, num=51)
    star_dists = np.linalg.norm(star_pos-center_pos, axis=1)
    dm_dists = np.linalg.norm(dm_pos-center_pos, axis=1)

    if r_half<=0:
        star_counts = np.zeros(50)*np.nan
        star_mass_dist = np.zeros(50)*np.nan
        dm_counts = np.zeros(50)*np.nan
        dm_mass_dist = np.zeros(50)*np.nan
    else:
        star_counts = np.cumsum(np.histogram(
            star_dists, 
            r_half_bins
        )[0])
        star_mass_dist = np.cumsum(binned_statistic(
            star_dists, 
            star_masses, 'sum', 
            r_half_bins
        )[0])
        dm_counts = np.cumsum(np.histogram(
            dm_dists, 
            r_half_bins
        )[0])
        dm_mass_dist = np.cumsum(binned_statistic(
            dm_dists, 
            dm_masses, 'sum', 
            r_half_bins
        )[0])

    totals = np.zeros(4)
    totals[0] = len(star_dists)
    totals[1] = np.sum(star_masses)
    totals[2] = len(dm_dists)
    totals[3] = np.sum(dm_masses)

    if return_totals:
        return (star_counts, star_mass_dist, dm_counts, dm_mass_dist, totals)  
    else:
        return(star_counts, star_mass_dist, dm_counts, dm_mass_dist)
