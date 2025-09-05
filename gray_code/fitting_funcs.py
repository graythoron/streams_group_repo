import numpy as np
import os
from ugali.analysis import kernel
import emcee
from contextlib import closing
from multiprocessing import Pool

def randomize_walkers(ra, dec, nwalkers=200):
    lon = np.random.normal(loc=np.mean(ra), scale=2.0, size=nwalkers)
    lat = np.random.normal(loc=np.mean(dec), scale=2.0, size=nwalkers)
    extension = np.random.uniform(0.0001, 10, size=nwalkers)
    ellipticity = np.random.uniform(0, 1, size=nwalkers)
    position_angle = np.random.uniform(0, 180, size=nwalkers)
    return np.vstack([lon, lat, extension, ellipticity, position_angle]).T

def log_prior(params):
    lon, lat, r_h, ellipticity, position_angle = params
    if lon < -180 or lon > 180:
        return -np.inf
    elif lat < -90 or lat > 90:
        return -np.inf
    elif r_h < 0.0001 or r_h > 10:
        return - np.inf
    elif ellipticity < 0 or ellipticity > 1:
        return -np.inf
    elif position_angle < 0 or position_angle > 180:
        return -np.inf
    else:
        return 0

def log_likelihood(params, ra, dec, plummer):
    lon, lat, r_h, ellipticity, position_angle = params 
    truncate = np.inf

    if not np.isfinite(log_prior(params)):
        return -np.inf

    try:
        plummer.setp('lon', lon)
        plummer.setp('lat', lat)
        plummer.setp('extension', r_h)
        plummer.setp('position_angle', position_angle)
        plummer.setp('ellipticity', ellipticity)
        plummer.setp('truncate', truncate)
    except ValueError:
        # uniform prior
        return -np.inf

    like = plummer.pdf(ra, dec)
    return np.sum(np.log(like))

def fit_plummer(ra, dec, ndim=5, nwalkers=200, nsteps=600, nthreads=10, outfile='', restart=True):
    plummer = kernel.factory(name='EllipticalPlummer')
    plummer.setp('extension', 0.1, bounds=[0.0001, 20.])
    plummer.setp('truncate', np.inf, bounds=[0., np.inf])
    backend = emcee.backends.HDFBackend(outfile)

    if restart and os.path.exists(outfile):
        ics = backend.get_last_sample()
    else:
        ics = randomize_walkers(ra, dec, nwalkers=nwalkers)
        backend.reset(nwalkers, ndim)

    with closing(Pool(processes=nthreads)) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers, 
            ndim=ndim, 
            log_prob_fn=log_likelihood, 
            args=(ra, dec, plummer), 
            pool=pool, 
            backend=backend
        )

        pos, prob, state = sampler.run_mcmc(ics, 200)
        sampler.reset()
        print('Finished burn-in')

        sampler.run_mcmc(pos, nsteps)

    samples = sampler.chain[:, :, :].reshape((-1, ndim))

    return sampler, samples
