import sys, os, psutil, h5py, time

import numpy as np
import astropy.units as u

from astropy.table import Table, QTable, hstack
from arviz import hdi
from pympler import asizeof
from concurrent.futures import ProcessPoolExecutor

sys.path.append('/home/graythoron/projects/code')
from bound_fraction import compute_boundness_recursive_BFE, _find_center_position
from morphology_diagnostic import compute_morphological_diagnostics
from plotting_funcs import superplotMoI, chainplot, cornerplot, superplot
from doing_funcs import radius_mask, density_median, output_file as output_file_maker
from fitting_funcs import fit_plummer

import matplotlib.pyplot as plt
points = {'s': 2, 'rasterized': True}
combos = [(0,1), (0,2), (1,2)]
label = ['x', 'y', 'z']

def log_memory():
    proc = psutil.Process(os.getpid())
    virtual_mem = psutil.virtual_memory()
    print(f"PID {os.getpid()} using {proc.memory_info().rss / 1e9:.2f} GB with {virtual_mem.available /1e9:.2f} available.")

def load_dm_ids(stream_id, halo, level, dm_snapshot_ids):
    f1 = h5py.File(f'/eagle/Auriga/Auriga/debris_group_dm_ids/L{level}_halo{halo}_dm_ids_by_debris_group.hdf5', 'r') 
    dm_data = f1[f'{stream_id}']
    dm_ids = np.asarray(dm_data['dm_ids'])
    f1.close()
    _, _, s_ids_index = np.intersect1d(dm_ids, dm_snapshot_ids, return_indices=True)
    return s_ids_index

def plot_stream(self, plot_dm = False, plot_rep='scatter', save_fig=False, title='', center=False):
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
    if center != False:
        (ax1, ax2, ax3) = gs.subplots()
    else:
        (ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')
    axes=(ax1, ax2, ax3)
    __axes__=(ax1, ax2, ax3)

    # linear fit from (2, 0.1) to (7.1, 0.004)
    if self.load_dm:
        dm_xyz = np.array([self.strm_dm_xyz.x, self.strm_dm_xyz.y, self.strm_dm_xyz.z]).T
        alpha_dm = 0.137647 - 0.0188235*np.log10(len(self.strm_dm_masses))

    alpha_star = 0.137647 - 0.0188235*np.log10(len(self.strm_masses))
    strm_xyz = np.array([self.strm_xyz.x, self.strm_xyz.y, self.strm_xyz.z]).T

    if plot_dm & self.load_dm:
        limit = np.max((
            np.sqrt(np.max(dm_xyz**2)),
            np.sqrt(np.max(strm_xyz**2))
        ))
    else:
        limit = np.max(np.sqrt(np.max(strm_xyz**2)))
    bins = np.arange(-limit, limit, 1)
    n_bins = 50
    L_lim = 1e7
    #limit = 100

    def plot_based_on_rep(ax, pltrep, xyz, c='tab:blue', label='', alpha=alpha_star, cmap='viridis'):
        if pltrep == 'scatter':
            ax.scatter(xyz[:,j], xyz[:,k], alpha=alpha, c=c, label=label, **points)
        elif pltrep == 'hist':
            ax.hist2d(xyz[:,j], xyz[:,k], label=label, cmap=cmap, bins=bins, norm='log')

    for ii in range(3):
        j, k = combos[ii]
        if (len(plot_rep) == 2) and plot_dm and self.load_dm:
            plot_based_on_rep(axes[ii], plot_rep[0], dm_xyz, 'k', 'Dark Matter', alpha_dm, 'plasma')
            plot_based_on_rep(axes[ii], plot_rep[1], strm_xyz, 'r', 'Star')
        elif (len(plot_rep) == 2):
            plot_rep = plot_rep[0]

        if plot_dm and self.load_dm:
            plot_based_on_rep(axes[ii], plot_rep, dm_xyz, 'k', 'Dark Matter', alpha_dm, 'plasma')    
        plot_based_on_rep(axes[ii], plot_rep, strm_xyz, 'r', 'Star')

        axes[ii].set_ylabel(label[k])
        axes[ii].set_xlabel(label[j])

    for axi in __axes__:
        axi.set_xlim(-limit, limit)
        axi.set_ylim(-limit, limit)
        axi.set_aspect('equal')
    if center != False:
        strm_center = _find_center_position(strm_xyz, self.strm_masses, 'kde')
        print(strm_center)
        if self.load_dm:
            dm_center = _find_center_position(dm_xyz, self.strm_dm_masses, 'kde')
            print(dm_center)
        for ii in range(3):
            j,k = combos[ii]
            axes[ii].scatter(strm_center[j], strm_center[k], c='tab:blue', marker='x', label='Star Centroid')
            axes[ii].scatter(self.halo_pos[j], self.halo_pos[k], c='tab:red', marker='x', label='Halo Centroid')
            if self.load_dm:
                axes[ii].scatter(dm_center[j], dm_center[k], c='tab:green', marker='x', label='Dark Matter Centroid')
            axes[ii].set_xlim(strm_center[j]-center, strm_center[j]+center)
            axes[ii].set_ylim(strm_center[k]-center, strm_center[k]+center)

    plt.title(title)
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(f'output_{title}.png')

def bound_fraction_task(self, strmID): 
    results_table = Table()
    results_table['n_part_dm'] = [len(self.strm_dm_masses)]
    results_table['n_part'] = [len(self.strm_masses)]

    results = []
    methods = [['star', True], ['star', False], ['both', True], ['both', False]]

    for i in range(len(methods)):
        results0 = compute_boundness_recursive_BFE(
            self.strm_dm_positions, self.strm_dm_velocities, self.strm_dm_masses,
            self.strm_positions,    self.strm_velocities,    self.strm_masses, 
            center_on=methods[i][0], center_vel_with_KDE=methods[i][1]
        )
    
        results_table[f'f_bound_part_dm_{i}'] = [np.mean(results0[0][0]==1)]
        results_table[f'f_bound_mass_dm_{i}'] = [np.sum(self.strm_dm_masses[results0[0][0]==1])/np.sum(self.strm_dm_masses)]
        results_table[f'f_bound_part_{i}'] = [np.mean(results0[0][1]==1)]
        results_table[f'f_bound_mass_{i}'] = [np.sum(self.strm_masses[results0[0][1]==1])/np.sum(self.strm_masses)]
        results_table[f'center_x_{i}'] = [results0[1][0]]
        results_table[f'center_y_{i}'] = [results0[1][1]]
        results_table[f'center_z_{i}'] = [results0[1][2]]
        results_table[f'center_v_x_{i}'] = [results0[2][0]]
        results_table[f'center_v_y_{i}'] = [results0[2][1]]
        results_table[f'center_v_z_{i}'] = [results0[2][2]]
        results.append(results0)

    m_i = np.argmax([np.mean(i[0][1]==1) for i in results])
    with h5py.File(f"/eagle/Auriga/gthoron/stream_cats/galframe/{self.levelID}_{self.haloID}_{strmID}.hdf5", "a") as ledger:
        if np.all(np.array(list(ledger[f'{self.snap}'].keys())) != f'dm_bound'):
            ledger[f'{self.snap}/dm_bound'] = results[m_i][0][0]
            ledger[f'{self.snap}/star_bound'] = results[m_i][0][1] 
        else:
            del ledger[f'{self.snap}/dm_bound']
            del ledger[f'{self.snap}/star_bound']
            ledger[f'{self.snap}/dm_bound'] = results[m_i][0][0]
            ledger[f'{self.snap}/star_bound'] = results[m_i][0][1]
        
        if np.all(np.array(list(ledger[f'{self.snap}'].keys())) != f'center_vel'):
                ledger[f'{self.snap}/center_vel'] = results[m_i][1]
                ledger[f'{self.snap}/center_pos'] = results[m_i][2] 
        else:
                del ledger[f'{self.snap}/center_vel']
                del ledger[f'{self.snap}/center_pos']
                ledger[f'{self.snap}/center_vel'] = results[m_i][1]
                ledger[f'{self.snap}/center_pos'] = results[m_i][2]
        
        for i in range(len(results)):
            if np.all(np.array(list(ledger[f'{self.snap}'].keys())) != f'dm_bound_{i}'):
                ledger[f'{self.snap}/dm_bound_{i}'] = results[i][0][0]
                ledger[f'{self.snap}/star_bound_{i}'] = results[i][0][1] 
            else:
                del ledger[f'{self.snap}/dm_bound_{i}']
                del ledger[f'{self.snap}/star_bound_{i}']
                ledger[f'{self.snap}/dm_bound_{i}'] = results[i][0][0]
                ledger[f'{self.snap}/star_bound_{i}'] = results[i][0][1]

            if np.all(np.array(list(ledger[f'{self.snap}'].keys())) != f'center_vel_{i}'):
                ledger[f'{self.snap}/center_vel_{i}'] = results[i][1]
                ledger[f'{self.snap}/center_pos_{i}'] = results[i][2] 
            else:
                del ledger[f'{self.snap}/center_vel_{i}']
                del ledger[f'{self.snap}/center_pos_{i}']
                ledger[f'{self.snap}/center_vel_{i}'] = results[i][1]
                ledger[f'{self.snap}/center_pos_{i}'] = results[i][2]
    
    results_table.add_column(results[m_i][2][2], name='center_v_z', index=0)
    results_table.add_column(results[m_i][2][1], name='center_v_y', index=0)
    results_table.add_column(results[m_i][2][0], name='center_v_x', index=0)
    results_table.add_column(results[m_i][1][2], name='center_z',   index=0)
    results_table.add_column(results[m_i][1][1], name='center_y',   index=0)
    results_table.add_column(results[m_i][1][0], name='center_x',   index=0)
    results_table.add_column(np.sum(self.strm_masses[results[m_i][0][1]==1])/np.sum(self.strm_masses), name='f_bound_mass', index=0)
    results_table.add_column(np.mean(results[m_i][0][1]==1), name='f_bound_part', index=0)
    results_table.add_column(np.sum(self.strm_dm_masses[results[m_i][0][0]==1])/np.sum(self.strm_dm_masses), name='f_bound_mass_dm', index=0)
    results_table.add_column(np.mean(results[m_i][0][0]==1), name='f_bound_part_dm', index=0)

    return results_table

def remember_stream_ids(self, strmID):
    print('Loading the snapshot to grab IDs because I forgot them.')
    snapdata =  self.load_snapshot_data(self.snap, loadonly = ['id'])
    # these ids will allow us to match later
    self.snapdata_ids = snapdata['id']
    del snapdata
        
    #Identifying indices in the bigger particle lists
    strm_idx = self.reader.load_stream_indices(strmID, self.plists['pkmassid'], self.plists['idacc'], self.snapdata_ids) 

    dm_snapdata = self.load_snapshot_data(self.snap, loadonly = ['id'], loadonlytype=[1])
    
    # these ids will allow us to match later
    self.dm_snapdata_ids = dm_snapdata['id']
    del dm_snapdata
        
    dm_strm_idx = load_dm_ids(strmID, self.haloID, self.levelID, self.dm_snapdata_ids)
    print('Closing snapshots')

    with h5py.File(f"/eagle/Auriga/gthoron/stream_cats/galframe/{self.levelID}_{self.haloID}_{strmID}.hdf5", "a") as ledger:
        if np.all(np.array(list(ledger[f'{self.snap}'].keys())) != f'star_id'):
            ledger[f'{self.snap}/star_id'] = self.snapdata_ids[strm_idx]
            ledger[f'{self.snap}/dm_id'] = self.dm_snapdata_ids[dm_strm_idx]
        else:
            del ledger[f'{self.snap}/star_id']
            del ledger[f'{self.snap}/dm_id']
            ledger[f'{self.snap}/star_id'] = self.snapdata_ids[strm_idx]
            ledger[f'{self.snap}/dm_id'] = self.dm_snapdata_ids[dm_strm_idx]
    print('Done!')
    return Table()

def MoI_task(self, strmID):
    if self.multisnap:
        outdir = f'halo{self.haloID}_stream{strmID}_L{self.levelID}_ep{self.ep}/'
    else:
        outdir = ''  

    #Projecting Mask Radius 
    # given by half mass radius * (ep) extention parameter
    hmsr_proj_expand = np.arctan(self.ep*self.shInfo[3]/self.shInfo[2]).to(u.degree).value
    print(f'    Half Stellar Mass Radius: {self.shInfo[3]}')
    print(f'    Distance to halo: {self.shInfo[2]}')
    print(f'    Halo radius*{self.ep} projection: {hmsr_proj_expand}')

    if self.ep == 0:
        hm_mask = self.bound_stars
    else:
        hm_mask = radius_mask(self.strm_skyra,self.strm_skydec,hmsr_proj_expand)
            
    print(f'    Bound particles: {np.sum(self.bound_stars)}')
    print(f'    Fitted particles: {np.sum(hm_mask)}')
    print(f'    Bound and fitted particles: {np.sum(hm_mask & self.bound_stars)}')

    outfile, filename = output_file_maker(self.haloID, self.levelID, strmID, self.save_dir, self.coordsys, self.ep, self.snap, self.continuation)

    xyz = np.array([self.strm_xyz.x.value,self.strm_xyz.y.value,self.strm_xyz.z.value]).T
    vels = np.array([self.strm_xyz.v_x.value,self.strm_xyz.v_y.value,self.strm_xyz.v_z.value]).T
    
    if len(self.strm_masses.value[hm_mask]) == 0:
        abc = [np.nan, np.nan, np.nan]
        dir_elong0 = [np.nan, np.nan, np.nan]
        dir_elong1 = [np.nan, np.nan, np.nan]
        dir_elong2 = [np.nan, np.nan, np.nan]
        vel = [np.nan, np.nan, np.nan]
        gal_center = [np.nan, np.nan, np.nan]
        dp_pm = np.nan
        dp_gc = np.nan
        ellip = np.nan
        triax = np.nan
    else: 
        center_of_mass = np.average(xyz[hm_mask], weights = self.strm_masses.value[hm_mask], axis=0)

        masked_xyz = xyz[hm_mask]-center_of_mass

        abc, eig_vecs, ellip, triax = compute_morphological_diagnostics(masked_xyz, self.strm_masses.value[hm_mask], vels[hm_mask])

        dir_elong0 = eig_vecs[0]
        dir_elong1 = eig_vecs[1]
        dir_elong2 = eig_vecs[2]

        vel = np.nanmean(vels[hm_mask], axis=0)

        gal_center = -center_of_mass
        dp_pm = np.dot(vel, dir_elong0)/(np.linalg.norm(dir_elong0)*np.linalg.norm(vel))
        dp_gc = np.dot(gal_center, dir_elong0)/(np.linalg.norm(dir_elong0)*np.linalg.norm(gal_center))

    results_table = Table()
    results_table['major_axis'] = [abc[0]]
    results_table['inter_axis'] = [abc[1]]
    results_table['minor_axis'] = [abc[2]]
    results_table['major_axis_x'] = [dir_elong0[0]]
    results_table['major_axis_y'] = [dir_elong0[1]]
    results_table['major_axis_z'] = [dir_elong0[2]]
    results_table['inter_axis_x'] = [dir_elong1[0]]
    results_table['inter_axis_y'] = [dir_elong1[1]]
    results_table['inter_axis_z'] = [dir_elong1[2]]
    results_table['minor_axis_x'] = [dir_elong2[0]]
    results_table['minor_axis_y'] = [dir_elong2[1]]
    results_table['minor_axis_z'] = [dir_elong2[2]]
    results_table['ellipticity'] = [ellip]
    results_table['triaxial'] = [triax]
    results_table['vel_x'] = [vel[0]]
    results_table['vel_y'] = [vel[1]]
    results_table['vel_z'] = [vel[2]]
    results_table['galcent_x'] = [gal_center[0]]
    results_table['galcent_y'] = [gal_center[1]]
    results_table['galcent_z'] = [gal_center[2]]
    results_table['dotproduct_pmotion'] = [dp_pm]
    results_table['dotproduct_galcenter'] = [dp_gc]
    results_table['fitted_count'] = [np.sum(hm_mask)]
    results_table['fitted_mass'] = [np.sum(self.strm_masses.value[hm_mask])]

    print('*~~~~~~~~~~~~~~~~*')
    print(f'Fitting Results')
    print(f'    {results_table[0]}')

    results_table['total_mass'] = np.sum(self.strm_masses).value
    results_table['median_feh'] = np.nanmedian(self.strm_metals[0])
    results_table['fitted_feh'] = np.nanmedian(self.strm_metals[0][hm_mask])
    results_table['median_mgfe'] = np.nanmedian(self.strm_metals[1])
    results_table['fitted_mgfe'] = np.nanmedian(self.strm_metals[1][hm_mask])

    moi_matrix = np.array((
            (results_table['major_axis_x'][0], results_table['major_axis_y'][0], results_table['major_axis_z'][0]),
            (results_table['inter_axis_x'][0], results_table['inter_axis_y'][0], results_table['inter_axis_z'][0]),
            (results_table['minor_axis_x'][0], results_table['minor_axis_y'][0], results_table['minor_axis_z'][0]),
            ))
    moi_axes = np.array((results_table['major_axis'][0],results_table['inter_axis'][0],results_table['minor_axis'][0]))
    
    if hmsr_proj_expand == 0:
        hmsr_proj_expand = np.max(xyz[self.bound_stars] - self.halo_pos.value)
    
    pm = superplotMoI(
            hmsr_proj_expand, 
            moi_axes, moi_matrix,
            self.bound_stars, 
            xyz*u.kpc,  
            self.halo_skycoords,
            self.strm_skycoords, 
            self.stream_cat['apo'][self.stream_cat['stream_id']==strmID], 
            'ICRS', strmID, f'{outdir}{filename}', self.halo_pos
        )
    print('*~~~~~~~~~~~~~~~~*')

    results_table.add_column(self.shInfo[3], name='shHMR', index = 0)
    results_table.add_column(self.shInfo[2], name='shDist', index = 0)
    results_table.add_column(self.shInfo[1], name='shDec', index = 0)
    results_table.add_column(self.shInfo[0], name='shRA', index = 0)
    return results_table

def plummer_task(self, strmID):
    if self.multisnap:
        outdir = f'halo{self.haloID}_stream{strmID}_L{self.levelID}_ep{self.ep}/'
    else:
        outdir = ''
    
    outfile, filename = output_file_maker(self.haloID, self.levelID, strmID, self.save_dir, self.coordsys, self.ep, self.snap, self.continuation)

    #Projecting Mask Radius 
    # given by half mass radius * (ep) extention parameter
    hmsr_proj_expand = np.arctan(self.ep*self.shInfo[3]/self.shInfo[2]).to(u.degree).value
    print(f'    Half Stellar Mass Radius: {self.shInfo[3]}')
    print(f'    Distance to halo: {self.shInfo[2]}')
    print(f'    Halo radius*{self.ep} projection: {hmsr_proj_expand}')

    if self.ep == 0:
        hm_mask = np.bool(self.bound_stars)
        output_file = f'{outdir}bound{filename}'
    else:
        hm_mask = radius_mask(self.strm_skyra,self.strm_skydec,hmsr_proj_expand)
        output_file = f'{outdir}{filename}'
        
    print(f'    Bound particles: {np.sum(self.bound_stars)}')
    print(f'    Fitted particles: {np.sum(hm_mask)}')
    print(f'    Bound and fitted particles: {np.sum(hm_mask & self.bound_stars)}')

    if np.any(hm_mask): 
            #Identifying the output file. 
        print('Running MCMC...')

        sampler, __ = fit_plummer(
                self.strm_skyra[hm_mask], self.strm_skydec[hm_mask], 
                nsteps=800, nthreads=5, nwalkers=500,
                outfile=outfile, restart=self.continuation
        )

        #Formatting the chain so we can extract information about each parameter
        sample = sampler.get_chain()
        live_samples = sample[0, :, 0] != sample[-1, :, 0]

        if np.any(live_samples):
            samples = sample[:, live_samples, :]
            sampling = samples.reshape(samples.shape[0]*samples.shape[1], 5)

            lon_result = density_median(sampling.T[0])
            lat_result = density_median(sampling.T[1])
            ext_result = density_median(sampling.T[2])
            ell_result = density_median(sampling.T[3])
            ang_result = (density_median(sampling.T[4]/90*np.pi - np.pi, circular=True)+np.pi)/np.pi*90
            kde_results = np.array((lon_result,lat_result, ext_result, ell_result, ang_result))

            result_uncerts0 = hdi(sampling.T[0], hdi_prob=0.68)
            result_uncerts1 = hdi(sampling.T[1], hdi_prob=0.68)
            result_uncerts2 = hdi(sampling.T[2], hdi_prob=0.68)
            result_uncerts3 = hdi(sampling.T[3], hdi_prob=0.68)
            result_uncerts4 = hdi(sampling.T[4]/90*np.pi - np.pi, hdi_prob=0.68, circular=True)*90/np.pi+ 90
            result_uncerts = np.array((result_uncerts0, result_uncerts1, result_uncerts2, result_uncerts3, result_uncerts4))
            fitting_result = np.array((kde_results, result_uncerts.T[0], result_uncerts.T[1]))

            #Plotting the chains
            chainplot(samples, output_file)
            cornerplot(samples, output_file)
        else:
            fitting_result = np.zeros((3,5))

    else:
        print(f'    No stars within {self.ep} x subhalo radius')
        fitting_result = np.zeros((3,5))
    print('*~~~~~~~~~~~~~~~~*')
    print(f'Fitting Results')
    print(f'    {fitting_result[0]}')

    if hmsr_proj_expand == 0:
        xyz_holder = np.array([self.strm_xyz.x.value,self.strm_xyz.y.value,self.strm_xyz.z.value]).T[np.bool(self.bound_stars)]
        hmsr_proj_expand = np.max(np.linalg.norm(xyz_holder - self.halo_pos.value))
    
    #Plotting the shape
    pm = superplot(
            hmsr_proj_expand, fitting_result[0], self.bound_stars, 
            np.array([self.strm_xyz.x.value,self.strm_xyz.y.value,self.strm_xyz.z.value]).T,  
            self.halo_skycoords, self.strm_skycoords, 
            self.stream_cat['apo'][self.stream_cat['stream_id']==strmID], 
            strmID, f'{outdir}{filename}', self.halo_pos
        )
    
    if np.any(hm_mask): 
        fitting_result = np.array(fitting_result).T
    else:
        fitting_result = np.nan*np.array(fitting_result).T

    shTable =QTable(
        [[self.shInfo[0]], [self.shInfo[1]], [self.shInfo[2]], [self.shInfo[3]]],
        names = ['shRA', 'shDec', 'shDist', 'shHMR']
    )

    pmTable = QTable(
        np.array(pm),
        units = [u.mas/u.yr, u.mas/u.yr, u.km/u.s],
        names = ['pm_ra', 'pm_dec', 'rad_vel']
    )

    strmTable  = QTable(
        [[np.sum(hm_mask)], [np.sum(self.strm_masses)], [np.sum(self.strm_masses[hm_mask])], [np.nanmedian(self.strm_metals[0])], 
         [np.nanmedian(self.strm_metals[0][hm_mask])], [np.nanmedian(self.strm_metals[1])], [np.nanmedian(self.strm_metals[1][hm_mask])]],
        names = ['fitted_particles', 'total_mass', 'fitted_mass', 'median_feh', 'fitted_feh', 'median_mgfe', 'fitted_mgfe']
    )
    unit_conv = [u.degree, u.degree, u.degree, None, u.degree]
    plummerTable = QTable(
        fitting_result[:, 0],
        units = unit_conv,
        names = ['lon', 'lat', 'extension', 'ellipticity', 'position_angle']
    )
    
    lowErrTable = QTable(
        fitting_result[:, 0] - fitting_result[:, 1],
        units = unit_conv,
        names = ['lon_low_err', 'lat_low_err', 'extension_low_err', 'ellipticity_low_err', 'position_angle_low_err']
    )
    
    upErrTable = QTable(
        fitting_result[:, 2] - fitting_result[:, 0],
        units = unit_conv,
        names = ['lonupw_err', 'lat_up_err', 'extension_up_err', 'ellipticity_up_err', 'position_angle_up_err']
    )

    return hstack((shTable, pmTable, strmTable, plummerTable, lowErrTable, upErrTable))

def stream_save_task(self, strmID):
    if os.path.exists(f"{self.cache_path}/{self.levelID}_{self.haloID}_{strmID}_{self.snap}.hdf5"):
        pass
    else:
        self.stream_loader(strmID)
        log_memory()
        print(f'    reader: {asizeof.asizeof(self.reader)/(10**9)} GB')
        print(f'    plists: {asizeof.asizeof(self.plists)/(10**9)} GB')
        print(f'    snapdata_ids: {asizeof.asizeof(self.snapdata_ids)/(10**9)} GB')
        print(f'    x-y-z positions: {asizeof.asizeof(self.cartpos)/(10**9)} GB')
        print(f'    stream x-y-z positions: {asizeof.asizeof(self.strm_xyz)/(10**9)} GB')
        print(f'    metals: {asizeof.asizeof(self.metalZ)/(10**9)} GB')
        if self.load_dm:
            print(f'    dm_snapdata_ids: {asizeof.asizeof(self.dm_snapdata_ids)/(10**9)} GB')
            print(f'    dm x-y-z positions: {asizeof.asizeof(self.dm_cartpos)/(10**9)} GB')
            print(f'    stream dm x-y-z positions: {asizeof.asizeof(self.strm_dm_xyz)/(10**9)} GB')
        with h5py.File(f"{self.cache_path}/{self.levelID}_{self.haloID}_{strmID}_{self.snap}.hdf5", "w") as ledger:
            ledger['star_id'] = np.array(self.snapdata_ids[self.strm_idx])
            ledger['star_pos'] = np.array([self.strm_xyz.x.value, self.strm_xyz.y.value, self.strm_xyz.z.value]).T
            ledger['star_vel'] = np.array([self.strm_xyz.v_x.value, self.strm_xyz.v_y.value, self.strm_xyz.v_z.value]).T
            ledger['star_mass'] = self.strm_masses
            ledger['age'] = self.strm_ages
            ledger['fe_h'] = self.strm_metals[0]
            ledger['mg_fe'] = self.strm_metals[1]
            ledger['bound'] = self.bound_stars
            ledger['dm_id'] = np.array(self.dm_snapdata_ids[self.dm_strm_idx])
            ledger['dm_pos'] = np.array([self.strm_dm_xyz.x.value, self.strm_dm_xyz.y.value, self.strm_dm_xyz.z.value]).T
            ledger['dm_vel'] = np.array([self.strm_dm_xyz.v_x.value, self.strm_dm_xyz.v_y.value, self.strm_dm_xyz.v_z.value]).T
            ledger['dm_mass'] = self.strm_dm_masses

def hdf5paths_exist(init_path, level, halo, snap, stream_ids):
    paths = []
    for stream_id in stream_ids:
        paths.append(f'{init_path}/{level}_{halo}_{stream_id}_{snap}.hdf5')
    
    paths = np.array(paths)
    paths_exist = np.zeros(shape=paths.shape, dtype=int) == 0

    for path_index in range(len(paths)):
        path = paths[path_index]
        paths_exist[path_index] = os.path.exists(path)
    return paths_exist

def halo_splitter(self):
    print("We need to run the halo splitting! Hold tight! This can take a while.")
    self.load_dm = True
    self.include_sky_info = False
    self.include_halo = False
    for snap in self.snaps:
        streams = self.streamids[self.super_times.T[snap]]
        snap_streams = hdf5paths_exist(self.cache_path,self.levelID, self.haloID, snap, streams)
        if np.all(snap_streams):
            pass
        else:
            print(snap_streams)
            print(streams[~snap_streams])
            self.snap_loader(snap)

            print(f'Approximate memory usage of snapshot {snap}:')
            #print(f'    halo_manager: {asizeof.asizeof(self)/(10**9)} GB')
            print(f'    reader: {asizeof.asizeof(self.reader)/(10**9)} GB')
            print(f'    plists: {asizeof.asizeof(self.plists)/(10**9)} GB')
            print(f'    snapdata_ids: {asizeof.asizeof(self.snapdata_ids)/(10**9)} GB')
            print(f'    x-y-z positions: {asizeof.asizeof(self.cartpos)/(10**9)} GB')
            print(f'    metals: {asizeof.asizeof(self.metalZ)/(10**9)} GB')
            if self.load_dm:
                print(f'    dm_snapdata_ids: {asizeof.asizeof(self.dm_snapdata_ids)/(10**9)} GB')
                print(f'    dm x-y-z positions: {asizeof.asizeof(self.dm_cartpos)/(10**9)} GB')
        
            log_memory()
            opt_stream_workers = optimize_workers(self, len(streams[~snap_streams]))
            print(f'Loading {opt_stream_workers} streams at a time, for a total of {len(streams[~snap_streams])}')
            with ProcessPoolExecutor(max_workers=opt_stream_workers) as executor:
                results = list(executor.map(self.stream_save_task, streams[~snap_streams]))

    for strmID in self.streamids:
        with h5py.File(f"/eagle/Auriga/gthoron/stream_cats/galframe/{self.levelID}_{self.haloID}_{strmID}.hdf5", "w") as ledger:
            snaps = self.reader.get_snap_array()[(self.super_times[self.streamids == strmID])[0]]
            ledger['snaps'] = snaps
            for snap in snaps:
                with h5py.File(f"{self.cache_path}/{self.levelID}_{self.haloID}_{strmID}_{snap}.hdf5", "r") as subledger:
                    for key in list(subledger.keys()):
                        ledger[f'{snap}/{key}'] = np.asarray(subledger[key])
                #os.remove(f"{self.cache_path}/{self.levelID}_{self.haloID}_{strmID}_{snap}.hdf5")

def optimize_workers(self, num):
    memory = psutil.Process(os.getpid()).memory_info().rss
    memory_avail = psutil.virtual_memory().available
    return int(np.max((1, np.min([num, self.stream_workers, memory_avail/memory]))))

def subfind_bound_fraction(self, strmID):
    snapdata = self.reader.load_snapshot_data(nsnap=self.snap, reading=True, saving=False, center_on_host=True, units=True)
    sf = self.reader.load_subfind(nsnap=self.snap)

    particle_ids = snapdata['id']

    strm_idx = self.reader.load_stream_indices(strmID, self.plists['pkmassid'], self.plists['idacc'], particle_ids)

    # starting from pkmassid, get last instance subhalo is descendant's first progenitor
    index, desc, desc_first_prog = [strmID] * 3
    time = self.tree.data['snum'][index]
    
    keys = ['desc', 'fpin']
    while (index == desc_first_prog) and (time!=self.snap+1):
        index = desc
        desc = self.tree.data[keys[(time > self.snap)*1]][index]
        desc_first_prog = self.tree.data[keys[(time <= self.snap)*1]][desc]
        time = self.tree.data['snum'][desc]

    final_tree_index = index

    fof_group = self.tree.data['sgnr'][final_tree_index]
    subhalo = self.tree.data['sbnr'][final_tree_index]
    first_halo_in_fof = self.tree.data['sbnr'][self.tree.data['fhfg'][final_tree_index]]
    #This line keeps breaking
    assert first_halo_in_fof == sf.data['ffsh'][fof_group]

    # get star particles in subhalo, taking advantage of data structure
    start = np.sum(sf.data['flty'][:fof_group,4])
    start += np.sum(sf.data['slty'][first_halo_in_fof:subhalo,4])
    stop = start + np.sum(sf.data['slty'][subhalo,4])
    ids_bound = particle_ids[start:stop]
        
    # guarantee only using particles from lists (no wind particles, radius cut)
    bound_in_plists = np.isin(ids_bound, particle_ids[strm_idx])

    # fraction of bound particles in merger
    fbound_part = np.sum(bound_in_plists) / len(strm_idx)
    fbound_mass = np.sum(snapdata['mass'][start:stop][bound_in_plists]) / np.sum(snapdata['mass'][strm_idx])
    fbound_mass = fbound_mass.value

    # machine precision kills sometimes
    fbound_part = 1 if fbound_part > 1 else fbound_part
    fbound_mass = 1 if fbound_mass > 1 else fbound_mass
    
    results_table = Table()
    results_table['n_part'] = [len(strm_idx)]
    results_table['f_bound_part'] = [fbound_part]
    results_table['f_bound_mass'] = [fbound_mass]
    return results_table

def halo_support_aux(halo_manager):
    halo_manager.plot_stream = plot_stream
    halo_manager.bound_fraction_task = bound_fraction_task
    halo_manager.MoI_task = MoI_task
    halo_manager.plummer_task = plummer_task
    halo_manager.halo_splitter = halo_splitter
    halo_manager.stream_save_task = stream_save_task
    halo_manager.remember_stream_ids = remember_stream_ids
    halo_manager.debug = False
