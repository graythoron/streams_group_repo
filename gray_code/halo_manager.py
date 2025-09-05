import sys, h5py, psutil, os
import numpy as np
import astropy.units as u
import astropy.coordinates as coords

from scipy.stats import gaussian_kde
from astropy.table import vstack, Table, QTable
from concurrent.futures import ProcessPoolExecutor

sys.path.append('/home/graythoron/auriga_streams/code')
from select_streams import load_combined_table
import read_data


def align_n_rotate(matrix, pos_data):
    data = {'pos':np.copy(pos_data['pos']), 'vel': np.copy(pos_data['vel'])}
    #align with disk 
    for prop in ['pos', 'vel']:
        spinmat = np.array(matrix.transpose(), dtype=data[prop].dtype)
        data[prop] = np.dot(data[prop], spinmat)
        # goes from (z, y, x) to (x, y, z)
        if len(data[prop].shape)<2:
            data[prop] = np.flip(data[prop])
        else:
            data[prop] = np.fliplr(data[prop])
    return data


def galcoords(xyz, vels=None, coordsys='spherical', initialized=False):
    if initialized == True:
        coordinates = xyz.copy()
    else:
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

def stream_2_subhalo(tree, pkmassid, nsnap=0):
    #Identifies halo id from stream id by following the merger tree and looking for the pkmassid 
    index, desc, desc_first_prog = [pkmassid] * 3
    time = tree.data['snum'][index]
    keys = ['desc', 'fpin']
    while (index == desc_first_prog) and (time!=nsnap):
        index = desc
        desc = tree.data[keys[(time > nsnap)*1]][index]
        desc_first_prog = tree.data[keys[(time < nsnap)*1]][desc]
        time = tree.data['snum'][desc]

    subhalo = tree.data['sbnr'][index]
    return subhalo

def load_dm_ids(stream_id, halo, level, dm_snapshot_ids):
    f1 = h5py.File(f'/eagle/Auriga/Auriga/debris_group_dm_ids/L{level}_halo{halo}_dm_ids_by_debris_group.hdf5', 'r') 
    dm_data = f1[f'{stream_id}']
    dm_ids = np.asarray(dm_data['dm_ids'])
    f1.close()
    _, _, s_ids_index = np.intersect1d(dm_ids, dm_snapshot_ids, return_indices=True)
    return s_ids_index

def log_memory():
    proc = psutil.Process(os.getpid())
    virtual_mem = psutil.virtual_memory()
    print(f"PID {os.getpid()} using {proc.memory_info().rss / 1e9:.2f} GB with {virtual_mem.available /1e9:.2f} available.")

class stream_info_holder:
    pass

class halo_manager:
    stream_workers = 9
    workers = 12
    snap_workers = 4
    continuation = False
    load_dm = True
    include_sky_info = True
    include_halo = True
    cache_path = '/eagle/Auriga/gthoron/.cache/halo_manager'
    user_name = ''

    def load_snapshot_data(self, snap, align_with_disk=False, loadonly = ['id', 'pos', 'vel', 'mass', 'age', 'gmet'], loadonlytype=[4]):
        if loadonly == ['id']:
            return self.reader.load_snapshot_data(nsnap = snap, reading=True, loadonly = ['id'], loadonlytype = loadonlytype)
        else:
            return self.reader.load_snapshot_data(
                nsnap = snap, reading=True,  saving=False,  center_on_host=True, units=True, 
                loadonly = loadonly, align_with_disk=align_with_disk, loadonlytype = loadonlytype
            )

    def __init__(self, levelID, haloID, ep, save_dir, coordsys='ICRS', multisnap = False, snap=None, min_load=False):

        self.reader = read_data.AurigaReader.from_yaml(
            haloID, levelID, 
            f"/home/graythoron/projects/auriga-config.yaml",  f'level{levelID}'
        )
    
        #This is me checking whether the snap is the final snap 
        # and if it is not, making sure we don't try to rotate to that disk
        # as the align_with_disk is meant for present day.
        
        self.reader.cache_path = f'/home/graythoron/output/level{levelID}/'

        # I get the star particle
        #data for center_n_align
        if min_load:
            print('Host Halo coordinates and rotation matrix have been excluded')
        else:
            self.host_coords = self.reader.get_host_coords(nsnap=self.reader.final_snap)
            snapdata = self.load_snapshot_data(self.reader.final_snap, True)
            self.matrix = snapdata['matrix'].transpose()
            #cleaning up
            del snapdata

        #I get the particle lists.
        self.plists = self.reader.load_particle_lists(reading=True, saving=False)
            # and the merger tree
        self.tree = self.reader.load_tree(nsnap=self.reader.final_snap)
            # and the halo catalog
        self.halos = self.reader.load_subfind().data

        #Grabbing the big stream catalog from Nora and Alex
        metricsTable = load_combined_table()
        #Making some sensible cuts
        metricMask = (
                (metricsTable['level']==levelID) &
                (metricsTable['halo']==haloID)
            )
        self.stream_cat = Table.from_pandas(metricsTable[metricMask])
        #Grabbing the stream ids for later use
        self.streamids = self.stream_cat['stream_id']
        del metricsTable

        #Getting the snaps we'll use to run things.
        all_snaps = self.reader.get_snap_array()
        try:
           self.lookback_times = self.reader.get_snaptime(all_snaps>-10).value
        except:
            try:
                scales = np.loadtxt(f'{self.reader.cache_path}snaptimes.txt', unpack=True)
            except:
                print('OH NO Things are going wrong!')
                print(f'All Snaps {all_snaps}')
                print(f'File Path {self.reader.cache_path}')
                print("Scales aren't loading!")
                scales = []
            if len(scales) == 0:
                with h5py.File(f'{self.reader.cache_path}snaptimes.h5', 'r') as ledger:
                    scales = np.asarray(ledger['scale'])
            print(f'Trying to unpack lookback times...')
            try:
                self.lookback_times = read_data.cosmo.lookback_time(1 / scales - 1).value
                print(f'Success!')
            except:
                print(f'Failed!')
                raise KeyboardInterrupt
        self.super_times = np.array((self.lookback_times,)) <= np.array((self.stream_cat['t_acc'],)).T
        snaps_with_stuff = all_snaps[np.sum(self.super_times, axis=0) != 0]
        #self.super_times = self.super_times[:, np.sum(self.super_times, axis=0) != 0]

        self.haloID = haloID
        self.levelID = levelID
        self.ep = ep
        self.multisnap = multisnap
        self.save_dir = save_dir
        self.coordsys = coordsys

        #Preparing the multisnap proceedure
        if self.multisnap:
            self.coordsys = 'spherical'
            snaps = np.array(snap)
            if ((len(snaps.shape) != 1) or (len(snaps) <= 1)) and (snap != None):
                print('When multisnap is True, snap must be an 1d array of snapshot ids or None for all snapshots.')
                raise KeyboardInterrupt
            else:
                if snap == None:
                    self.snaps = snaps_with_stuff
                else:
                    self.snaps = snaps
        else:
            if snap == None:
                self.snaps = np.array([self.reader.final_snap])
            elif type(snap) == int:
                self.snaps = np.array([snap])
                self.coordsys = 'spherical'
            else:
                print('When multisnap is False, snap must be None or an integer.')

        print('')
        if self.snaps[0] == -1:
            print('Loading initial halos...')
        elif min_load != True:
            print(f' There are {len(self.snaps)} snapshots with {np.sum(self.super_times.T[self.snaps])} halos present.')
            print(' They are:')
            for snap in self.snaps:
                print(f'    {snap} ({len(self.streamids[self.super_times.T[snap]])}) :  {np.array(self.streamids[self.super_times.T[snap]])}')
   
    def task_manager(self, final_file):
        if callable(getattr(self, 'user_stream_task', None)):
            log_memory()

            strm_snap_ids = []
            strm_snap_ran = []

            if (len(self.snaps) == 1) & (self.snaps[0] == -1):
                indexes = np.array(np.where(self.super_times))
                initial_halos = []
                for index in np.unique(indexes[0]):
                    initial_halos.append((index, indexes[1][np.where(indexes[0] == index)[0][0]]))
                for halo in initial_halos:
                    if os.path.exists(f"{self.cache_path}/{self.user_name}_{self.levelID}_{self.haloID}_{halo[0]}_{halo[1]}.csv"):
                        strm_snap_ran.append((halo[0], halo[1]))
                    else:
                        strm_snap_ids.append((halo[0], halo[1]))
            
            else:
                for snap in self.snaps:
                    for stream in self.streamids[self.super_times.T[snap]]:
                        if os.path.exists(f"{self.cache_path}/{self.user_name}_{self.levelID}_{self.haloID}_{stream}_{snap}.csv"):
                            strm_snap_ran.append((stream, snap))
                        else:
                            strm_snap_ids.append((stream, snap))
            
            print(np.array(strm_snap_ids))
            print(np.array(strm_snap_ids).shape)
            print(np.array(strm_snap_ran))
            print(np.array(strm_snap_ran).shape)
            
            if len(strm_snap_ids) > 0:
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    results = list(executor.map(self.stream_snap_task, strm_snap_ids))
                result_table = vstack(results)
            else:
                result_table = QTable()
            
            for ID in strm_snap_ran:
                ledger = QTable.read(f"{self.cache_path}/{self.user_name}_{self.levelID}_{self.haloID}_{ID[0]}_{ID[1]}.csv")
                result_table = vstack((result_table, ledger))
                #os.remove(f"{self.cache_path}/{self.user_name}_{self.levelID}_{self.haloID}_{ID[0]}_{ID[1]}.csv")

            print(f'Writing to {final_file}...')  
            result_table.write(final_file, overwrite=True)
            return 'Done!'
        else: 
            print('You need to define user_stream_task before continuing.')
            raise KeyboardInterrupt
    
    def stream_snap_task(self, IDs):
        strmID = IDs[0]
        snap = IDs[1]
        self.stream_snap_loader(strmID, snap)
        self.snap = snap
        log_memory()
        results_table = self.user_stream_task(strmID)
        results_table.add_column(strmID, name='stream_id', index = 0)
        results_table.add_column(snap, name='snap', index = 1)
        results_table.add_column(self.lookback_times[snap], name='lookback', index = 2)
        results_table.write(f"{self.cache_path}/{self.user_name}_{self.levelID}_{self.haloID}_{strmID}_{snap}.csv")
        return results_table

    def stream_snap_loader(self, strmID, snap, destination=None):
        if destination == None:
            destination = self
        with h5py.File(f"/eagle/Auriga/gthoron/stream_cats/galframe/{self.levelID}_{self.haloID}_{strmID}.hdf5", "r") as ledger:
            destination.strm_xyz = galcoords(np.asarray(ledger[f'{snap}/star_pos'])*u.kpc, vels=np.asarray(ledger[f'{snap}/star_vel'])*u.km/u.s, coordsys=None)
            destination.strm_masses = np.asarray(ledger[f'{snap}/star_mass'])
            destination.strm_ages = np.asarray(ledger[f'{snap}/age'])
            destination.strm_metals = np.array((np.asarray(ledger[f'{snap}/fe_h']), np.asarray(ledger[f'{snap}/mg_fe'])))
            destination.bound_stars = np.asarray(ledger[f'{snap}/bound'])
            destination.strm_dm_xyz = galcoords(np.asarray(ledger[f'{snap}/dm_pos'])*u.kpc, vels=np.asarray(ledger[f'{snap}/dm_vel'])*u.km/u.s, coordsys=None)
            destination.strm_dm_masses = np.asarray(ledger[f'{snap}/dm_mass'])
            try:
                destination.bound_stars = np.bool(np.asarray(ledger[f'{snap}/star_bound']))
                destination.bound_dm = np.bool(np.asarray(ledger[f'{snap}/dm_bound']))
                destination.strm_ids = np.asarray(ledger[f'{snap}/star_id'])
                destination.strm_dm_ids = np.asarray(ledger[f'{snap}/dm_id'])
            except:
                pass

        destination.strm_positions = np.array((
            destination.strm_xyz.x.value, 
            destination.strm_xyz.y.value,
            destination.strm_xyz.z.value)).T
        destination.strm_velocities = np.array((
            destination.strm_xyz.v_x.value, 
            destination.strm_xyz.v_y.value,
            destination.strm_xyz.v_z.value)).T
        destination.strm_dm_positions = np.array((
            destination.strm_dm_xyz.x.value, 
            destination.strm_dm_xyz.y.value,
            destination.strm_dm_xyz.z.value)).T
        destination.strm_dm_velocities = np.array((
            destination.strm_dm_xyz.v_x.value, 
            destination.strm_dm_xyz.v_y.value,
            destination.strm_dm_xyz.v_z.value)).T
        
        if self.include_halo or self.include_sky_info:
            try:
                centerTable = Table.read(f'/eagle/Auriga/gthoron/stream_plummer/cats/halo{self.haloID}_L{self.levelID}_allsnaps_streams_bound.csv')
                cutTable = centerTable[(centerTable['stream_id'] == strmID) & (centerTable['snap'] == snap)]
                assert len(cutTable) == 1
                destination.halo_pos = np.array((cutTable['center_x'], cutTable['center_y'], cutTable['center_z'])).T[0]*u.kpc
            except:
                print(f'Could not find /eagle/Auriga/gthoron/stream_plummer/cats/halo{self.haloID}_L{self.levelID}_allsnaps_streams_bound.csv')
                print(' Alternatively, there were two ')
                try:
                    kde = gaussian_kde(destination.strm_positions.T[destination.bound_stars], weights=destination.strm_masses[destination.bound_stars])
                    sample = destination.strm_positions[np.random.choice(
                        len(destination.strm_positions),
                        size=min(10000, len(destination.strm_positions)), replace=False
                   )]  
                    dens = kde(sample.T)
                    Npick = max(10, int(len(dens)*0.01))
                    idxs = np.argsort(dens)[-Npick:]
                    destination.halo_pos = np.average(sample[idxs], axis=0)*u.kpc
                except:
                    print('Gaussian KDE failed! Retrieving information...')
                    print(f'Number of bound particles {np.sum(destination.bound_stars)}')
                    print(f'Shape of init_sample {destination.strm_positions.shape}')
                    print('Attempting mean value of all bound particles...')
                    print(f'Shape of init_sample {destination.strm_positions.shape}')
                    kde = gaussian_kde(destination.strm_positions.T, weights=destination.strm_masses)
                    sample = destination.strm_positions[np.random.choice(len(destination.strm_positions), size=min(10000, len(destination.strm_positions)), replace=False)]
                    dens = kde(sample.T)
                    Npick = max(10, int(len(dens)*0.01))
                    idxs = np.argsort(dens)[-Npick:]
                    destination.halo_pos = np.average(sample[idxs], axis=0)*u.kpc

        if self.include_sky_info:
            #Getting the subhalo that matches the stream from the right snapshot
            subhaloID = stream_2_subhalo(self.tree, strmID, snap)

            #Getting the Half Mass Stellar Radius (HMSR)
            hmsr = self.halos['shmt'][subhaloID][4]*u.Mpc

            if hmsr.to(u.kpc).value < 1e-8:
                print(f'The half mass radius is very small! Issues could occur! Using pre-existing hmr: {hmsr.to(u.kpc)}...')

            #Halo Sky Positions
            halo_skycoords = galcoords(destination.halo_pos, coordsys=self.coordsys)
            print(f'    Halo position: {halo_skycoords.spherical.lon.to(u.degree).value} RA, {halo_skycoords.spherical.lat.to(u.degree).value} Dec')

            #The stream's position on the sky, centered on the subhalo identified above.
            #Gotta do a coordinate transformation before continuing
            destination.strm_skycoords = galcoords(
                destination.strm_xyz, coordsys=self.coordsys, initialized=True
            ).transform_to(coords.SkyOffsetFrame(origin=halo_skycoords))
            destination.strm_skycoords.representation_type=coords.SphericalRepresentation
            destination.strm_skycoords.differential_type=coords.SphericalDifferential
            destination.strm_skyra = destination.strm_skycoords.lon.value
            destination.strm_skydec = destination.strm_skycoords.lat.value

            destination.halo_skycoords=halo_skycoords
            destination.shInfo = (halo_skycoords.spherical.lon, halo_skycoords.spherical.lat, halo_skycoords.distance, hmsr)

    def multisnap_loader(self, strmID):
        with h5py.File(f"/eagle/Auriga/gthoron/stream_cats/galframe/{self.levelID}_{self.haloID}_{strmID}.hdf5", "r") as ledger:
            snaps = np.asarray(ledger['snaps'])
        self.stream_info = {'snaps': snaps}
        for snap in snaps:
            self.stream_info[f'{snap}'] = stream_info_holder()
            self.stream_snap_loader(strmID, snap, destination=self.stream_info[f'{snap}'])

    def snap_loader(self, snap=None):
        if snap != None:
            self.snap = snap
        print('')
        print(f'Reading in snapshot {self.snap}...')
        # I get the star particles
        snapdata =  self.load_snapshot_data(self.snap)
        # these ids will allow us to match later
        self.snapdata_ids = snapdata['id']

        #star particle info
        xyz = snapdata['pos'].to(u.kpc)
        vels = snapdata['vel'].to(u.km/u.s)
        
        #aligning with present day disk
        posinfo = align_n_rotate(self.matrix, {'pos': xyz.value, 'vel':vels.value})
        xyz = posinfo['pos']*u.kpc
        vels = posinfo['vel']*u.km/u.s
        del posinfo

        #Giving us a Sky_Coord variable but it's still in galactocentric coords
        self.cartpos=galcoords(xyz, vels=vels, coordsys=None)
        del xyz, vels

        self.masses = snapdata['mass'].to(u.Msun)
        fe = np.array(snapdata['gmet'][:,8])
        h = np.array(snapdata['gmet'][:,0])
        mg = snapdata['gmet'][:,6]
        mg[mg<= 1e-30] = 1e-30
        fe[fe<= 1e-30] = 1e-30
        snapdata['feh'] = np.log10(fe/h) - np.log10(55.845/1.008) - (7.5-12)
        snapdata['mgfe'] = np.log10(mg/fe) - np.log10(24.305/55.845) - (7.6-7.5)

        self.metalZ = np.array((snapdata['feh'], snapdata['mgfe']))
        self.ages = snapdata['age']
        #cleaning up
        del snapdata

        if self.load_dm:
            dm_snapdata = self.load_snapshot_data(self.snap, loadonly = ['id', 'pos', 'vel', 'mass'], loadonlytype=[1])
    
            # these ids will allow us to match later
            self.dm_snapdata_ids = dm_snapdata['id']

            #star particle info
            dm_xyz = dm_snapdata['pos'].to(u.kpc)
            dm_vels = dm_snapdata['vel'].to(u.km/u.s)
            self.dm_masses = dm_snapdata['mass'].to(u.Msun)
            del dm_snapdata

            posinfo = align_n_rotate(self.matrix, {'pos': dm_xyz.value, 'vel':dm_vels.value})
            dm_xyz = posinfo['pos']*u.kpc
            dm_vels = posinfo['vel']*u.km/u.s
            del posinfo
            self.dm_cartpos=galcoords(dm_xyz, vels=dm_vels, coordsys=None)
            del dm_xyz, dm_vels
 
    def stream_loader(self, strmID):
        print('')
        print(f'Reading in stream {strmID} for snapshot {self.snap}...')

        #Getting the Bound Stars
        #First identifying where strm is in the peak mass ids list
        iacc, = np.where(strmID == self.plists['pkmassid'])
        
        #Identifying indices in the bigger particle lists
        self.strm_idx = self.reader.load_stream_indices(strmID, self.plists['pkmassid'], self.plists['idacc'], self.snapdata_ids) 
        
        #Then, identifying the intersection of the two
        snap_idx = np.intersect1d(self.snapdata_ids, self.plists['idacc'][iacc], return_indices=True)[2]
        #Finally, whether the particles have been accreted or not.
        self.bound_stars= self.plists['aflag'][iacc][snap_idx]==1  
        
        #Grabbing the galactocentric coordinates for plotting
        self.strm_xyz = self.cartpos[self.strm_idx]
        self.strm_masses = self.masses[self.strm_idx]
        self.strm_metals = self.metalZ.T[self.strm_idx].T
        self.strm_ages = self.ages[self.strm_idx]
        if self.load_dm:
            self.dm_strm_idx = load_dm_ids(strmID, self.haloID, self.levelID, self.dm_snapdata_ids)
            self.strm_dm_xyz = self.dm_cartpos[self.dm_strm_idx]
            self.strm_dm_masses = self.dm_masses[self.dm_strm_idx]

        #Identifying that halo's position by a kernel density estimation
        if self.include_halo:
            positions = np.array((
                self.strm_xyz[self.bound_stars].x.value, 
                self.strm_xyz[self.bound_stars].y.value,
                self.strm_xyz[self.bound_stars].z.value)).T
            try:
                kde = gaussian_kde(positions.T, weights=self.strm_masses)
                sample = positions[np.random.choice(len(positions), size=min(10000, len(positions)), replace=False)]
                dens = kde(sample.T)
                Npick = max(10, int(len(dens)*0.01))
                idxs = np.argsort(dens)[-Npick:]
                self.halo_pos = np.average(sample[idxs], axis=0)*u.kpc
            except:
                print('Gaussian KDE failed! Retrieving information...')
                print(f'Number of bound particles {np.sum(self.bound_stars)}')
                print(f'Shape of init_sample {positions.shape}')
                print('Attempting mean value of all bound particles...')
                positions = np.array((self.strm_xyz.x.value, self.strm_xyz.y.value, self.strm_xyz.z.value)).T
                try:
                    kde = gaussian_kde(positions.T, weights=self.strm_masses)
                    sample = positions[np.random.choice(len(positions), size=min(10000, len(positions)), replace=False)]
                    dens = kde(sample.T)
                    Npick = max(10, int(len(dens)*0.01))
                    idxs = np.argsort(dens)[-Npick:]
                    self.halo_pos = np.average(sample[idxs], axis=0)*u.kpc
                except:
                    print("Uh oh, things still aren't working... raising an error")
                    raise KeyboardInterrupt
 
        if self.include_sky_info:
            #Getting the subhalo that matches the stream from the right snapshot
            subhaloID = stream_2_subhalo(self.tree, strmID, self.snap)

            #Getting the Half Mass Stellar Radius (HMSR)
            hmsr = self.halos['shmt'][subhaloID][4]*u.Mpc

            if hmsr.to(u.kpc).value < 1e-8:
                print(f'The half mass radius is very small! Issues could occur! Using pre-existing hmr: {hmsr.to(u.kpc)}...')

            #Halo Sky Positions
            halo_skycoords = galcoords(self.halo_pos, coordsys=self.coordsys)
            print(f'    Halo position: {halo_skycoords.spherical.lon.to(u.degree).value} RA, {halo_skycoords.spherical.lat.to(u.degree).value} Dec')

            #The stream's position on the sky, centered on the subhalo identified above.
            #Gotta do a coordinate transformation before continuing
            self.strm_skycoords = galcoords(
                self.strm_xyz, coordsys=self.coordsys, initialized=True
            ).transform_to(coords.SkyOffsetFrame(origin=halo_skycoords))
            self.strm_skycoords.representation_type=coords.SphericalRepresentation
            self.strm_skycoords.differential_type=coords.SphericalDifferential
            self.strm_skyra = self.strm_skycoords.lon.value
            self.strm_skydec = self.strm_skycoords.lat.value

            self.halo_skycoords=halo_skycoords
            self.shInfo = (halo_skycoords.spherical.lon, halo_skycoords.spherical.lat, halo_skycoords.distance, hmsr)

        
import halo_manager_aux
halo_manager_aux.halo_support_aux(halo_manager)
