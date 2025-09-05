import numpy as np
import astropy.coordinates as coords
import astropy.units as u
from scipy.stats import gaussian_kde
import sys
import os
from arviz import hdi, kde
from astropy.table import QTable, vstack
sys.path.append('/home/graythoron/auriga_streams/code')
from select_streams import load_combined_table

def radius_mask(ra, dec, rad):
    mask = np.sqrt((ra)**2+(dec)**2) <= rad
    return mask

def stream_2_subhalo(tree, pkmassid, nsnap=0):
    #Identifies halo id from stream id by following the merger tree and looking for the pkmassid 
    index, desc, desc_first_prog = [pkmassid] * 3
    time = tree.data['snum'][index]
    if time < nsnap:
        while (index == desc_first_prog) and (time!=nsnap):
            index = desc
            desc = tree.data['desc'][index]
            desc_first_prog = tree.data['fpin'][desc]
            time = tree.data['snum'][desc]
    else:
        while (index == desc_first_prog) & (time!=nsnap):
            index = desc
            desc = tree.data['fpin'][index]
            desc_first_prog = tree.data['desc'][desc]
            time = tree.data['snum'][index]

    final_tree_index = index

    subhalo = tree.data['sbnr'][final_tree_index]
    return subhalo


def center_n_align(matrix, pos_data, host_coords):
    data = {'pos':np.copy(pos_data['pos']), 'vel': np.copy(pos_data['vel'])}
    # center on host
    data['pos'] -= host_coords[:3]
    data['vel'] -= host_coords[3:]
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
        coordinates = xyz
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

def center_coords_on(sky_coords, origin):
    centered_coords = sky_coords.transform_to(coords.SkyOffsetFrame(origin=origin))
    centered_coords.representation_type=coords.SphericalRepresentation
    centered_coords.differential_type=coords.SphericalDifferential
    return centered_coords, centered_coords.lon.value, centered_coords.lat.value

def output_file(haloID, levelID, strmID, save_dir, coordsys, ep, nsnap, continuation):
    filename = f'halo{haloID}_stream{strmID}_L{levelID}_crds{coordsys}_ep{ep}_snap{nsnap}'
    outfile_string=f'{save_dir}/output/{filename}'
    outfile=f'{outfile_string}.h5'
    if continuation and os.path.exists(outfile):
        print(f'Continuing for Halo {haloID} Stream {strmID}...')
        print(f'Reading {outfile} ...')
    elif os.path.exists(outfile):
        print(f'{outfile} exists ... creating new file ...')
        file_filled = True
        vers = 1
        while file_filled:
            outfile = f'{outfile_string}_vers{vers}.h5'
            file_filled = os.path.exists(outfile)
            vers += 1
        print(f'Saving to {outfile} ...')
    else:
        print(f'Running new for Halo {haloID} Stream {strmID}...')
        print(f'Saving to {outfile} ...')
    return outfile, filename

def density_median(sample, circular=False):
    sample_kde = kde(sample, circular=circular)
    return sample_kde[0][np.lexsort(sample_kde)[-1]]

def output_dir(directory):
    if os.path.isdir(directory): pass
    else: os.makedirs(directory, exist_ok=True)

def if_final_snap(reader, nsnap, coordsys):
    if nsnap == None: nsnap = reader.final_snap
    if reader.final_snap==nsnap: pass
    else: coordsys='spherical'
    return reader.final_snap==nsnap, coordsys, nsnap

def all_that_jazz(reader, levelID, haloID, nsnap, is_final_snap, rotinfo=None, f_bound_min=0):
    #This is me checking whether the snap is the final snap 
    # and if it is not, making sure we don't try to rotate to that disk
    # as the align_with_disk is meant for present day.
    
    # I get the star particles
    snapdata = reader.load_snapshot_data(
        nsnap = nsnap,
        reading=True, 
        saving=False, 
        center_on_host=True, 
        units=True, 
        loadonly = ['id', 'pos', 'vel', 'mass', 'age', 'gmet'],
        align_with_disk=is_final_snap
    )
    # these ids will allow us to match later
    snapdata_ids = snapdata['id']

    #star particle info
    xyz = snapdata['pos'].to(u.kpc)
    vels = snapdata['vel'].to(u.km/u.s)
    masses = snapdata['mass'].to(u.Msun)
    fe = snapdata['gmet'][:,8]
    h = snapdata['gmet'][:,0]
    mg = snapdata['gmet'][:,6]
    snapdata['feh'] = np.log10(fe/h) - np.log10(55.845/1.008) - (7.5-12)
    snapdata['mgfe'] = np.log10(mg/fe) - np.log10(24.305/55.845) - (7.6-7.5)

    metals = np.array((snapdata['feh'], snapdata['mgfe']))

    if rotinfo == None:
        #I get the particle lists.
        plists = reader.load_particle_lists(reading=True, saving=False)
        # and the merger tree
        tree = reader.load_tree(nsnap=reader.final_snap)
        # and the halo catalog
        halos = reader.load_subfind().data

        #data for center_n_align
        host_coords = reader.get_host_coords(nsnap=reader.final_snap)
        rotmat = snapdata['matrix'].transpose()
         #Grabbing the big stream catalog from Nora and Alex
        metricsTable = load_combined_table()
        #Making some sensible cuts
        metricMask = (
            (metricsTable['level']==levelID) &
            (metricsTable['halo']==haloID) & 
            (metricsTable['f_bound_part']>f_bound_min)
        )
        stream_met_table = QTable.from_pandas(metricsTable[metricMask])
        #Grabbing the stream ids for later use
        streamids = stream_met_table['stream_id']
    else:
        posinfo = center_n_align(rotinfo[0], {'pos': xyz.value, 'vel':vels.value}, rotinfo[1])
        xyz = posinfo['pos']*u.kpc
        vels = posinfo['vel']*u.km/u.s
        del posinfo

    #cleaning up
    del snapdata

    #Giving us a Sky_Coord variable but it's still in galactocentric coords
    cart_coords=galcoords(xyz, vels=vels, coordsys=None)
    del xyz, vels

    #Regurgitating it out
    if rotinfo == None:
        return (plists, tree, halos, 
            snapdata_ids,
            masses, metals, 
            host_coords, rotmat, 
            cart_coords, 
            stream_met_table, streamids) 
    else:
        return (snapdata_ids,
            masses, metals, 
            cart_coords) 

def dark_jazz(reader, nsnap, rotinfo=None):
    #This is me checking whether the snap is the final snap 
    # and if it is not, making sure we don't try to rotate to that disk
    # as the align_with_disk is meant for present day.
    
    # I get the star particles
    snapdata = reader.load_snapshot_data(
        nsnap = nsnap,
        reading=True, 
        saving=False, 
        center_on_host=True, 
        units=True, 
        loadonly = ['id', 'pos', 'vel', 'mass'],
        loadonlytype=[1]
    )
    # these ids will allow us to match later
    snapdata_ids = snapdata['id']

    #star particle info
    xyz = snapdata['pos'].to(u.kpc)
    vels = snapdata['vel'].to(u.km/u.s)
    masses = snapdata['mass'].to(u.Msun)

    if rotinfo == None:
        pass
    else:
        posinfo = center_n_align(rotinfo[0], {'pos': xyz.value, 'vel':vels.value}, rotinfo[1])
        xyz = posinfo['pos']*u.kpc
        vels = posinfo['vel']*u.km/u.s
        del posinfo

    #cleaning up
    del snapdata

    #Giving us a Sky_Coord variable but it's still in galactocentric coords
    cart_coords=galcoords(xyz, vels=vels, coordsys=None)
    del xyz, vels

    #Regurgitating it out
    if rotinfo == None:
        return (snapdata_ids,
            masses, 
            cart_coords) 
    else:
        return (snapdata_ids,
            masses, 
            cart_coords) 


def stream_jazz(reader, strmID, plists, snapdata_ids, tree, nsnap, halos, cart_coords, masses, metalz, coordsys):
    #Identifying indices in the bigger particle lists
    strm_idx = reader.load_stream_indices(strmID, plists['pkmassid'], plists['idacc'], snapdata_ids) 
    
        #Getting the Bound Stars
        #First identifying where strm is in the peak mass ids list
    iacc, = np.where(strmID == plists['pkmassid'])
        #Then, identifying the intersection of the two
    snap_idx = np.intersect1d(snapdata_ids, plists['idacc'][iacc], return_indices=True)[2]
        #Finally, whether the particles have been accreted or not.
    bound_stars= plists['aflag'][iacc][snap_idx]==1  
    
    #Grabbing the galactocentric coordinates for plotting
    xyzpos = cart_coords[strm_idx]

    #Getting the subhalo that matches the stream from the right snapshot
    subhaloID = stream_2_subhalo(tree, strmID, nsnap)

    #Identifying that halo's position by a kernel density estimation
    init_sample = np.array((xyzpos[bound_stars].x.value, xyzpos[bound_stars].y.value,xyzpos[bound_stars].z.value))
    try:
        halo_kde = gaussian_kde(init_sample)
        halo_pos = init_sample.T[np.argmax(halo_kde.pdf(init_sample))]*u.kpc
    except:
        print('Gaussian KDE failed! Retrieving information...')
        print(f'Number of bound particles {np.sum(bound_stars)}')
        print(f'Shape of init_sample {init_sample.shape}')
        print('Attempting mean value of all bound particles...')
        xyz = np.array((xyzpos.x.value, xyzpos.y.value, xyzpos.z.value))
        try:
            halo_kde = gaussian_kde(xyz)
            halo_pos = np.mean(init_sample, axis=1)*u.kpc
        except:
            print("Uh oh, things still aren't working... raising an error")
            raise KeyboardInterrupt
    
    #Getting the Half Mass Stellar Radius (HMSR)
    hmsr = halos['shmt'][subhaloID][4]*u.Mpc

    if hmsr.to(u.kpc).value < 1e-8:
        print(f'Half Mass Radius is too small! ({hmsr.to(u.kpc)})')
        print(f'Searching for a new half mass radius...')
        xyz = np.array((xyzpos.x.value, xyzpos.y.value, xyzpos.z.value))
        try:
            kde_sample = halo_kde.pdf(xyz)
            kde_norms = np.linalg.norm(xyz[kde_sample>np.max(kde_sample)/2]-xyz[np.argmax(kde_sample)], axis=1)
            hmsr = np.max(kde_norms)*u.kpc
            print(f'New half mass radius: {hmsr}')
        except:
            print('Looking for a new half mass radius failed! Keeping old half mass radius.')

    #Halo Sky Positions
    halo_sky_coords = galcoords(halo_pos, coordsys=coordsys)
    print(f'    Halo position: {halo_sky_coords.spherical.lon.to(u.degree).value} RA, {halo_sky_coords.spherical.lat.to(u.degree).value} Dec')

    #The stream's position on the sky, centered on the subhalo identified above.
    stream_sky_coords, skyra, skydec = center_coords_on(
        #Gotta do a coordinate transformation before continuing
        galcoords(cart_coords[strm_idx], coordsys=coordsys, initialized=True),
        halo_sky_coords
    )

    #stream_max_pos = xyzpos[np.argmax()]

    #Subhalo info to return
    shInfo = (halo_sky_coords.spherical.lon, halo_sky_coords.spherical.lat, halo_sky_coords.distance, hmsr)

    return xyzpos, stream_sky_coords, skyra, skydec, shInfo, masses[strm_idx], metalz.T[strm_idx].T, bound_stars, halo_pos, halo_sky_coords
