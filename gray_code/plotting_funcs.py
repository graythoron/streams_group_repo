import numpy as np
import sys
sys.path.append('/home/graythoron/projects/code')
from doing_funcs import galcoords
import matplotlib.pyplot as plt
import astropy.units as u
import corner
import matplotlib.lines as mlines 
import astropy.coordinates as coords


points = {'s': 2, 'rasterized': True}
combos = [(0,1), (0,2), (1,2)]
label = ['x', 'y', 'z']
labels = ["lon", "lat", "extension",  'ellipticity', "position_angle", 'truncate']



def ellipse(a, ell, rot_angle):
    thetas = np.linspace(0, 2*np.pi, 100)
    coss = np.cos(thetas+np.radians(rot_angle))
    sins = np.sin(thetas+np.radians(rot_angle))
    rrs = a*(1-ell)/np.sqrt((sins*(1-ell))**2+coss**2)
    return np.cos(thetas)*rrs, np.sin(thetas)*rrs

def ellipse3D(x,y):    
    thetas = np.linspace(0, 2*np.pi, 100)
    return np.array((x*np.cos(thetas), y*np.sin(thetas), thetas*0))

def inEllipse(data_x, data_y, a, ell, rot_ang):
    x_p = data_x*np.cos(np.radians(rot_ang)) - data_y*np.sin(np.radians(rot_ang))
    y_p = data_y*np.cos(np.radians(rot_ang)) + data_x*np.sin(np.radians(rot_ang))
    return (x_p/(1-ell))**2 + y_p**2 <= a**2

def inEllipsoid(data_x, data_y, data_z, a, b, c, matrix, center_pos):
    x_p = np.copy(data_x) - center_pos[0]
    y_p = np.copy(data_y) - center_pos[1]
    z_p = np.copy(data_z) - center_pos[2]
    position = np.fliplr(np.dot(np.array((x_p, y_p, z_p)).T, matrix))
    return (position[:, 0]/a)**2 + (position[:, 1]/b)**2 + (position[:, 2]/c)**2 <= 1

def DXY(height):
    if np.array(height).shape == (2,2):
        xs = height[0]
        ys = height[1]
        dx = xs[1]-xs[0]
        dy = ys[1]-ys[0]
        return dx, dy
    elif np.array(height).shape == (2,):
        dx = height[0]
        dy = height[1]
        return dx, dy
    else:
        print(f'Uh oh! Shape {np.array(height).shape} was not anticipated.')

def makeArrow(length, height, x_pos, y_pos, ax_obj, below=False, buff=1, c='k', label=''):
    pos_vec = np.array((x_pos,y_pos))
    dx, dy = DXY(height)
    if below:
        length*=-1
    len_vec = -length/np.sqrt(dx**2+dy**2)*np.array((dx, dy))
    pos_vec=pos_vec.reshape(len_vec.shape)
    ax_obj.annotate('', 
        xytext=pos_vec+(buff-1)*len_vec, textcoords='data',
        xy=pos_vec+buff*len_vec, xycoords='data', 
        size=15, arrowprops=dict(arrowstyle="simple", color=c, label=label)
    )

def pointingPlot(ax_obj, strmID, skyra, skydec, shHMR_ang, bound_stars, 
                 extension, ellipticity, pos_ang, pm, galcent=None, bins=50, bin_size=False):

    ax_obj.set_title(f'Stream ID {strmID}')
    ax_obj.set_facecolor((0.050383,0.029803, 0.527975, 1.0))

    lims = np.array([-1,1])*shHMR_ang

    ra_mask = (skyra>lims[0]) & (skyra<lims[1])
    dec_mask = (skydec>lims[0]) & (skydec<lims[1])
    true_mask = bound_stars & ra_mask & dec_mask

    elps = ellipse(a = extension, ell = ellipticity, rot_angle= pos_ang)

    h, xeds, yeds = np.histogram2d(skyra[true_mask], skydec[true_mask])
    bins = np.asarray(np.ceil(np.array([xeds[-1]-xeds[0], yeds[-1]-yeds[0]])/bin_size), dtype=int) 
    h, xeds, yeds = np.histogram2d(skyra[true_mask], skydec[true_mask], bins=bins)

    if np.max(h)<2:
        ax_obj.scatter(
            skyra[bound_stars], skydec[bound_stars],
            color=(0.940015, 0.975158, 0.131326, 1), 
            s=10, marker='.', alpha=0.66 
        )
    else:
        im = ax_obj.hist2d(
            skyra[true_mask], skydec[true_mask], 
            bins=bins, cmap='plasma',
            #label='Bound Stars'
        )
        plt.colorbar(im[3], ax=ax_obj)

    alpha = 0.20 - 0.02*np.log10(len(skyra))
    ax_obj.scatter(
        skyra[bound_stars==False], skydec[bound_stars==False], 
        alpha=alpha, color='w', **points, 
        #label='Unbound Stars'
    )

    ax_obj.plot(elps[0], elps[1], c='forestgreen')
    circ = ellipse(a = shHMR_ang, ell = 0, rot_angle= 0)
    ax_obj.plot(circ[0], circ[1], c='springgreen')

    makeArrow(
        extension*2, 
        (np.sin(np.radians(pos_ang)), np.cos(np.radians(pos_ang))),
        0,0, ax_obj, 
        below=True, buff=1.5, c='forestgreen', 
        #label='Rotation Angle'
    )
    makeArrow(
        extension*2, pm, 0,0, ax_obj, 
        buff=1.5, c='mediumorchid', below=True,
        #label='Proper Motion'
    )

    if galcent != None:
        makeArrow(
            extension*2, galcent, 0,0, ax_obj, 
            buff=1.5, c='darkorange', below=True,
            #label='Center of the galaxy'
        )

    ax_obj.set_xlim(lims)
    ax_obj.set_ylim(lims)
    ax_obj.set_aspect('equal')

def pointingPlotMoI(ax_obj, strmID, skyra, skydec, shHMR_ang, bound_stars, 
                 a_b_ellipse, a_c_ellipse, b_c_ellipse, 
                 extension, pm, galcent=None, bins=50, bin_size=False):

    ax_obj.set_title(f'Stream ID {strmID}')
    ax_obj.set_facecolor((0.050383,0.029803, 0.527975, 1.0))

    lims = np.array([-1,1])*shHMR_ang

    ra_mask = (skyra>lims[0]) & (skyra<lims[1])
    dec_mask = (skydec>lims[0]) & (skydec<lims[1])
    true_mask = bound_stars & ra_mask & dec_mask

    h, xeds, yeds = np.histogram2d(skyra[true_mask], skydec[true_mask])
    bins = np.asarray(np.ceil(np.array([xeds[-1]-xeds[0], yeds[-1]-yeds[0]])/bin_size), dtype=int) 
    h, xeds, yeds = np.histogram2d(skyra[true_mask], skydec[true_mask], bins=bins)

    if np.max(h)<2:
        ax_obj.scatter(
            skyra[bound_stars], skydec[bound_stars],
            color=(0.940015, 0.975158, 0.131326, 1), 
            s=10, marker='.', alpha=0.66 
        )
    else:
        im = ax_obj.hist2d(
            skyra[true_mask], skydec[true_mask], 
            bins=bins, cmap='plasma',
            #label='Bound Stars'
        )
        plt.colorbar(im[3], ax=ax_obj)

    alpha = 0.20 - 0.02*np.log10(len(skyra))
    ax_obj.scatter(
        skyra[bound_stars==False], skydec[bound_stars==False], 
        alpha=alpha, color='w', **points, 
        #label='Unbound Stars'
    )

    ax_obj.plot(a_b_ellipse.lon, a_b_ellipse.lat, c='cyan')
    ax_obj.plot(a_c_ellipse.lon, a_c_ellipse.lat, c='green')
    ax_obj.plot(b_c_ellipse.lon, b_c_ellipse.lat, c='red')
    circ = ellipse(a = shHMR_ang, ell = 0, rot_angle= 0)
    ax_obj.plot(circ[0], circ[1], c='springgreen')

    makeArrow(
        extension*2, pm, 0,0, ax_obj, 
        buff=1.5, c='mediumorchid', below=True,
        #label='Proper Motion'
    )

    if galcent != None:
        makeArrow(
            extension*2, galcent, 0,0, ax_obj, 
            buff=1.5, c='darkorange', below=True,
            #label='Center of the galaxy'
        )

    ax_obj.set_xlim(lims)
    ax_obj.set_ylim(lims)
    ax_obj.set_aspect('equal')

def superplot(hmrad_ang, fitting_info, bound_stars, xyz, halo, centered_sky_coords, apocenter, strmID,  outfile, halo_pos, debug=False):

    ra_nudge, dec_nudge, extension, ellipticity, pos_ang = fitting_info
    lims = np.array([-1,1])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), height_ratios=[4,4,4])
    gs = axes[0, 1].get_gridspec()
    # remove the underlying Axes
    for ax in axes[0, 1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, 1:])
    halo = halo.spherical

    skyra  = centered_sky_coords.lon.value-ra_nudge
    skydec = centered_sky_coords.lat.value-dec_nudge
    ellsID = inEllipse(skyra, skydec, extension, ellipticity, pos_ang)
    pm = (
        np.nanmean(centered_sky_coords.pm_lon[ellsID]).value, 
        np.nanmean(centered_sky_coords.pm_lat[ellsID]).value,
        np.nanmean(centered_sky_coords.radial_velocity[ellsID]).value
    )
    print(f'Proper Motion')
    print(f'    {pm}')

    pointingPlot(
        axes[0,0], strmID, skyra, skydec, hmrad_ang, bound_stars,
        extension, ellipticity, pos_ang, pm[:2], 
        galcent=(-halo.lon.value,-halo.lat.value), bin_size=1/20
    )
    axes[0,0].set_xlabel('Lon (deg)')
    axes[0,0].set_ylabel('Lat (deg)')

    alpha = 0.137647 - 0.0188235*np.log10(len(xyz))
    axbig.scatter(skyra, skydec, c='k', alpha = alpha, **points)
    elps = ellipse(a = extension, ell = ellipticity, rot_angle= pos_ang)
    axbig.plot(elps[0], elps[1], c='forestgreen')
    circ = ellipse(a = hmrad_ang, ell = 0, rot_angle= 0)
    axbig.plot(circ[0], circ[1], c='springgreen')
    axbig.set_xlabel('Lon (deg)')
    axbig.set_ylabel('Lat (deg)')
    axbig.set_xlim(lims*6*hmrad_ang)
    axbig.set_ylim(lims*3*hmrad_ang)
    if debug:
        axbig.set_xlim(lims*180)
        axbig.set_ylim(lims*90)

    limit = np.round(apocenter+25)
    if limit >= 25:
        bins = np.arange(-limit, limit, 1)
    else:
        bins = np.arange(-25, 25, 1)

    for ii in range(3):
        j, k = combos[ii]
        axes[1, ii].scatter(
            xyz[:,j], xyz[:,k], 
            alpha=alpha, color='k', **points
        ) 
        axes[2, ii].hist2d(
            xyz[:,j], xyz[:,k], 
            bins=bins, norm='log'
        )
        axes[1, ii].set_ylabel(label[k])
        axes[2, ii].set_xlabel(label[j])
        axes[2, ii].set_ylabel(label[k])
        axes[1, ii].scatter(
            halo_pos[j], halo_pos[k], 
            color='orange', marker='x', s=10
        ) 
        axes[2, ii].scatter(
            halo_pos[j], halo_pos[k], 
            color='orange', marker='x', s=10
        ) 

    for axi in axes[1:].flatten():
        axi.set_xlim(-limit, limit)
        axi.set_ylim(-limit, limit)
        axi.set_aspect('equal')


    plt.tight_layout()
    fig.savefig(f'/eagle/Auriga/gthoron/stream_plummer/plots/{outfile}.png')
    return pm

def superplotMoI(hmrad_ang, axises, matrix, bound_stars, xyz, _halo_, centered_sky_coords, apocenter, coordsys, strmID,  outfile, halo_pos, debug=False):

    (a, b, c) = axises
    lims = np.array([-1,1])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), height_ratios=[4,4,4])
    gs = axes[0, 1].get_gridspec()
    # remove the underlying Axes
    for ax in axes[0, 1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, 1:])
    halo = _halo_.spherical

    skyra, skydec = centered_sky_coords.lon.value, centered_sky_coords.lat.value
    ellsID = inEllipsoid(xyz[:,0],xyz[:,1],xyz[:,2], a, b, c, matrix, halo_pos)
    pm = (
        np.nanmean(centered_sky_coords.pm_lon[ellsID]).value, 
        np.nanmean(centered_sky_coords.pm_lat[ellsID]).value,
        np.nanmean(centered_sky_coords.radial_velocity[ellsID]).value
    )
    print(f'Proper Motion')
    print(f'    {pm}')

    a_b_ellipse = ellipse3D(a, b)
    a_c_ellipse = ellipse3D(a, c)
    b_c_ellipse = ellipse3D(b, c)

    a_b_ellipse = np.array((a_b_ellipse[0], a_b_ellipse[1], a_b_ellipse[2])).T
    a_c_ellipse = np.array((a_c_ellipse[0], a_c_ellipse[2], a_c_ellipse[1])).T
    b_c_ellipse = np.array((b_c_ellipse[2], b_c_ellipse[0], b_c_ellipse[1])).T

    a_b_rot_ellip = np.fliplr(np.dot(a_b_ellipse, matrix))*u.kpc+halo_pos
    a_c_rot_ellip = np.fliplr(np.dot(a_c_ellipse, matrix))*u.kpc+halo_pos
    b_c_rot_ellip = np.fliplr(np.dot(b_c_ellipse, matrix))*u.kpc+halo_pos

    a_b_sky_ellip = galcoords(a_b_rot_ellip, coordsys = coordsys).transform_to(coords.SkyOffsetFrame(origin=_halo_))
    a_b_sky_ellip.representation_type=coords.SphericalRepresentation
    a_b_sky_ellip.differential_type=coords.SphericalDifferential
    a_c_sky_ellip = galcoords(a_c_rot_ellip, coordsys = coordsys).transform_to(coords.SkyOffsetFrame(origin=_halo_))
    a_c_sky_ellip.representation_type=coords.SphericalRepresentation
    a_c_sky_ellip.differential_type=coords.SphericalDifferential
    b_c_sky_ellip = galcoords(b_c_rot_ellip, coordsys = coordsys).transform_to(coords.SkyOffsetFrame(origin=_halo_))
    b_c_sky_ellip.representation_type=coords.SphericalRepresentation
    b_c_sky_ellip.differential_type=coords.SphericalDifferential

    pointingPlotMoI(
        axes[0,0], strmID, skyra, skydec, hmrad_ang, bound_stars,
        a_b_sky_ellip, a_c_sky_ellip, b_c_sky_ellip,
        a, pm[:2], 
        galcent=(-halo.lon.value,-halo.lat.value), bin_size=1/20
    )
    axes[0,0].set_xlabel('Lon (deg)')
    axes[0,0].set_ylabel('Lat (deg)')

    alpha = 0.137647 - 0.0188235*np.log10(len(xyz))
    axbig.scatter(skyra, skydec, c='k', alpha = alpha, **points)
    axbig.plot(a_b_sky_ellip.lon, a_b_sky_ellip.lat, c='blue')
    axbig.plot(a_c_sky_ellip.lon, a_c_sky_ellip.lat, c='green')
    axbig.plot(b_c_sky_ellip.lon, b_c_sky_ellip.lat, c='red')
    circ = ellipse(a = hmrad_ang, ell = 0, rot_angle= 0)
    axbig.plot(circ[0], circ[1], c='springgreen')
    axbig.set_xlabel('Lon (deg)')
    axbig.set_ylabel('Lat (deg)')
    axbig.set_xlim(lims*6*hmrad_ang)
    axbig.set_ylim(lims*3*hmrad_ang)
    if debug:
        axbig.set_xlim(lims*180)
        axbig.set_ylim(lims*90)

    limit = np.round(apocenter+25)
    bins = np.arange(-limit, limit, 1)

    for ii in range(3):
        j, k = combos[ii]
        axes[1, ii].scatter(
            xyz[:,j], xyz[:,k], 
            alpha=alpha, color='k', **points
        ) 
        axes[2, ii].hist2d(
            xyz[:,j].value, xyz[:,k].value, 
            bins=bins, norm='log'
        )
        axes[1, ii].set_ylabel(label[k])
        axes[2, ii].set_xlabel(label[j])
        axes[2, ii].set_ylabel(label[k])
        axes[1, ii].scatter(
            halo_pos[j], halo_pos[k], 
            color='orange', marker='x', s=10
        ) 
        axes[2, ii].scatter(
            halo_pos[j], halo_pos[k], 
            color='orange', marker='x', s=10
        ) 
        axes[1, ii].plot(a_b_rot_ellip[:, j], a_b_rot_ellip[:, k], c='blue') 
        axes[2, ii].plot(a_b_rot_ellip[:, j], a_b_rot_ellip[:, k], c='blue') 
        axes[1, ii].plot(a_c_rot_ellip[:, j], a_c_rot_ellip[:, k], c='green') 
        axes[2, ii].plot(a_c_rot_ellip[:, j], a_c_rot_ellip[:, k], c='green') 
        axes[1, ii].plot(b_c_rot_ellip[:, j], b_c_rot_ellip[:, k], c='red') 
        axes[2, ii].plot(b_c_rot_ellip[:, j], b_c_rot_ellip[:, k], c='red') 

    for axi in axes[1:].flatten():
        axi.set_xlim(-limit, limit)
        axi.set_ylim(-limit, limit)
        axi.set_aspect('equal')

    for ii in range(3):
        j, k = combos[ii]
        if a==a:
            limits = 5*a*np.array((-1, 1))
        else:
            limits = 5*np.array((-1, 1))
        axes[2, ii].set_xlim(halo_pos[[j,k]].value[0] + limits[0], halo_pos[[j,k]].value[0] + limits[1])
        axes[2, ii].set_ylim(halo_pos[[j,k]].value[1] + limits[0], halo_pos[[j,k]].value[1] + limits[1])
        axes[2, ii].set_aspect('equal')


    #plt.tight_layout()
    fig.savefig(f'/eagle/Auriga/gthoron/stream_plummer/plots/{outfile}_MoI.png')
    #return pm

def cornerplot(samples, outfile):

    fig = corner.corner(samples, labels = labels, bins = 35,
                show_titles=True, title_kwargs={"fontsize": 12},
                quantiles=(0.16, 0.5, 0.84), plot_contours = True,
                fill_contours=True, smooth=True, plot_datapoints = False)
    
    fig.savefig(f'/eagle/Auriga/gthoron/stream_plummer/plots/{outfile}_corner.png')
    
def chainplot(chain, outfile):
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)

    for i in range(len(labels)-1):
        ax = axes[i]
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(chain[:, :, 0]))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    fig.savefig(f'/eagle/Auriga/gthoron/stream_plummer/plots/{outfile}_chains.png')



bound_handle = mlines.Line2D(
    [], [], 
    color=(0.940015, 0.975158, 0.131326, 1),
    ls='', label='Bound stars', marker='.', markersize=10
)
unbound_handle = mlines.Line2D([], [], color='w', ls='', marker='.', label='Unbound stars', markersize=10)
posang_handle = mlines.Line2D([], [], color='forestgreen', marker='>',
                          markersize=15, label='Position Angle')
cog_handle = mlines.Line2D([], [], color='darkorange', marker='>',
                          markersize=15, label='Center of the Galaxy')
pm_handle = mlines.Line2D([], [], color='mediumorchid', marker='>',markersize=15, 
                          label='Proper Motion')
