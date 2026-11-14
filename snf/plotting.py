import numpy as np
import matplotlib.pyplot as plt
from snf.fields import grid_axes

def plot_sphere(theta, phi, radius=1.0, title='Measurement points'):
    '''
    Plot probe positions on the measurement sphere.

    theta (np.array): polar angle. Dimensions (I)
    phi (np.array): azimuthal angle. Dimensions (I)
    radius (float): sphere radius

    Returns fig, ax
    '''

    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    x, y, z = radius * st * cp, radius * st * sp, radius * ct

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Reference sphere
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 60), np.linspace(0, np.pi, 30))
    ax.plot_wireframe(radius*np.sin(v)*np.cos(u), radius*np.sin(v)*np.sin(u), radius*np.cos(v),
                      color='gray', alpha=0.15, linewidth=0.5)

    ax.scatter(x, y, z, color='crimson', s=12, depthshade=False)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    lim = 1.1 * radius
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)
    return fig, ax

def make_full_theta_cut(theta_deg, pos, neg):
    # phi = 0 (or 90) column -> +theta, phi = 180 (or 270) column -> -theta
    return np.r_[-theta_deg[::-1], theta_deg], np.r_[neg[::-1], pos]

def plot_EH_cuts(xyz, F, label='', title='', ylim=(-50, 5)):
    # Grid
    th_1d, ph_1d, _ = grid_axes(xyz)
    n_th, n_ph = th_1d.size, ph_1d.size
    theta_deg = np.degrees(th_1d)

    # phi columns for each cut
    nearest = lambda a: np.argmin(np.abs(np.angle(np.exp(1j * (ph_1d - a)))))

    # find tangential magnitude
    n = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    F_t = F - np.sum(F * n, axis=1)[:, None] * n
    F_dB = 20 * np.log10(np.linalg.norm(F_t, axis=1).reshape(n_th, n_ph))

    # Stitch opposite phi columns into full -180 to 180 deg cuts
    theta_e, F_e = make_full_theta_cut(theta_deg, F_dB[:, nearest(0)], F_dB[:, nearest(np.pi)])
    theta_h, F_h = make_full_theta_cut(theta_deg, F_dB[:, nearest(np.pi / 2)], F_dB[:, nearest(3 * np.pi / 2)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, th, F_cut, plane in ((ax1, theta_e, F_e, 'E'), (ax2, theta_h, F_h, 'H')):
        ax.plot(th, F_cut - F_cut.max(), 'r--', lw=1.5, label=label) # Normalize each cut to its own peak
        ax.set(xlabel=r'Theta (deg)', ylabel='Field strength (dB)', title=f'{title} {plane}-plane',
               xlim=(-180, 180), ylim=ylim)
        ax.grid(True)
        if label:
            ax.legend(loc='best')

    plt.tight_layout()
    return fig
