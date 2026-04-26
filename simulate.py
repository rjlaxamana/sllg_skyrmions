import numpy as np
import pandas as pd
from scipy.special import genlaguerre
from tqdm import tqdm
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- FIXED PARAMETERS ---
N = 30
L = 2*N
a = 1
h = a*np.sqrt(3)/2
J1 = -1
J3 = 0.5
p = 0
m = 5
initial_state = "ferro"
alpha = 0.1
gamma = 1.0
dt = 0.02
t0 = 50
steps = int((t0+50)/dt)+1

# --- UTILITIES ---
def coordinate_pairs(x_coords, y_coords):
    coordinates = []
    count = 0
    X = 0
    Y = 0
    r = np.zeros((L, L, 3))
    z = 0

    for x, y in zip(x_coords, y_coords):
        coordinates.append((x, y))
        if not count % (L) and count!=0:
            flag = True
        else:
            flag = False
        if flag:
            Y += 1
            X = 0
        r[Y, X] = [x, y, z]
        count += 1
        X += 1
    return coordinates, r

def get_intensity(x_coords, y_coords, w):
    def max_intensity(x_coords, y_coords):
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        R = np.sqrt(x_coords**2 + y_coords**2)
        phi = np.arctan2(y_coords, x_coords)
        laguerre = genlaguerre(p, m)(2*R**2/(w**2))
        sqrt_intensity = (R/w)**(abs(m))*np.exp(-((R/w)**2)+ 1j*m*phi)*laguerre/np.sqrt(w)
        intensity = abs(sqrt_intensity)**2
        max_intensity = intensity.max()
        return max_intensity

    intensity = np.zeros((L, L))
    coords, _ = coordinate_pairs(x_coords, y_coords)

    count = 0
    X = 0
    Y = 0

    for coord in coords:
        x, y = coord
        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        laguerre = genlaguerre(p, m)(2*R**2/(w**2))
        sqrt_intensity = (R/w)**(abs(m))*np.exp(-((R/w)**2)+ 1j*m*phi)*laguerre/np.sqrt(w)
        local_intensity = abs(sqrt_intensity)**2/max_intensity(x_coords, y_coords)
        if not count % (L) and count!=0:
            flag = True
        else:
            flag = False
        if flag:
            Y += 1
            X = 0
        intensity[Y, X] = local_intensity
        count += 1
        X += 1

    return intensity

def thermal_field(t, T0, intensity, seed):
    if t == 0 or t > t0:
        T = 0
    else:
        T = T0 * (1-t/t0)
    temperature = T * intensity
    
    random = np.random.default_rng(seed=seed*int(t/dt))

    sigma = np.sqrt(2 * alpha * temperature/dt)
    result = random.normal(0, sigma[:, :, np.newaxis], size=(L, L, 3))

    return result

def normalize(spins):
    norm = np.linalg.norm(spins, axis=-1, keepdims=True)
    return np.where(norm == 0, spins, spins / norm)

def calculate_total_energy(spins, H, A, Ja):
    total_energy = 0

    left = np.roll(spins, shift=-1, axis=1) # spin to the left of the current spin
    right = np.roll(spins, shift=1, axis=1) # spin to the right of the current spin
    up = np.roll(spins, shift=1, axis=0) # upper right spin if y is odd, else upper left spin
    down = np.roll(spins, shift=-1, axis=0) # lower right spin if y is odd, else lower left spin

    j1_interaction = np.zeros((L, L))
    ja_interaction = np.zeros((L, L))

    j1_interaction += np.sum(spins * left, axis=-1) # exchange interaction to the left spin
    j1_interaction += np.sum(spins * right, axis=-1) # exchange interaction to the right spin
    j1_interaction += np.sum(spins * up, axis=-1) # exchange interaction to the upper spin
    j1_interaction += np.sum(spins * down, axis=-1) # exchange interaction to the lower spin

    angle2 = 2*np.pi/3 # bond direction at 120 (and by extension 300)
    angle3 = 4*np.pi/3 # bond direction at 240 (and by extension 60)

    for y in range(L):
        spin_x = spins[y, :, 0]
        spin_y = spins[y, :, 1]

        if not y % 2: #if y is odd
            up_adj = np.roll(spins, shift=(-1, 1), axis=(1, 0)) # upper left spin
            down_adj = np.roll(spins, shift=(-1, -1), axis=(1, 0)) # lower left spin
        
            # angle2 bond direction corresponds to the upper left spin and lower right spin
            delta2x = spin_x*up_adj[y, :, 0]*np.cos(angle2) - spin_x*up_adj[y, :, 1]*np.sin(angle2) 
            delta2y = -1*spin_y*up_adj[y, :, 1]*np.cos(angle2) - spin_y*up_adj[y, :, 0]*np.sin(angle2)

            delta2x += spin_x*down[y, :, 0]*np.cos(angle2) - spin_x*down[y, :, 1]*np.sin(angle2)
            delta2y += -1*spin_y*down[y, :, 1]*np.cos(angle2) - spin_y*down[y, :, 0]*np.sin(angle2)

            # angle3 bond direction corresponds to the lower left spin and upper right spin
            delta3x = spin_x*down_adj[y, :, 0]*np.cos(angle3) - spin_x*down_adj[y, :, 1]*np.sin(angle3)
            delta3y = -1*spin_y*down_adj[y, :, 1]*np.cos(angle3) - spin_y*down_adj[y, :, 0]*np.sin(angle3)

            delta3x += spin_x*up[y, :, 0]*np.cos(angle3) - spin_x*up[y, :, 1]*np.sin(angle3)
            delta3y += -1*spin_y*up[y, :, 1]*np.cos(angle3) - spin_y*up[y, :, 0]*np.sin(angle3)
        else:
            up_adj = np.roll(spins, shift=(1, 1), axis=(1, 0))   # upper right spin
            down_adj = np.roll(spins, shift=(1, -1), axis=(1, 0)) # lower right spin

            # angle2 bond direction corresponds to the upper left spin and lower right spin
            delta2x = spin_x*up[y, :, 0]*np.cos(angle2) - spin_x*up[y, :, 1]*np.sin(angle2)
            delta2y = -1*spin_y*up[y, :, 1]*np.cos(angle2) - spin_y*up[y, :, 0]*np.sin(angle2)

            delta2x += spin_x*down_adj[y, :, 0]*np.cos(angle2) - spin_x*down_adj[y, :, 1]*np.sin(angle2)
            delta2y += -1*spin_y*down_adj[y, :, 1]*np.cos(angle2) - spin_y*down_adj[y, :, 0]*np.sin(angle2)

            # angle3 bond direction corresponds to the lower left spin and upper right spin
            delta3x = spin_x*down[y, :, 0]*np.cos(angle3) - spin_x*down[y, :, 1]*np.sin(angle3)
            delta3y = -1*spin_y*down[y, :, 1]*np.cos(angle3) - spin_y*down[y, :, 0]*np.sin(angle3)

            delta3x += spin_x*up_adj[y, :, 0]*np.cos(angle3) - spin_x*up_adj[y, :, 1]*np.sin(angle3)
            delta3y += -1*spin_y*up_adj[y, :, 1]*np.cos(angle3) - spin_y*up_adj[y, :, 0]*np.sin(angle3)

        j1_interaction[y, :] += np.sum(spins[y, :] * up_adj[y, :], axis=-1) # exchange interaction to the upper spin
        j1_interaction[y, :] += np.sum(spins[y, :] * down_adj[y, :], axis=-1) # exchange interaction to the lower spin

        # angle1 bond direction corresponds to the left and right spins
        delta1x = right[y, :, 0] + left[y, :, 0] 
        delta1y = -1*right[y, :, 1] -left[y, :, 1]

        # adding the contributions from all bond directions
        ja_interaction[y, :] += 2*Ja*(delta1x + delta2x + delta3x)
        ja_interaction[y, :] += 2*Ja*(delta1y + delta2y + delta3y)
    
    # spins for j3 interaction (similar to how j1 interaction was calculated)
    left2 = np.roll(spins, shift=-2, axis=1)
    right2 = np.roll(spins, shift=2, axis=1)
    up_left2 = np.roll(spins, shift=(-1, 2), axis=(1, 0))
    down_left2 = np.roll(spins, shift=(-1, -2), axis=(1, 0))
    up_right2 = np.roll(spins, shift=(1, 2), axis=(1, 0))
    down_right2 = np.roll(spins, shift=(1, -2), axis=(1, 0))

    j3_interaction = np.zeros((L, L))

    j3_interaction += np.sum(spins * left2, axis=-1)
    j3_interaction += np.sum(spins * right2, axis=-1)
    j3_interaction += np.sum(spins * up_left2, axis=-1)
    j3_interaction += np.sum(spins * down_left2, axis=-1)
    j3_interaction += np.sum(spins * up_right2, axis=-1)
    j3_interaction += np.sum(spins * down_right2, axis=-1)
    
    j1_energy = J1 * np.sum(j1_interaction)/2 # divided by 2 because it sums over the entire lattice twice
    j3_energy = J3 * np.sum(j3_interaction)/2
    ja_energy = np.sum(ja_interaction)/2

    exchange_energy = j1_energy + j3_energy + ja_energy

    zeeman_energy = -H * np.sum(spins[:, :, 2])

    anisotropy_energy = -A * np.sum(spins[:, :, 2]**2)

    total_energy = exchange_energy + zeeman_energy + anisotropy_energy

    return total_energy/(L**2)

def save_csv(x_coords, y_coords, spins, path, t):
    spin_config = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "m_x": spins[:, :, 0].flatten(),
        "m_y": spins[:, :, 1].flatten(),
        "m_z": spins[:, :, 2].flatten()
    })
    spin_config.to_csv(path+f"spin_{t}.csv", index=False)

def skyrmion_number(spins):
    solid_angle_sum = np.zeros((L, L))

    s1 = spins
    s2 = np.roll(spins, shift=1, axis=1) # spin to the right of s1
    s3 = np.roll(spins, shift=-1, axis=0) # lower right spin if y is odd, else lower left spin
    s4 = np.roll(s3, shift=1, axis=1) # lower right right spin if y is odd, else lower right spin

    tri_up_num = np.zeros((L, L))
    tri_up_den = np.zeros((L, L))
    tri_down_num = np.zeros((L, L))
    tri_down_den = np.zeros((L, L))

    for y in range(L):
        if not y%2: # if odd, the up triangle is s2->s3->s4 and the down triangle is s2->s1->s3
            tri_up_num[y, :] = np.sum(s2[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            tri_up_den[y, :] = (1+np.sum(s2[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s2[y, :], axis=-1))
            tri_down_num[y, :] = np.sum(s2[y, :]*np.cross(s1[y, :], s3[y, :]), axis=-1)
            tri_down_den[y, :] = (1+np.sum(s2[y, :]*s1[y, :], axis=-1)+np.sum(s1[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s2[y, :], axis=-1))
        else: # if even, the up triangle is s1->s3->s4 and the down triangle is s2->s1->s4
            tri_up_num[y, :] = np.sum(s1[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            tri_up_den[y, :] = (1+np.sum(s1[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s1[y, :], axis=-1))
            tri_down_num[y, :] = np.sum(s2[y, :]*np.cross(s1[y, :], s4[y, :]), axis=-1)
            tri_down_den[y, :] = (1+np.sum(s2[y, :]*s1[y, :], axis=-1)+np.sum(s1[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s2[y, :], axis=-1))
    
    solid_angle_sum = np.sum(2*np.arctan2(tri_up_num, tri_up_den)+2*np.arctan2(tri_down_num, tri_down_den))
    skx = solid_angle_sum/(4*np.pi)

    return round(skx)

def abs_skyrmion_number(spins):
    solid_angle_sum = np.zeros((L, L))

    s1 = spins
    s2 = np.roll(spins, shift=1, axis=1) # spin to the right of s1
    s3 = np.roll(spins, shift=-1, axis=0) # lower right spin if y is odd, else lower left spin
    s4 = np.roll(s3, shift=1, axis=1) # lower right right spin if y is odd, else lower right spin

    tri_up_num = np.zeros((L, L))
    tri_up_den = np.zeros((L, L))
    tri_down_num = np.zeros((L, L))
    tri_down_den = np.zeros((L, L))

    for y in range(L):
        if not y%2: # if odd, the up triangle is s2->s3->s4 and the down triangle is s2->s1->s3
            tri_up_num[y, :] = np.sum(s2[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            tri_up_den[y, :] = (1+np.sum(s2[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s2[y, :], axis=-1))
            tri_down_num[y, :] = np.sum(s2[y, :]*np.cross(s1[y, :], s3[y, :]), axis=-1)
            tri_down_den[y, :] = (1+np.sum(s2[y, :]*s1[y, :], axis=-1)+np.sum(s1[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s2[y, :], axis=-1))
        else: # if even, the up triangle is s1->s3->s4 and the down triangle is s2->s1->s4
            tri_up_num[y, :] = np.sum(s1[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            tri_up_den[y, :] = (1+np.sum(s1[y, :]*s3[y, :], axis=-1)+np.sum(s3[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s1[y, :], axis=-1))
            tri_down_num[y, :] = np.sum(s2[y, :]*np.cross(s1[y, :], s4[y, :]), axis=-1)
            tri_down_den[y, :] = (1+np.sum(s2[y, :]*s1[y, :], axis=-1)+np.sum(s1[y, :]*s4[y, :], axis=-1)+np.sum(s4[y, :]*s2[y, :], axis=-1))
    
    solid_angle_sum = np.sum(abs(2*np.arctan2(tri_up_num, tri_up_den))+abs(2*np.arctan2(tri_down_num, tri_down_den)))
    skx = solid_angle_sum/(4*np.pi)

    return round(skx)

def chirality(spins):
    chi = np.zeros((L, L))

    s1 = spins
    s2 = np.roll(spins, shift=1, axis=1) 
    s3 = np.roll(spins, shift=-1, axis=0) 
    s4 = np.roll(s3, shift=1, axis=1) 

    for y in range(L):
        if not y%2: 
            chi_up = np.sum(s2[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            chi_down = np.sum(s2[y, :]*np.cross(s1[y, :], s3[y, :]), axis=-1)
        else: 
            chi_up = np.sum(s1[y, :]*np.cross(s3[y, :], s4[y, :]), axis=-1)
            chi_down = np.sum(s2[y, :]*np.cross(s1[y, :], s4[y, :]), axis=-1)
        
        chi[y, :] = (chi_up + chi_down) / 2

    return chi

def custom_cmap():
    inferno = plt.get_cmap('inferno')
    new_colors = inferno(np.linspace(0.2, 1, 256))
    new_cmap = mcolors.LinearSegmentedColormap.from_list("purple_inferno", new_colors)
    return new_cmap

def spin_struc(spins):
    mz = spins[:, :, 2]
    mz_fluctuation = mz - np.mean(mz) # removing the peak at q=0
    fq = np.fft.fftshift(np.fft.fft2(mz_fluctuation)) # fourier transform
    Sq = np.abs(fq)**2/(L**2) # spin structure factor
    Sq_smooth = ndi.gaussian_filter(Sq, sigma=2) # smoothened sq (a weighted average, where closer sites have higher weights)
    Sq_smooth = Sq_smooth/np.max(Sq_smooth)
    _, num_peaks = ndi.label(Sq_smooth > 0.7) # takes the number of peaks (any site with intensity higher than 0.7 is considered as a peak; adjacent peaks are not allowed)
    return num_peaks, Sq_smooth

def heun(spins, Ja, H, A, t, T0, intensity, seed):
    def effective_field(spins, Ja, H, A, t, T0, intensity, seed):
        H_eff = np.zeros_like(spins)

        # similar notation with total energy calculation, below are ths spins for J3
        right = np.roll(spins, shift=1, axis=1)
        left = np.roll(spins, shift=-1, axis=1)
        up = np.roll(spins, shift=1, axis=0)
        down = np.roll(spins, shift=-1, axis=0)

        H_eff += J1 * (left + right + down + up)

        # below are the spins used for J3
        right2 = np.roll(spins, shift=2, axis=1)
        left2 = np.roll(spins, shift=-2, axis=1)
        up_left2 = np.roll(left, shift=2, axis=0)
        down_left2 = np.roll(left, shift=-2, axis=0)
        up_right2 = np.roll(right, shift=2, axis=0)
        down_right2 = np.roll(right, shift=-2, axis=0)
        
        H_eff += J3 * (left2 + right2 + down_left2 + up_left2 + down_right2 + up_right2)

        angle2 = 2*np.pi/3
        angle3 = 4*np.pi/3

        for y in range(L):
            if not y % 2:
                up_adj = np.roll(spins, shift=(-1, 1), axis=(1, 0))
                down_adj = np.roll(spins, shift=(-1, -1), axis=(1, 0))

                delta2x = up_adj[y, :, 0]*np.cos(angle2) - up_adj[y, :, 1]*np.sin(angle2)
                delta2y = -1*up_adj[y, :, 1]*np.cos(angle2) - up_adj[y, :, 0]*np.sin(angle2)

                delta2x += down[y, :, 0]*np.cos(angle2) - down[y, :, 1]*np.sin(angle2)
                delta2y += -1*down[y, :, 1]*np.cos(angle2) - down[y, :, 0]*np.sin(angle2)

                delta3x = down_adj[y, :, 0]*np.cos(angle3) - down_adj[y, :, 1]*np.sin(angle3)
                delta3y = -1*down_adj[y, :, 1]*np.cos(angle3) - down_adj[y, :, 0]*np.sin(angle3)

                delta3x += up[y, :, 0]*np.cos(angle3) - up[y, :, 1]*np.sin(angle3)
                delta3y += -1*up[y, :, 1]*np.cos(angle3) - up[y, :, 0]*np.sin(angle3)
                
            else:
                up_adj = np.roll(spins, shift=(1, 1), axis=(1, 0))
                down_adj = np.roll(spins, shift=(1, -1), axis=(1, 0))

                delta2x = up[y, :, 0]*np.cos(angle2) - up[y, :, 1]*np.sin(angle2)
                delta2y = -1*up[y, :, 1]*np.cos(angle2) - up[y, :, 0]*np.sin(angle2)

                delta2x += down_adj[y, :, 0]*np.cos(angle2) - down_adj[y, :, 1]*np.sin(angle2)
                delta2y += -1*down_adj[y, :, 1]*np.cos(angle2) - down_adj[y, :, 0]*np.sin(angle2)

                delta3x = down[y, :, 0]*np.cos(angle3) - down[y, :, 1]*np.sin(angle3)
                delta3y = -1*down[y, :, 1]*np.cos(angle3) - down[y, :, 0]*np.sin(angle3)

                delta3x += up_adj[y, :, 0]*np.cos(angle3) - up_adj[y, :, 1]*np.sin(angle3)
                delta3y += -1*up_adj[y, :, 1]*np.cos(angle3) - up_adj[y, :, 0]*np.sin(angle3)
                
            H_eff[y, :] += J1 * (down_adj[y, :] + up_adj[y, :])

            delta1x = right[y, :, 0] + left[y, :, 0]
            delta1y = -1*right[y, :, 1] -left[y, :, 1]

            H_eff[y, :, 0] += 2*Ja*(delta1x + delta2x + delta3x)
            H_eff[y, :, 1] += 2*Ja*(delta1y + delta2y + delta3y)

        H_eff[:, :, 2] -= H
        H_eff[:, :, 2] -= 2*A*spins[:, :, 2]
        H_eff -= thermal_field(t, T0, intensity, seed)
        return H_eff

    H1 = effective_field(spins, Ja, H, A, t, T0, intensity, seed)
    dm1 = (gamma/(1+alpha**2)) * (
        np.cross(spins, H1) +
        alpha * np.cross(spins, np.cross(spins, H1))
    )
    spins_temp = normalize(spins + dt * dm1)

    H2 = effective_field(spins_temp, Ja, H, A, t, T0, intensity, seed)
    dm2 = (gamma/(1+alpha**2)) * (
        np.cross(spins_temp, H2) +
        alpha * np.cross(spins_temp, np.cross(spins_temp, H2))
    )
    spins_new = spins + (dt / 2) * (dm1 + dm2)

    return normalize(spins_new)

def simulate(**kwargs):

    seed = kwargs.get("seed", 0)
    T0 = kwargs.get("T0", 0)
    H = kwargs.get("H", 0)
    A = kwargs.get("A", 0)
    Ja = kwargs.get("Ja", 0)
    w = kwargs.get("w", 0)

    # triangular lattice initialization
    x_coords = []
    y_coords = []
    for i in range(N, -N, -1):
        for j in range(N, -N, -1):
            x = j * a + (a / 2 if i % 2 else 0)
            y = i * h
            x_coords.append(x)
            y_coords.append(y)


    save_path = "./sllg_data/"

    # intiial ferromagnetic state
    spins = np.zeros((L, L, 3))
    spins[:, :, 2] = 1

    # lg beam intensity
    intensity = get_intensity(x_coords, y_coords, w)

    for i in tqdm(range(steps), desc="Heun's Method"):
        t = 0 + i*dt
        spins = heun(spins, Ja, H, A, t, T0, intensity, seed)

        if i%(int(50/dt)) == 0: # save csv every 50 simulation time units
            t = round(t)
            save_csv(x_coords, y_coords, spins, save_path, t)

    # save the final spin configuration
    spin_config = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "m_x": spins[:, :, 0].flatten(),
        "m_y": spins[:, :, 1].flatten(),
        "m_z": spins[:, :, 2].flatten()
    })
    spin_config.to_csv(save_path+"spin.csv")

    jet = plt.get_cmap('jet')
    fig, ax = plt.subplots(figsize=(6, 6))
    spin_map = ax.scatter(x_coords, y_coords, c=spins[:, :, 2], cmap='jet', vmin=-1, vmax=1, s=20)
    cbar = plt.colorbar(spin_map, ax=ax, shrink=0.7, pad=0.04)
    cbar.ax.set_title(r"$S^z$")
    ax.set_facecolor(jet(1.0))
    plt.gca().set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-29, 30)
    ax.set_ylim(-29*np.sqrt(3)/2, 30*np.sqrt(3)/2)
    ax.quiver(x_coords, y_coords, spins[:, :, 0].flatten(), spins[:, :, 1].flatten(), color="black", pivot="middle", scale_units="xy", scale=0.5)
    plt.title("Final Spin Configuration")
    plt.savefig(save_path+"spins.pdf", bbox_inches='tight')
    plt.close()

    chi = chirality(spins)

    jet = plt.get_cmap('jet')
    fig, ax = plt.subplots(figsize=(6, 6))
    chi_map = ax.scatter(x_coords, y_coords, c=chi[:, :], cmap='jet', vmin=-1, vmax=1, s=20)
    cbar = plt.colorbar(chi_map, ax=ax, shrink=0.7, pad=0.04)
    cbar.ax.set_title(r"$\chi$")
    ax.set_facecolor(jet(1.0))
    plt.gca().set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-29, 30)
    ax.set_ylim(-29*np.sqrt(3)/2, 30*np.sqrt(3)/2)
    plt.title("Scalar Spin Chirality")
    plt.savefig(save_path+"chirality.pdf", bbox_inches='tight')
    plt.close()

    num_peaks, Sq = spin_struc(spins)
    cmap = custom_cmap()

    fig, ax = plt.subplots(figsize=(6, 6))
    imshow = ax.imshow(Sq, cmap=cmap, origin='lower')
    ax.tick_params(axis='both')
    cbar = plt.colorbar(imshow, ax=ax, shrink=0.82, pad=0.04)
    cbar.ax.set_title(r"$|S(q)|$")
    plt.xlabel("$q_x$")
    plt.ylabel("$q_y$")
    plt.title(f"Spin Structure Factor\nNumber of Detected Peaks = {num_peaks}")
    plt.savefig(save_path+"sq.pdf", bbox_inches='tight')

# parameters to be used in the simulation
params = {
    "seed": 0, # never use seed = 0 (thermal_field requires non-zero seed for simulated randomness)
    "Ja": 0,
    "T0": 0,
    "H": 0,
    "A": 0,
    "w": 0,
}

simulate(**params)