import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

fs = 14
cbarfs = 12

phasediagram = np.load("./data/real_space_phase_diagram.npy")

plt.imshow(phasediagram.transpose(), origin='lower', aspect='auto',
           extent=(-2, 2, 0, 2*np.pi), interpolation='bicubic', cmap='RdBu')

cbar = plt.colorbar()
cbar.set_label(label=r'$\mathrm{deg}~\hat{\mathbf{n}}$', size=fs)

eps = 1e-10

n_m = 300
for delta in [-4*np.pi,-2*np.pi, 0, 2*np.pi,4*np.pi]:
    ms = np.linspace(-np.sqrt(3)+eps, np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( 3*(np.pi + np.arccos(m / np.sqrt(3)))) +delta)
    plt.plot(ms,phase,'-', color='black')


    ms = np.linspace(-np.sqrt(3)+eps, np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( 3*(np.pi - np.arccos(m / np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'-', color='black')

    ms = np.linspace(-1/np.sqrt(3)+eps, 1/np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( (2*np.pi - np.arccos(m * np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'-', color='black')

    ms = np.linspace(-1/np.sqrt(3)+eps, 1/np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( (4*np.pi + np.arccos(m * np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'-', color='black')


# -- deg = -1

label_fs = 11
plt.text(0.3, np.pi - 0.1, r'$\mathrm{deg}\hat{\ \mathbf{n}}=-1$',{'fontsize': label_fs})

# -- deg = 0

plt.text(-2+0.15, np.pi - 0.1, r'$\mathrm{deg}\hat{\ \mathbf{n}}=0$',{'fontsize': label_fs})
plt.text(1.27, 2*np.pi-0.5, r'$\mathrm{deg}\hat{\ \mathbf{n}}=0$',{'fontsize': label_fs})
plt.text(1.27, 0.3, r'$\mathrm{deg}\hat{\ \mathbf{n}}=0$',{'fontsize': label_fs})

# -- deg = 1

plt.text(-1.0, 2*np.pi-0.5 , r'$\mathrm{deg}\hat{\ \mathbf{n}}=1$',{'fontsize': label_fs})
plt.text(-1.0, 0.5-0.2 , r'$\mathrm{deg}\hat{\ \mathbf{n}}=1$',{'fontsize': label_fs})

# -- deg = -2

plt.text(0.5, np.pi/2- 0.15, r'$\mathrm{deg}\hat{\ \mathbf{n}}=-2$',{'fontsize': label_fs})
plt.arrow(0.95, (np.pi/2- 0.1)-0.2, -0.25,-1.1 , color='lightgray',zorder=20)

plt.text(0.5,  2*np.pi-(np.pi/2)-0.02, r'$\mathrm{deg}\hat{\ \mathbf{n}}=-2$',{'fontsize': label_fs})
plt.arrow(0.95,  2*np.pi-(np.pi/2- 0.1)+0.2, -0.25,1.1 , color='lightgray',zorder=20)

# -- deg = 2

plt.text(-1.15, np.pi-1.1, r'$\mathrm{deg}\hat{\ \mathbf{n}}=2$',{'fontsize': label_fs})

plt.arrow(-0.685, np.pi-1 + 0.2, 0,0.8 , color='lightgray',zorder=20)


xticks = [-np.sqrt(3),-np.sqrt(3)/2,-1/np.sqrt(3),0,1/np.sqrt(3), np.sqrt(3)/2, np.sqrt(3)]
xticks_labels = [r'$-\sqrt{3}$',r'$-\sqrt{3}/2$        ',r'    $-1/\sqrt{3}$',r'$0$',r'$1/\sqrt{3}$     ',r'   $\sqrt{3}/2$', r'$\sqrt{3}$']
plt.xticks(xticks, xticks_labels)

yticks = [0, np.pi, 2*np.pi]
yticks_labels = [r'$0$',r'$\pi$',r'$2\pi$']
plt.yticks(yticks, yticks_labels)

ax = plt.gca()
ax.set_xlabel(r"$m$", fontsize=fs)
ax.set_ylabel(r"$\varphi$", fontsize=fs)

ax.set_ylim((0,2*np.pi))

plt.tight_layout()
plt.savefig("./real_space_phase_diagram.png", dpi=300)
plt.clf()
