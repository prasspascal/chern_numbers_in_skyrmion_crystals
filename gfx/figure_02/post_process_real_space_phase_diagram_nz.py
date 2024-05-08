import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

fs = 14
cbarfs = 12

phasediagram = np.load("./data/real_space_phase_diagram_nz.npy")

plt.imshow(phasediagram.transpose(), origin='lower', aspect='auto',
           extent=(-2, 2, 0, 2*np.pi), interpolation='nearest', cmap='RdBu')

cbar = plt.colorbar()
cbar.set_label(label=r'$\langle{\hat{n}_z}\rangle$', size=fs)

eps = 1e-10

n_m = 300
for delta in [-4*np.pi,-2*np.pi, 0, 2*np.pi,4*np.pi]:
    ms = np.linspace(-np.sqrt(3)+eps, np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( 3*(np.pi + np.arccos(m / np.sqrt(3)))) +delta)
    plt.plot(ms,phase,'--', color='lightgray')


    ms = np.linspace(-np.sqrt(3)+eps, np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( 3*(np.pi - np.arccos(m / np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'--', color='lightgray')

    ms = np.linspace(-1/np.sqrt(3)+eps, 1/np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( (2*np.pi - np.arccos(m * np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'--', color='lightgray')

    ms = np.linspace(-1/np.sqrt(3)+eps, 1/np.sqrt(3)-eps,n_m)
    phase = []
    for m in ms:
        phase.append( ( (4*np.pi + np.arccos(m * np.sqrt(3))))  +delta)
    plt.plot(ms,phase,'--', color='lightgray')


plt.contour(phasediagram.transpose(), levels=[-2,0,2],origin='lower',
           extent=(-2, 2, 0, 2*np.pi), colors=['black', 'black', 'black'])

plt.text(-0.2, np.pi - 0.1, r'$\langle{\hat{n}_z}\rangle=0$',{'fontsize': 11})

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
plt.savefig("./real_space_phase_diagram_nz.png", dpi=100)
plt.clf()
