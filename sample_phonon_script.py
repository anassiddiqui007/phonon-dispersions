from nff.io import NeuralFF, AtomsBatch
from ase.phonons import Phonons
from ase.build.surface import mx2
import numpy as np
from ase.calculators.espresso import Espresso
import matplotlib.pyplot as plt 

#Supercell size and cell parameters used for getting phonon curves
N = 6 
a_SWPot = 3.1957418172700582
t_SWPot = 3.1938122319833404

#Define primitive and super cells
prim_opt=mx2(formula='MoS2',kind='2H',a=a_SWPot,thickness=t_SWPot,vacuum=8,size=(1,1,1)) 
sup_opt=mx2(formula='MoS2',kind='2H',a=a_SWPot,thickness=t_SWPot,vacuum=8,size=(N,N,1))

#Perform PaiNN calculations
#props dictionary for AtomsBatch object of nff
props = {
    'lattice': prim_opt.cell.tolist(),
}

# Painn model/calculator to use (change folder to use different trained potential)
nff_ase = NeuralFF.from_file('painn_MoS2_new/', device=0) 

sup_opt.set_calculator(nff_ase)
sup_opt = AtomsBatch(sup_opt,props=props,directed=True,pbc=True,requires_large_offsets=True)

ph_painn = Phonons(prim_opt,nff_ase,supercell=(N, N, 1), delta=0.08, name='MoS2_phonon_painn')
#Remove Phonon json data
ph_painn.clean()  
#Run requires the supercell argument for PaiNN to use neighbor information
ph_painn.run(sup_opt)  
#Acoustic branches should converge to zero at k(gamma) = 0
ph_painn.read(acoustic=True)

#Read Espresso calculations (Don't use ph.run())
calc = Espresso()

ph_espr = Phonons(prim_opt,calc,supercell=(N, N, 1), delta=0.08, name='MoS2_phonon_espresso') 
ph_espr.read(acoustic=True)

#Get dispersion curves
path = prim_opt.cell.bandpath('GMKG', npoints=150) #Plot along gamma M K gamma path
bs_painn = ph_painn.get_band_structure(path)
bs_espr = ph_espr.get_band_structure(path) #Plot along same path as before

# Plot the curves and DOS:
fig = plt.figure(1, figsize=(7, 4))
ax = fig.add_axes([.12, .07, .67, .85])
dos_painn = ph_painn.get_dos(kpts=(60, 60, 1)).sample_grid(npts=1000, width=2e-4)
dos_espr = ph_espr.get_dos(kpts=(60, 60, 1)).sample_grid(npts=1000, width=2e-4)
cm1 = 8065.54011  #factor changing energies from ev to cm-1
bs_painn._energies *= cm1
bs_espr._energies *= cm1
#y-axis frequency scale
emin = -25; emax = 500
bs_painn.plot(ax=ax, emin=emin, emax=emax, colors='r', ylabel=r"$\mathrm{\omega}$ (cm$^{-1}$)",label='PaiNN')
bs_espr.plot(ax=ax, emin=emin, emax=emax, colors='g', ylabel=r"$\mathrm{\omega}$ (cm$^{-1}$)",label='Espresso',linestyle='--')

ax.set_title('Phonon dispersion curves for MoS2')

dosax = fig.add_axes([.8, .07, .17, .85])
dosax.fill_between(dos_painn.get_weights(), dos_painn.get_energies()*cm1, y2=0, color='white',
                   edgecolor='r', lw=1)

dosax.fill_between(dos_espr.get_weights(), dos_espr.get_energies()*cm1, y2=0, color='white',
                   edgecolor='g', lw=1,linestyle='--')

dosax.set_ylim(emin, emax)
dosax.set_yticks([])
dosax.set_xticks([])
dosax.set_xlabel("DOS", fontsize=18)

#fig.savefig('MoS2_PaiNN_vs_Espresso.png')

plt.show()

#Get error metrics
freq_diff = abs(bs_painn.energies.flatten()-bs_espr.energies.flatten())
print((np.mean(freq_diff)),np.std(freq_diff))
