'''
Électromagnétisme - Travail numérique : Un détecteur de radiation
'''
'''
Chaînes de Markov absorbantes
'''
'''
Léane Simard, Marc-Antoine Pelletier et Marylise Larouche
'''
import numpy as np
import matplotlib.pyplot as plt
import time


# Initialiser la matrice de probabilités vides
matrice_P = np.zeros((22,22))

# Probabilité de transistion des noeuds fixes
for i in range(8, 22):
    matrice_P[i, i] = 1
   
# Probabilité de transition des noeuds libres (Analyse du dessin de la question 3a)
matrice_P[0:8, :] = [[0,1/6,0,0,0,0,0,0,1/6,0,0,0,0,0,4/6,0,0,0,0,0,0,0],
                     [1/6,0,4/6,0,0,0,0,0,0,1/6,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,1/8,0,2/8,0,0,0,0,0,0,0,0,0,0,2/8,0,3/8,0,0,0,0,0],
                     [0,0,2/8,0,2/8,0,0,0,0,1/8,0,0,0,0,0,0,0,3/8,0,0,0,0],
                     [0,0,0,2/8,0,2/8,0,0,0,0,1/8,0,0,0,0,0,0,0,3/8,0,0,0],
                     [0,0,0,0,2/8,0,2/8,0,0,0,0,1/8,0,0,0,0,0,0,0,3/8,0,0],
                     [0,0,0,0,0,2/8,0,2/8,0,0,0,0,1/8,0,0,0,0,0,0,0,3/8,0],
                     [0,0,0,0,0,0,2/8,0,0,0,0,0,0,1/8,0,2/8,0,0,0,0,0,3/8]]

# Montrer la matrice de probabilitées
from matplotlib.ticker import FormatStrFormatter
ax1 = plt.subplot(111)
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax1.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
plt.imshow(matrice_P, cmap='viridis')
plt.colorbar(label='Probabilité')
plt.title('Matrice P')
plt.show()

start = time.time()

# Découper dans la matrice P la matrice Q des transistions vers les noeuds transitoires 
matrice_Q = matrice_P[0:8,0:8]

# Créer la matrice fondamentale N
matrice_N = np.linalg.inv(np.eye(8,8) - matrice_Q)

# Découper dans la matrice P la matrice Q des transistions vers les noeuds absorbants 
matrice_R = matrice_P[0:8,8:22]

# Créer la matrice B (Probabilité du noeuds i d'être absorbé par le noeud j)
matrice_B = np.dot(matrice_N, matrice_R)

# Déterminer le potentiel des noeuds transitoires (libres)
V_noeuds_fixes = np.array([-300,0,0,0,0,0,-300,0,-300,-300,-300,-300,-300,-300])
V_noeuds_libres = np.dot(matrice_B,V_noeuds_fixes)
print(V_noeuds_libres)
print(f"Chaînes de Markov absorbantes :\nTemps d'exécution = {round(time.time()-start, 5)} secondes")


# Construire la chambre contenant le potentiel final
chambre_finale = np.zeros((3,9))
for i in range(3):
    chambre_finale[i,i] = -300

chambre_finale[2, 3:-1] = V_noeuds_fixes[8:-1]
chambre_finale[0, 1:3] = V_noeuds_libres[0:2]
chambre_finale[1, 2:-1] = V_noeuds_libres[2:]

chambre_complete = np.concatenate((np.flip(chambre_finale,0)[:-1, :], chambre_finale))
plt.imshow(chambre_complete, cmap='viridis', origin='upper', extent=(12,0,-3,3))
plt.colorbar(label='Potentiel [V]')
# plt.title('Potentiel dans la chambre à ionisation')
plt.xlabel('z [mm]')
plt.ylabel('r [mm]')
plt.show()