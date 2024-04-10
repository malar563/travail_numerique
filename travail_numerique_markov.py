'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt
import time



'''Chaînes de Markov absorbantes'''

# Initialiser la chambre avec les dimensions spécifiées
n_noeuds_r = 5
n_noeuds_z = 9
matrice_noeuds = -150*np.ones((n_noeuds_r, n_noeuds_z))


# Appliquer les CF à la chambre 
# La matrice en entrée est aplatie ou non
# On retourne une matrice aplatie (vectorisée)
def applique_CF_aplatie(chambre_plate, r, z):

    chambre = chambre_plate.reshape(r, z)

    # Conditions frontières
    CF_cote_lignenoir = (-300*np.ones(z-(r//2)))
    CF_electr_centre = np.zeros(z-((r//2)+1))
    CF_base_bleu = np.zeros(r)
    CF_electr_centre = np.zeros(int(5*z/9))

    # Pour l'électrode du centre
    chambre[(r//2), int(4*z/9):] = CF_electr_centre

    # Pour les côtés de la chambre
    CF_cote_lignenoir = (-300*np.ones(z-(r//2)))
    chambre[0, (r//2):] = CF_cote_lignenoir
    chambre[r-1, (r//2):] = CF_cote_lignenoir

    # Pour la base de la chambre
    chambre[:,z-1] = CF_base_bleu
    # Pour l'extérieur de la chambre
    for ligne in range(1, (r//2)+1):
        chambre[ligne-1, :((r//2)+1) - ligne] = 0
        chambre[-1*ligne, :((r//2)+1) - ligne] = 0
    # Pour les côtés diagonaux
    for ligne in range(1, (r//2)+1):
        chambre[ligne-1, ((r//2)+1) - ligne] = -300
        chambre[-1*ligne, ((r//2)+1) - ligne] = -300
    chambre[(r//2), 0] = -300

    chambre_vide_aplatie = np.ravel(chambre)

    return chambre_vide_aplatie

# Appliquer les conditions initiales à la chambre
chambre_plate = applique_CF_aplatie(matrice_noeuds, n_noeuds_r, n_noeuds_z)


# print(chambre_plate)




# Créer la matrice de probabilitées P
# La nième ligne de la matrice correspond aux probabilitées de transitionner vers les noeuds voisins du noeud n
P = np.zeros((n_noeuds_r*n_noeuds_z, n_noeuds_r*n_noeuds_z))

for i in range(len(chambre_plate)):
    # Pour un noeud fixe, P[i, i] = 1 (il est certain que le noeud conserve le même potentiel)
    if chambre_plate[i] == 0 or chambre_plate[i] == -300:
        P[i,i] = 1
    # Pour un noeud libre, il est relié à 4 noeuds : la possibilité de transitionner sur ces derniers est de 25% chacun
    else:
        P[i, i-1] = 0.25
        P[i, i+1] = 0.25
        P[i, i-n_noeuds_z] = 0.25
        P[i, i+n_noeuds_z] = 0.25

# Montrer la matrice de probabilitées
plt.imshow(P, cmap='viridis')
plt.colorbar(label='Probabilité')
plt.title('Probabilité')
plt.show()



# Multiplier la matrice de probabilités de transition par les conditions initiales de la chambre ayant été vectorisée
def chaines_de_Markov(chambre):
    matrice_precedante= chambre.flatten()
    for fois in range(500):
        matrice_suivante = np.dot(P, matrice_precedante)
        matrice_precedante = matrice_suivante
    chambre_pleine = matrice_precedante.reshape(n_noeuds_r, n_noeuds_z)
    return chambre_pleine

# La chambre a ionisation contenant le potentiel final
chambre_finale = chaines_de_Markov(chambre_plate)


# Créer le graphique
plt.imshow(chambre_finale, cmap='viridis', origin='upper', extent=(12,0,-3,3))
plt.colorbar(label='Potentiel')
plt.title('Potentiel')
plt.xlabel('z')
plt.ylabel('r')
plt.show()





