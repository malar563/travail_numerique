'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt
import time



'''Chaînes de Markov absorbantes'''

noeuds_r = 5
noeuds_z = 9
matrice_noeuds = -150*np.ones((noeuds_r, noeuds_z))

'''
OK
'''
def applique_CF_aplatie(chambre_plate, r, z):

    chambre = chambre_plate.reshape(5, 9)

    # Conditions frontières
    CF_cote_lignenoir = (-300*np.ones(z-2))
    CF_electr_centre = np.zeros(z-3)
    CF_base_bleu = np.zeros(r)
    
    chambre[0,0:2] = np.zeros(2)
    chambre[r-1,0:2] = np.zeros(2)
    chambre[0,2:] = CF_cote_lignenoir
    chambre[1,0:2] = np.array([0, -300])
    chambre[r-2,0:2] = np.array([0, -300])
    chambre[2,0] = -300
    chambre[2,3:] = CF_electr_centre
    chambre[r-1,2:] = CF_cote_lignenoir
    chambre[:,z-1] = CF_base_bleu

    chambre_vide_aplatie = np.ravel(chambre)

    return chambre_vide_aplatie



chambre = applique_CF_aplatie(matrice_noeuds, noeuds_r, noeuds_z)
Vci = applique_CF_aplatie(matrice_noeuds, noeuds_r, noeuds_z)
print(chambre)

'''
OK
'''
x = 5
y = 9
P = np.zeros((x*y, x*y))
'''
OK
'''
# Créer la matrice de probabilitées P
# La nième ligne de la matrice correspond aux possibilitées de transitionner vers les noeuds voisins du noeud n


# for i in range(12, 18):
#     P[i, i-1] = 0.25
#     P[i, i+1] = 0.25
#     P[i, i-9] = 0.25
#     P[i, i+9] = 0.25
# for i in range(20, 22):
#     P[i, i-1] = 0.25
#     P[i, i+1] = 0.25
#     P[i, i-9] = 0.25
#     P[i, i+9] = 0.25
# for i in range(30, 35):
#     P[i, i-1] = 0.25
#     P[i, i+1] = 0.25
#     P[i, i-9] = 0.25
#     P[i, i+9] = 0.25


# for i in range(12):
#     P[i,i] = 1
# for i in range(18, 20):
#     P[i,i] = 1
# for i in range(22, 30):
#     P[i,i] = 1
# for i in range(36, 45):
#     P[i,i] = 1

for i in range(len(Vci)):
    if Vci[i] == 0 or Vci[i] == -300: #ajoute une valeur de 1 à tous les noeuds fixes 
        P[i,i] = 1
    else: #ajoute une valeur de 25% à tous les noeuds adjacents aux noeuds variables
        P[i, i-1] = 0.25
        P[i, i+1] = 0.25
        # if i - x >= 0:
        P[i, i-9] = 0.25
        P[i, i+9] = 0.25
        # else: #si le noeud est situé à l'extrémité supérieure, la valeur associée au noeud supérieur est ajouté au noeud inférieur
        #     P[i,i+x] = 0.5



def graphique(matrice):

    # Créer le graphique
    plt.imshow(matrice, cmap='viridis')
    plt.colorbar(label='Potentiel')
    plt.title('Potentiel')
    plt.xlabel('z')
    plt.ylabel('r')
    plt.show()

graphique(P)



def chaines_de_Markov(chambre):
    matrice_precedante= chambre.flatten()
    for fois in range(20):
        matrice_suivante = np.dot(P, matrice_precedante)
        matrice_precedante = matrice_suivante
    chambre_pleine = matrice_precedante.reshape(x, y)
    return chambre_pleine

chambre_finale = chaines_de_Markov(Vci)
graphique(chambre_finale)



