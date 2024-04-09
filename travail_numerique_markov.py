'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt
import time



'''Chaînes de Markov absorbantes'''

noeuds_r = 5
noeuds_z = 9
matrice_noeuds = -150*np.ones((noeuds_r, noeuds_z))


def applique_CF(chambre, r, z):

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
    
    
    # for case in range(r):
        # chambre[case, case] = -300

    return chambre

print(applique_CF(matrice_noeuds, noeuds_r, noeuds_z))