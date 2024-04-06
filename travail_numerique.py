'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt


# Initialiser la chambre à ionisation : 
# dimensions de 12mm x 3mm pour une moitié (symétrie en ϕ)
# Un pas de 0,1mm
chambre_vide = np.zeros((30, 120))

def applique_CF(chambre):
    # Conditions frontières
    CF_electr_centre = np.zeros(75)
    CF_base_bleu = np.zeros(30)
    CF_cote_lignenoir = -300*np.ones(90)

    chambre[0,45:] = CF_electr_centre
    chambre[:,119] = CF_base_bleu
    chambre[29,30:] = CF_cote_lignenoir
    for case in range(30):
        chambre[case, case] = -300
    return chambre

chambre_avec_CF = applique_CF(chambre_vide)
print(chambre_avec_CF)



'''Méthode de relaxation'''


def methode_de_relax(nombre_iteration, chambre_vieille):
    print(chambre_vieille)

    for iter in range(nombre_iteration):

        # Matrice décalée vers le haut
        V_haut = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_haut[0:-2, 1:-1] = chambre_vieille

        # Matrice décalée vers le bas
        V_bas = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_bas[2:, 1:-1] = chambre_vieille

        # Matrice décalée vers la gauche
        V_gauche = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_gauche[1:-1, 0:-2] = chambre_vieille

        # Matrice décalée vers la gauche
        V_droite = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_droite[1:-1, 2:] = chambre_vieille

        # Moyenne des cases autour
        chambre_nouvelle = (V_haut + V_bas + V_gauche + V_droite)/4
        chambre_nouvelle_petite = chambre_nouvelle[1:-1, 1:-1]
        
        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite)
    
    chambre_pleine = chambre_vieille

    return chambre_pleine

 #BIZZZZZ ÇA ME DONNE LE REFLET COMME DANS UN MIROIR DES CF QUE JIMPOSE


# Créer le graphique
plt.imshow(methode_de_relax(100000, chambre_avec_CF), cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Potentiel')
plt.title('Potentiel')
plt.xlabel('z')
plt.ylabel('r')
plt.show()
