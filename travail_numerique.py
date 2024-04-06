'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt
import time


start = time.time()

# Initialiser la chambre à ionisation : 
# dimensions de 12mm x 3mm pour une moitié (symétrie en ϕ)
# Un pas de 0,1mm
r = 30
z = 120
chambre_vide = np.zeros((r, z))

def applique_CF(chambre, r, z):
    # Conditions frontières
    CF_electr_centre = np.zeros(int(5*z/8))
    CF_base_bleu = np.zeros(r)
    CF_cote_lignenoir = -300*np.ones(int(3*z/4))

    chambre[0,int(3*z/8):] = CF_electr_centre
    chambre[:,z-1] = CF_base_bleu
    chambre[r-1,r:] = CF_cote_lignenoir
    for case in range(r):
        chambre[case, case] = -300
    return chambre

chambre_avec_CF = applique_CF(chambre_vide, r, z)
# print(chambre_avec_CF)



'''Méthode de relaxation'''


def methode_de_relax(chambre_vieille):
    # print(chambre_vieille)

    n=0
    plus_grand_pourcent = 100

    while plus_grand_pourcent > 0.001 :

        # Éviter un crash en cas de problème
        if n > 5000:
            break

        n+=1
        
        # Matrice décalée vers le haut
        V_haut = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_haut[0:-2, 1:-1] = chambre_vieille

        # Matrice décalée vers le bas
        V_bas = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_bas[2:, 1:-1] = chambre_vieille
        V_bas[1, 1:-1] = chambre_vieille[1, :]

        # Matrice décalée vers la gauche
        V_gauche = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_gauche[1:-1, 0:-2] = chambre_vieille

        # Matrice décalée vers la gauche
        V_droite = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
        V_droite[1:-1, 2:] = chambre_vieille

        # Moyenne des cases autour
        chambre_nouvelle = (V_haut + V_bas + V_gauche + V_droite)/4
        chambre_nouvelle_petite = chambre_nouvelle[1:-1, 1:-1]
        
        # Tester la différence entre les deux itérations
        diff = np.abs(applique_CF(chambre_nouvelle_petite, r, z)-chambre_vieille)
        # plus_grande_diff = np.max(diff)

        pourcent = np.abs((diff/applique_CF(chambre_nouvelle_petite, r, z)))*100
        plus_grand_pourcent = np.nanmax(pourcent)
        # print(plus_grand_pourcent)

        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)


    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0
    #print(plus_grande_diff)
    print(n)

    return chambre_pleine

chambre_complete = np.concatenate((np.flip(methode_de_relax(chambre_avec_CF), 0)[:-1, :], methode_de_relax(chambre_avec_CF)))

temps_methode_de_relax = time.time()
print(f"Le temps d'éxécution pour la méthode de relaxation de {round(temps_methode_de_relax-start, 4)} secondes")

# Créer le graphique
plt.imshow(chambre_complete, cmap='viridis', origin='upper', extent=(12,0,-3,3))
plt.colorbar(label='Potentiel')
plt.title('Potentiel')
plt.xlabel('z')
plt.ylabel('r')
plt.show()




'''Méthode de la sur-relaxation'''


