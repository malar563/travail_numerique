'''Électromagnétisme - Travail numérique : Un détecteur de radiation'''
import numpy as np
import matplotlib.pyplot as plt
import time


'''
Méthode de relaxation et de sur-relaxation
'''


"Initialisation de la chambre à ionisation"
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


# Chambre initiale
chambre_avec_CF = applique_CF(chambre_vide, r, z)



'''Méthode de relaxation'''


# Fonction permettant de trouver le potentiel pour l'itération suivante
# Il suffit de faire la moyenne des 4 cases autour
def decaler_matrices(chambre_vieille):
        
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

    return chambre_nouvelle_petite


# Fonction pour tester la différence entre deux itérations
def test_diff(chambre_vieille, chambre_nouvelle_petite):
    # Tester la différence entre les deux itérations
    diff = np.abs(applique_CF(chambre_nouvelle_petite, r, z)-chambre_vieille)

    pourcent = np.abs((diff/applique_CF(chambre_nouvelle_petite, r, z)))*100
    plus_grand_pourcent = np.nanmax(pourcent)

    diff_max = np.max(diff)

    # return plus_grand_pourcent
    return diff_max



def methode_de_relax(chambre_vieille, omega = 0):
    # Calculer le temps d'éxécution
    start = time.time()
    # Compter le nombre d'itérations
    n=0
    # Initialiser un critère pour permettre à la boucle de s'arrêter au moment approprié
    plus_grand_pourcent = 100

    while plus_grand_pourcent > 0.005 :

        # Éviter un crash en cas de problème
        if n > 5000:
            break
        n+=1    
        
        # Chambre de l'itération suivante
        chambre_nouvelle_petite = decaler_matrices(chambre_vieille)
        
        # Tester la différence entre deux itérations
        plus_grand_pourcent = test_diff(chambre_vieille, chambre_nouvelle_petite)

        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)

    # Donner la bonne forme à la chambre
    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0

    # Temps d'éxécution de la méthode de relaxation
    temps_methode_de_relax = time.time()
    print(f"Méthode de relaxation :\nTemps d'exécution = {round(temps_methode_de_relax-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine



def graphique(fonction, chambre_avec_CF, omega=0):

    # Créer la chambre complète (symétrie)
    fct = fonction(chambre_avec_CF, omega)
    chambre_complete = np.concatenate((np.flip(fct, 0)[:-1, :], fct))

    # Créer le graphique
    plt.imshow(chambre_complete, cmap='viridis', origin='upper', extent=(12,0,-3,3))
    plt.colorbar(label='Potentiel')
    plt.title('Potentiel')
    plt.xlabel('z')
    plt.ylabel('r')
    plt.show()


# Graphique produit avec la méthode de relaxation
graphique(methode_de_relax, chambre_avec_CF)




'''Méthode de la sur-relaxation'''

def methode_de_surrelax(chambre_vieille, omega):
    # Calculer le temps d'éxécution
    start = time.time()
    # Compter le nombre d'itérations
    n=0
    # Initialiser un critère pour permettre à la boucle de s'arrêter au moment approprié
    plus_grand_pourcent = 100


    # BIZZZZZ PAS LE MM POURCENTAGE QUEN HAUT
    while plus_grand_pourcent > 0.005 :

        # Éviter un crash en cas de problème
        if n > 5000:
            break
        n+=1
        
        # Chambre de l'itération suivante
        "ÇA CHANGE ICIIIIIIII PK ÇA MARCHE PAS"
        chambre_nouvelle_petite = ((1+omega)*decaler_matrices(chambre_vieille)) - (omega*chambre_vieille)
        
        # Tester la différence entre deux itérations
        plus_grand_pourcent = test_diff(chambre_vieille, chambre_nouvelle_petite)

        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)

    # Donner la bonne forme à la chambre
    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0
    #print(plus_grande_diff)

    # Temps d'éxécution de la méthode de relaxation
    temps_methode_de_surrelax = time.time()
    print(f"Méthode de sur-relaxation :\nTemps d'exécution = {round(temps_methode_de_surrelax-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine


# Graphique produit avec la méthode de sur-relaxation
graphique(methode_de_surrelax, chambre_avec_CF, 0.001)






