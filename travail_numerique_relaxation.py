'''
Électromagnétisme - Travail numérique : Un détecteur de radiation
'''
'''
Méthode de relaxation et de sur-relaxation
'''
import numpy as np
import matplotlib.pyplot as plt
import time


"Initialisation de la chambre à ionisation" 
# dimensions de 12mm x 3mm pour une moitié (symétrie en ϕ)
# Un pas de 0,1mm
r = 30
z = 120
chambre_vide = np.zeros((r, z))


def applique_CF(chambre, r, z):
    """
    Appliquer les conditions frontières à la chambre à ionisation.

    Args:
    chambre (numpy.ndarray): Chambre sur laquelle on veut appliquer les CF.
    r (int): Rayon de la chambre à ionisation.
    z (int): Longueur de la chambre à ionisation.

    Returns:
    chambre (numpy.ndarray) : Chambre sur laquelle ont été appliquées les CF
    """
    # Conditions frontières
    CF_electr_centre = np.zeros(int(5*z/8))
    CF_base_bleu = np.zeros(r)
    CF_cote_lignenoir = -300*np.ones(int(3*z/4))

    # Électrode du centre
    chambre[0,int(3*z/8):] = CF_electr_centre
    # Côtés noirs
    chambre[r-1,r:] = CF_cote_lignenoir
    for case in range(r):
        chambre[case, case] = -300
    # Base bleue
    chambre[:,z-1] = CF_base_bleu
    return chambre

# Chambre initiale
chambre_avec_CF = applique_CF(chambre_vide, r, z)


'''
Méthode de relaxation
'''

def decaler_matrices(chambre_vieille):
    """
    Trouver le potentiel dans chaque case de la matrice pour l'itération suivante.
    Le potentiel de la case à l'itération suivante correspond à la moyenne des 
    4 cases autour de la case d'intérêt.

    Args:
    chambre (numpy.ndarray): Chambre pour laquelle on cherche le potentiel à l'itération suivante.

    Returns:
    chambre_nouvelle_petite (numpy.ndarray) : Chambre contenant le potentiel de l'itération suivante.
    """
        
    # Matrice décalée vers le haut
    V_haut = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_haut[0:-2, 1:-1] = chambre_vieille

    # Matrice décalée vers le bas
    V_bas = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_bas[2:, 1:-1] = chambre_vieille
    # Puisque la chambre est symétrique, il faut rajouter :
    V_bas[1, 1:-1] = chambre_vieille[1, :]

    # Matrice décalée vers la gauche
    V_gauche = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_gauche[1:-1, 0:-2] = chambre_vieille

    # Matrice décalée vers la gauche
    V_droite = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_droite[1:-1, 2:] = chambre_vieille

    # Le potentiel à l'itération suivante correspond à la moyenne des cases autour
    # Moyenne : addition des 4 matrices et division par 4
    chambre_nouvelle = (V_haut + V_bas + V_gauche + V_droite)/4
    # Restreindre la chambre à sa grandeur d'origine
    chambre_nouvelle_petite = chambre_nouvelle[1:-1, 1:-1]

    return chambre_nouvelle_petite



def test_diff(chambre_vieille, chambre_nouvelle):
    """
    Trouver la différence de voltage entre les itérations n 
    et n+1 dans la chambre.

    Args:
    chambre_vieille (numpy.ndarray): Chambre de l'itération n.
    chambre_nouvelle (numpy.ndarray): Chambre de l'itération n+1.

    Returns:
    diff_max (float) : Plus grand écart entre le potentiel de l'itération n et n+1.
    """
    diff = np.abs(applique_CF(chambre_nouvelle, r, z)-chambre_vieille)
    diff_max = np.max(diff)

    return diff_max


# Initialiser la liste des plus grandes différences de potentiel entre chaque itération
liste_diff_relax = []


def methode_de_relax(chambre_vieille, omega = 0):
    """
    Trouver le potentiel final dans la chambre à ionisation

    Args:
    chambre_vieille (numpy.ndarray): Chambre initiale où seulement les CF ont été appliquées.
    omega (float): coefficient entre 0 et 1 permettant de faire converger la solution plus vite.

    Returns:
    chambre_pleine (numpy.ndarray) : Chambre finale où le potentiel est considéré constant.
    """
    # Compter le temps d'éxécution
    start = time.time()
    # Compter le nombre d'itérations
    n=0
    
    for iter in range(1200):
        n+=1 
        # Chambre de l'itération suivante
        chambre_nouvelle_petite = decaler_matrices(chambre_vieille)
        
        # Tester la différence entre deux itérations
        plus_grand_pourcent = test_diff(chambre_vieille, chambre_nouvelle_petite)
        liste_diff_relax.append(plus_grand_pourcent)

        # Réimposer les CF à la chambre
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)

    # Pour ce qui ne fait pas partie de la chambre, le potentiel est nul
    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0

    # Temps d'éxécution de la méthode de relaxation
    temps_methode_de_relax = time.time()
    print(f"Méthode de relaxation :\nTemps d'exécution = {round(temps_methode_de_relax-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine


def graphique(fonction, chambre_avec_CF, omega=0):
    """
    Créer la représentation du potentiel final dans la chambre à ionisation

    Args:
    fonction (func): fonction à appliquer sur la chambre pour déterminer le potentiel dans celle-ci.
    chambre_avec_CF (numpy.ndarray): Chambre initiale où seulement les CF ont été appliquées.
    omega (float): coefficient entre 0 et 1 permettant de faire converger la solution plus vite.

    Show:
    chambre_complete (numpy.ndarray) : Chambre finale où le potentiel est considéré constant.
    """

    # Créer la chambre complète (symétrie)
    fct = fonction(chambre_avec_CF, omega)
    chambre_complete = np.concatenate((np.flip(fct, 0)[:-1, :], fct))

    # Créer le graphique
    plt.imshow(chambre_complete, cmap='viridis', origin='upper', extent=(12,0,-3,3))
    plt.colorbar(label='Potentiel [V]')
    plt.title('Potentiel dans la chambre à ionisation')
    plt.xlabel('z [mm]')
    plt.ylabel('r [mm]')
    plt.show()


def graph_difference(liste_diff):
    """
    Créer un graphique de la différence du potentiel entre chaque itération en fonction du nombre d'itérations.

    Args:
    list_diff (list): Chambre initiale où seulement les CF ont été appliquées.

    Show:
    Différence du potentiel entre chaque itération en fonction du nombre d'itérations.
    """
    n_iterations = np.linspace(0, len(liste_diff), len(liste_diff))
    plt.plot(n_iterations, liste_diff)
    plt.title("Différence de potentiel entre chaque itération selon le nombre d'itérations")
    plt.xlabel("Nombre d'itérations")
    plt.grid(True)
    plt.ylabel("Différence entre l'itération n et n+1 [V]")
    plt.yscale("log")
    plt.show()


"Potentiel final dans la chambre à ionisation"
graphique(methode_de_relax, chambre_avec_CF)

"Différence du potentiel entre deux itérations selon le nombre d'itérations"
graph_difference(liste_diff_relax)









'''
Méthode de la sur-relaxation
'''

liste_diff_surrelax = []

def methode_de_surrelax(chambre_vieille, omega):
    # Calculer le temps d'éxécution
    start = time.time()
    # Compter le nombre d'itérations
    n=0
    
    for iter in range(2500):
        n+=1
        
        # Chambre de l'itération suivante
        "ÇA CHANGE ICIIIIIIII PK ÇA MARCHE PAS"
        chambre_nouvelle_petite = ((1+omega)*decaler_matrices(chambre_vieille)) - (omega*chambre_vieille)
        
        # Tester la différence entre deux itérations
        plus_grand_pourcent = test_diff(chambre_vieille, chambre_nouvelle_petite)
        liste_diff_surrelax.append(plus_grand_pourcent)

        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)

    # Donner la bonne forme à la chambre
    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0

    # Temps d'éxécution de la méthode de relaxation
    temps_methode_de_surrelax = time.time()
    print(f"Méthode de sur-relaxation :\nTemps d'exécution = {round(temps_methode_de_surrelax-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine


# Graphique produit avec la méthode de sur-relaxation
graphique(methode_de_surrelax, chambre_avec_CF, 0)
graph_difference(liste_diff_surrelax)






