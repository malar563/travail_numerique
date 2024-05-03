'''
Électromagnétisme - Travail numérique : Un détecteur de radiation
'''
'''
Méthode de sur-relaxation
'''
'''
Léane Simard, Marc-Antoine Pelletier et Marylise Larouche
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


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

    # Matrice décalée vers la gauche
    V_gauche = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_gauche[1:-1, 0:-2] = chambre_vieille

    # Matrice décalée vers la gauche
    V_droite = np.zeros((chambre_vieille.shape[0]+2, chambre_vieille.shape[1]+2))
    V_droite[1:-1, 2:] = chambre_vieille

    # Matrice contenant les valeurs de r
    matrice_r = np.ones(V_droite.shape)
    for iter in range(0, r+2):
        matrice_r[iter-1,:] = iter*matrice_r[iter-1,:]
 
    # Le potentiel à l'itération suivante est calculé selon l'expression trouvée à la question 1a)
    chambre_nouvelle = (1*(V_bas-V_haut)/(2*matrice_r) + V_haut + V_bas + V_gauche + V_droite)/4 

    # Pour r = 0, le potentiel à l'itération suivante est calculé selon l'expression trouvée à la question 1b)
    chambre_nouvelle[1, :] = (4*V_haut[1, :] + V_droite[1, :] + V_gauche[1, :])/6 
    
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
    fct = fonction(chambre_avec_CF, omega)[0]
    chambre_complete = np.concatenate((np.flip(fct, 0)[:-1, :], fct))

    # Créer le graphique
    plt.imshow(chambre_complete, cmap='viridis', origin='upper', extent=(12,0,-3,3))
    plt.colorbar(label='Potentiel [V]')
    # plt.title('Potentiel dans la chambre à ionisation')
    plt.xlabel('z [mm]')
    plt.ylabel('r [mm]')
    plt.show()

    return chambre_complete


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


# Initialiser la liste des plus grandes différences de potentiel entre chaque itération
liste_diff_surrelax = []


def methode_de_surrelax(chambre_vieille, omega):
    # Calculer le temps d'éxécution
    start = time.time()
    # Compter le nombre d'itérations
    n=0

    # Définir un critère d'arrêt
    plus_grand_diff = 300
    
    while plus_grand_diff > 0.01 and n<5000:
        n+=1
        
        # Chambre de l'itération suivante
        chambre_nouvelle_petite = ((1-omega)*decaler_matrices(chambre_vieille)) + ((omega)*chambre_vieille)
        
        # Tester la différence entre deux itérations
        plus_grand_diff = test_diff(chambre_vieille, chambre_nouvelle_petite)
        liste_diff_surrelax.append(plus_grand_diff)

        # Réimposer les CF
        chambre_vieille = applique_CF(chambre_nouvelle_petite, r, z)

    # Donner la bonne forme à la chambre
    chambre_pleine = chambre_vieille
    for case in range(r-1):
        chambre_pleine[case+1, :case+1] = 0

    # Temps d'éxécution de la méthode de relaxation
    temps_methode_de_surrelax = time.time()
    #print(f"Méthode de sur-relaxation :\nTemps d'exécution = {round(temps_methode_de_surrelax-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine, n, round(temps_methode_de_surrelax-start, 4)


# Test de convergence en fonction de omega
liste_omega = np.linspace(0,0.005,1) # Pour les test de convergence en fonction de omega, nous avions mis 1000 ici
liste_iterations = []
liste_temps = []
# On teste différentes valeurs de omega et on regarde le temps et le nombre d'itération nécessaire pour que la différence entre deux itérations soit de 0.01V
for i in tqdm(range(len(liste_omega))):
    chambre, nb_iter, temps = methode_de_surrelax(chambre_avec_CF, liste_omega[i])
    liste_iterations.append(nb_iter)
    liste_temps.append(temps)

# Générer les graphiques pour les tests de convergence
plt.plot(liste_omega, liste_iterations)
plt.xlabel(r"$\omega$", fontsize=15)
plt.ylabel("Nombre d'itérations", fontsize=15)
plt.show()
plt.plot(liste_omega, liste_temps)
plt.xlabel(r"$\omega$", fontsize=15)
plt.ylabel("Temps de convergence", fontsize=15)
plt.show()

# Graphique produit avec la méthode de sur-relaxation
liste_diff_surrelax = []
# Afficher la chambre finale lorsque la valeur de omega est de 0.0015
graphique(methode_de_surrelax, chambre_avec_CF, 0.0015)

# Afficher le graphique de la différence de potentiel entre une itération et la suivante
graph_difference(liste_diff_surrelax)