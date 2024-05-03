import numpy as np
import matplotlib.pyplot as plt
import time

    # chambre_nouvelle = (1*(V_haut-V_bas)/(2*matrice_r) + V_haut + V_bas + V_gauche + V_droite)/4 #LUI CALCULÉ MAIS PAS BEAU : Erreur de comprehension
    #LUI CALCULÉ MAIS PAS BEAU
    # chambre_nouvelle = (1*(V_gauche-V_droite)/(2*matrice_r) + V_haut + V_bas + V_gauche + V_droite)/4 #LUI PARFAIT MAIS PAS CA QUON A CALULÉ EN 1
    
    # chambre_nouvelle[1, 1:-1] = (2*chambre_vieille[1, :] + 2*chambre_vieille[0, :])/4 NONNNN
    # chambre_nouvelle[1, :] = (V_droite[1, :] + V_gauche[1, :] + 2*V_haut[0, :])/4 #VRM MIEUX MAIS PAS PARFAIT (Thomas)
    # chambre_nouvelle[1, :] = (V_droite[1, :] + V_gauche[1, :] + 4*V_haut[1, :])/6 #En 3d : moyenne des 6 cases autour
    #En 3d : Leane ÇA MARCHHHHHEEEEEEEEEEE
    #chambre_nouvelle[1, 1:-1] = (((1)/(4*1+3))*(2*chambre_vieille[1, :]) + ((2*1+3)/(4*1+3))*chambre_vieille[0, :])
    #chambre_nouvelle[1, :] = (((1)/(4*1+3))*(V_droite[1, :]) + ((1)/(4*1+3))*(V_gauche[1, :]) + ((2*1+3)/(4*1+3))*V_haut[0, :]) #VRM MIEUX MAIS PAS PARFAIT (Aurélie)
    #chambre_nouvelle[1, 1:-1] = (2*chambre_vieille[1, :] + 2*chambre_vieille[0, :])/4

    
# Gauche droite comme Xav
matrice_P[0:8, :] = [[0,1/6,0,0,0,0,0,0,1/6,0,0,0,0,0,4/6,0,0,0,0,0,0,0],
                     [1/6,0,4/6,0,0,0,0,0,0,1/6,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,2/8,0,3/8,0,0,0,0,0,0,0,0,0,0,1/8,0,2/8,0,0,0,0,0],
                     [0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0,0,0],
                     [0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0,0],
                     [0,0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0],
                     [0,0,0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0],
                     [0,0,0,0,0,0,1/8,0,0,0,0,0,0,2/8,0,1/8,0,0,0,0,0,2/8]]

matrice_P = np.zeros((22,22))

# Gauche droite pas comme
matrice_P[0:8, :] = [[0,1/6,0,0,0,0,0,0,1/6,0,0,0,0,0,4/6,0,0,0,0,0,0,0],
                     [1/6,0,4/6,0,0,0,0,0,0,1/6,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,2/8,0,3/8,0,0,0,0,0,0,0,0,0,0,1/8,0,2/8,0,0,0,0,0],
                     [0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0,0,0],
                     [0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0,0],
                     [0,0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0,0],
                     [0,0,0,0,0,1/8,0,3/8,0,0,0,0,2/8,0,0,0,0,0,0,0,2/8,0],
                     [0,0,0,0,0,0,1/8,0,0,0,0,0,0,2/8,0,1/8,0,0,0,0,0,2/8]]



# Haut bas   
matrice_P[0:8, :] = [[0,1/6,0,0,0,0,0,0,1/6,0,0,0,0,0,4/6,0,0,0,0,0,0,0],
                     [1/6,0,4/6,0,0,0,0,0,0,1/6,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,3/8,0,2/8,0,0,0,0,0,0,0,0,0,0,2/8,0,1/8,0,0,0,0,0],
                     [0,0,2/8,0,2/8,0,0,0,0,3/8,0,0,0,0,0,0,0,1/8,0,0,0,0],
                     [0,0,0,2/8,0,2/8,0,0,0,0,3/8,0,0,0,0,0,0,0,1/8,0,0,0],
                     [0,0,0,0,2/8,0,2/8,0,0,0,0,3/8,0,0,0,0,0,0,0,1/8,0,0],
                     [0,0,0,0,0,2/8,0,2/8,0,0,0,0,3/8,0,0,0,0,0,0,0,1/8,0],
                     [0,0,0,0,0,0,2/8,0,0,0,0,0,0,3/8,0,2/8,0,0,0,0,0,1/8]]



# Compter le temps d'éxécution
start = time.time()

"Initialisation de la chambre à ionisation" 
# dimensions de 12mm x 6mm
# Un noeud à chaque 1,5mm
n_noeuds_r = 5
n_noeuds_z = 9
# Initialisation avec une valeur de potentiel quelconque
matrice_noeuds = -150*np.ones((n_noeuds_r, n_noeuds_z))


def applique_CF_aplatie(chambre_plate, r, z):
    """
    Appliquer les conditions frontières à la chambre à ionisation aplatie ou non.

    Args:
    chambre_plate (numpy.ndarray): Chambre vectorisée ou non sur laquelle on veut appliquer les CF.
    r (int): nombre de noeuds sur le diamètre de la chambre à ionisation.
    z (int): nombre de noeuds sur la longueur de la chambre à ionisation.

    Returns:
    numpy.ndarray : Chambre vectorisée sur laquelle ont été appliquées les CF
    """
    chambre = chambre_plate.reshape(r, z)

    # Conditions frontières
    CF_cote_lignenoir = (-300*np.ones(z-(r//2)))
    CF_electr_centre = np.zeros(z-((r//2)+1))
    CF_base_bleu = np.zeros(r)
    CF_electr_centre = np.zeros(int(6*z/9))

    # Pour l'électrode du centre
    chambre[(r//2), int(3*z/9):] = CF_electr_centre

    # Pour les côtés de la chambre
    CF_cote_lignenoir = (-300*np.ones(z-(r//2)))
    chambre[0, (r//2):] = CF_cote_lignenoir
    chambre[r-1, (r//2):] = CF_cote_lignenoir

    # Pour la base bleue de la chambre
    chambre[:,z-1] = CF_base_bleu

    # Pour ce qui ne fait pas partie de la chambre, le potentiel est nul
    for ligne in range(1, (r//2)+1):
        chambre[ligne-1, :((r//2)+1) - ligne] = 0
        chambre[-1*ligne, :((r//2)+1) - ligne] = 0

    # Pour les côtés diagonaux
    for ligne in range(1, (r//2)+1):
        chambre[ligne-1, ((r//2)+1) - ligne] = -300
        chambre[-1*ligne, ((r//2)+1) - ligne] = -300
    chambre[(r//2), 0] = -300

    # Réaplatir la chambre
    return np.ravel(chambre)


# Appliquer les CF à la chambre
chambre_plate = applique_CF_aplatie(matrice_noeuds, n_noeuds_r, n_noeuds_z)


"Créer la matrice de probabilitées P"

# Initialiser la matrice de probabilité
P = np.zeros((n_noeuds_r*n_noeuds_z, n_noeuds_r*n_noeuds_z))

# La nième ligne de la matrice correspond aux probabilitées de transitionner vers les noeuds voisins du noeud n
# Fixer les probabilités de transitionner pour chaque noeud à la bonne colonne
for i in range(len(chambre_plate)):
    # Pour un noeud fixe, P[i, i] = 1 (il est certain que le noeud conserve le même potentiel)
    if chambre_plate[i] == 0 or chambre_plate[i] == -300:
        P[i,i] = 1
    # Pour un noeud libre, il est relié à 4 noeuds : la possibilité de transitionner sur ces derniers est de 25% chacun
    else:
        # Noeuds de gauche et de droite
        P[i, i-1] = 0.25
        P[i, i+1] = 0.25
        # Noeud du haut et du bas
        P[i, i-n_noeuds_z] = 0.25
        P[i, i+n_noeuds_z] = 0.25

# Montrer la matrice de probabilitées
# plt.imshow(P, cmap='viridis')
# plt.colorbar(label='Probabilité')
# plt.title('Matrice P')
# plt.show()


# Initialiser la liste des plus grandes différences de potentiel entre chaque itération
liste_diff_Markov = []

# Multiplier la matrice de probabilités de transition par les conditions initiales de la chambre ayant été vectorisée
def chaines_de_Markov(chambre):
    """
    Multiplier la matrice de probabilités de transition par la chambre vectorisée jusqu'à 
    atteindre le potentiel considéré comme final dans la chambre

    Args:
    chambre (numpy.ndarray): Chambre vectorisée ou non sur laquelle les CF ont été appliquées.

    Returns:
    numpy.ndarray : Chambre non vectorisée (forme originale) sur laquelle ont été appliquées les CF
    """
    matrice_precedante= np.ravel(chambre)

    n=0
    for n_iter in range(500):
        n+=1
        # Multiplier les matrices jusqu'à convergence
        matrice_suivante = np.dot(P, matrice_precedante)
        # Observer la différence de voltage maximale entre les deux itérations
        diff = np.max(np.abs(matrice_suivante - matrice_precedante))
        liste_diff_Markov.append(diff)

        matrice_precedante = matrice_suivante

    # Redonner la bonne forme à la chambre contenant le potentiel final
    chambre_pleine = matrice_precedante.reshape(n_noeuds_r, n_noeuds_z)

    # Temps d'éxécution de la méthode des chaînes de Markov absorbantes
    temps_Markov = time.time()
    print(f"Chaînes de Markov absorbantes :\nTemps d'exécution = {round(temps_Markov-start, 4)} secondes \nNombre d'itérations = {n}")

    return chambre_pleine


# La chambre a ionisation contenant le potentiel final
chambre_finale = chaines_de_Markov(chambre_plate)

# Trouver la largeur maximale parmi toutes les cellules pour un joli affichage du tableau
largeur_max = max(max(map(lambda x: len(str(round(x, 3))), row)) for row in chambre_finale)

# Imprimer le tableau du potentiel de chaque case avec des cellules de largeur uniforme
for ligne in chambre_finale:
    print("|", end="")
    for case in ligne:
        print(f" {str(round(case, 3)).ljust(largeur_max)} |", end="")
    print("\n" + "-"*(len(ligne)*(largeur_max + 3) + 1))


"Potentiel final dans la chambre à ionisation"
# plt.imshow(chambre_finale, cmap='viridis', origin='upper', extent=(12,0,-3,3))
# plt.colorbar(label='Potentiel [V]')
# # plt.title('Potentiel dans la chambre à ionisation')
# plt.xlabel('z [mm]')
# plt.ylabel('r [mm]')
# plt.show()


"Graphique de la différence du potentiel entre chaque itération en fonction du nombre d'itérations."
# n_iterations = np.linspace(0, len(liste_diff_Markov), len(liste_diff_Markov))
# plt.plot(n_iterations, liste_diff_Markov)
# plt.title("Différence de potentiel entre chaque itération selon le nombre d'itérations")
# plt.xlabel("Nombre d'itérations")
# plt.grid(True)
# plt.ylabel("Différence entre l'itération n et n+1 [V]")
# plt.yscale("log")
# plt.show()


"Methode a Oli"
# nouveau_pixel = int(chambre_complete.shape[1]/9)
# nouvelle_chambre_complete = np.empty((5,9))
# for x in range(9):
#     for y in range(5):
#         nouvelle_chambre_complete[y,x] = np.mean(chambre_complete[y*nouveau_pixel:(y+1)*nouveau_pixel, x*nouveau_pixel:(x+1)*nouveau_pixel])
# plt.imshow(nouvelle_chambre_complete)
# plt.show()