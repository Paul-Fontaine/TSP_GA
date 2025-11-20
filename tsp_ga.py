import time
from collections import defaultdict
from math import sqrt
from tsp_graph_init import Graph, Affichage, Route, Lieu
import random

# valeurs par défaut
TAILLE_POPULATION = 100
PROB_MUTATION = 0.1
NOMBRE_GENERATIONS = 500


class TSP_GA:
    def __init__(self, graph: Graph, affichage: Affichage | None = None,
                 taille_pop=TAILLE_POPULATION,
                 taille_pop_enfants=None,
                 prob_mutation=PROB_MUTATION,
                 nb_generations=NOMBRE_GENERATIONS):
        self.graph: Graph = graph
        self.affichage: Affichage | None = affichage
        self.taille_pop = int(taille_pop)
        self.taille_pop_enfants = int(taille_pop_enfants or int(self.taille_pop * 0.7))
        self.prob_mutation = float(prob_mutation)
        self.nb_generations = int(nb_generations)

        self.N = graph.N

        # Population initiale
        start_time = time.time()
        self.population: list[Route] = self._creer_pop_initiale(self.taille_pop)
        self.best_route: Route = min(self.population)
        end_time = time.time()
        print(f"Population initiale créée en {end_time - start_time:.2f} secondes.")
        print(f"Meilleure distance initiale : {self.best_route.distance:.2f}")

    def _new_route(self, ordre: list[int]) -> Route:
        """Crée une Route à partir d'un ordre et calcule sa distance."""
        r = Route(ordre)
        r.distance = self.graph.calcul_distance_route(r)
        return r

    def _route_plus_proche_voisin(self, depart: int) -> Route:
        """
        Construis une route par plus proche voisin en partant de 'depart'.
        """
        route = Route([depart])
        visites = {depart}

        while len(route) < self.N:
            dernier_lieu = route.ordre[-1]
            prochain_lieu = self.graph.plus_proche_voisin(dernier_lieu, visites)
            if prochain_lieu is None:
                break
            route.ajouter_lieu(prochain_lieu)
            visites.add(prochain_lieu)

        # retour au départ
        if route.ordre[-1] != depart:
            route.ajouter_lieu(depart)
        route.reordonner()
        return route

    def _2_opt(self, route: Route) -> Route:
        """Amélioration locale sur une route en échangeant deux arêtes.
           Complexité : O(N²) pire cas, mais converge rapidement, car ça s'arrête dès qu'une amélioration est trouvée."""
        ameliore = True
        meilleure = route
        meilleure.distance = self.graph.calcul_distance_route(meilleure)

        while ameliore:
            ameliore = False
            for i in range(1, len(meilleure) - 3):
                for j in range(i + 1, len(route) - 1):
                    candidate = Route(meilleure[:i] + meilleure[i:j][::-1] + meilleure[j:])
                    candidate.distance = self.graph.calcul_distance_route(candidate)
                    if candidate.distance < meilleure.distance:
                        meilleure = candidate
                        ameliore = True
        return meilleure

    def _farthest_insertion(self) -> Route:
        """Heuristique constructive : insère toujours la ville la plus éloignée.
           Complexité : O(N²)"""

        max_d = -1
        depart, prochain_lieu = 0, None
        for j in range(1, self.N):
            d = self.graph.distance_ij(depart, j)
            if d > max_d:
                max_d = d
                prochain_lieu = j

        ordre = [0, prochain_lieu, 0]
        candidates = set(range(self.N)) - {0, prochain_lieu}

        while candidates:
            # trouve le lieu le plus éloigné de tous les lieux déjà dans la sous-route
            lieu_loin = max(
                candidates,
                key=lambda l: min(self.graph.distance_ij(l, lieu) for lieu in ordre)
            )
            # Insère le lieu à la position optimale pour minimiser l'augmentation de distance
            best_pos = min(
                range(1, len(ordre)),
                key=lambda i: (
                    self.graph.distance_ij(ordre[i - 1], lieu_loin)
                    + self.graph.distance_ij(lieu_loin, ordre[i])
                    - self.graph.distance_ij(ordre[i - 1], ordre[i])
                )
            )
            ordre.insert(best_pos, lieu_loin)
            candidates.remove(lieu_loin)

        return Route(ordre)

    def construire_tour_grille(self):
        """Découpe le plan en une grille puis parcourt les cellules de la grille
           pour construire localement des sous-routes avec un plus proche voisin local.
           Complexité globale : O(N) (en pratique ~ O(N) car les cellules sont petites).
        """

        nb_cellules = int(sqrt(self.N/15))  # 15 lieux par cellule en moyenne

        # 1. Récupérer limites du plan
        xs = [l.x for l in self.graph.liste_lieux]
        ys = [l.y for l in self.graph.liste_lieux]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Taille d'une cellule (évite division par 0 si tous les points sont alignés)
        w = max_x - min_x
        h = max_y - min_y
        cell_w = (w / nb_cellules) if w > 0 else 1.0
        cell_h = (h / nb_cellules) if h > 0 else 1.0

        # 2. Créer une grille vide : grid[cx][cy] = liste d'indices de lieux
        grid = [[[] for _ in range(nb_cellules)] for _ in range(nb_cellules)]

        # 3. Placer chaque ville dans la bonne cellule → O(N)
        for i, lieu in enumerate(self.graph.liste_lieux):
            cx = min(int((lieu.x - min_x) / cell_w), nb_cellules - 1)
            cy = min(int((lieu.y - min_y) / cell_h), nb_cellules - 1)
            grid[cx][cy].append(i)

        # 4. Construire la route en parcourant la grille
        depart = random.choice(grid[0][0])
        ordre = [depart]
        lieux_utilises = {depart}

        # Ordre des lignes légèrement aléatoire pour varier les routes
        rows = list(range(nb_cellules))
        random.shuffle(rows)

        for cx in rows:
            # Pour chaque ligne, on décide un sens de parcours des colonnes
            cols = list(range(nb_cellules))
            if random.random() < 0.5:
                cols.reverse()

            for cy in cols:
                if not grid[cx][cy]:
                    continue

                # Points de cette cellule qui ne sont pas encore utilisés
                local_points = [i for i in grid[cx][cy] if i not in lieux_utilises]
                if not local_points:
                    continue

                # On construit une sous-route locale par plus proche voisin
                # en partant du dernier point global ajouté
                last = ordre[-1]

                while local_points:
                    # choisit le point le plus proche de "last"
                    nxt = min(
                        local_points,
                        key=lambda l: self.graph.distance_ij(last, l)
                    )
                    ordre.append(nxt)
                    lieux_utilises.add(nxt)
                    local_points.remove(nxt)
                    last = nxt

        ordre.append(0)
        route = self._new_route(ordre)
        route.reordonner()
        return route

    def _creer_pop_initiale(self, taille_pop: int) -> list[Route]:
        """Crée la population initiale en combinant les heuristiques plus proche voisin, farthest insertion
           et des routes aléatoires.
           Complexité : O(p*N²) avec p la taille de la population qui n'est pas générée aléatoirement."""
        if self.N > 10000:
            nb_ppv = 0
            nb_fi = 0
            nb_grille = 1
            nb_aleatoire = taille_pop - nb_grille
        elif self.N > 500:
            nb_ppv = 1
            nb_fi = 0
            nb_grille = 5
            nb_aleatoire = taille_pop - nb_grille
        elif self.N > 200:
            nb_ppv = 4
            nb_fi = 1
            nb_grille = int(0.4 * taille_pop)
            nb_aleatoire = taille_pop - nb_grille - nb_fi
        else:
            n_operations_max = 1e6
            pourcentage_aleatoire = 1 - n_operations_max / (taille_pop * self.N * self.N)
            pourcentage_aleatoire = max(0.5, min(pourcentage_aleatoire, 0.9999))
            print(f"  - Pourcentage de routes aléatoires ajusté à {pourcentage_aleatoire*100:.2f}% pour limiter p*N² à {n_operations_max:.0f}.")

            nb_aleatoire = int(taille_pop * pourcentage_aleatoire)
            nb_restant = taille_pop - nb_aleatoire - 1
            nb_fi = 1
            nb_grille = int(nb_restant / 2)
            nb_ppv = taille_pop - nb_aleatoire - nb_fi - nb_grille

        print(f"  - Création de la population initiale : {nb_ppv} PPV, {nb_fi} Farthest Insertion, {nb_grille} heuristique grille, {nb_aleatoire} aléatoires.")

        population = []
        seen = set()

        # 1) plus proche voisin + amélioration 2-opt
        start_time = time.time()
        if nb_ppv > 0:
            starts = random.sample(range(self.N), nb_ppv)
            for start in starts:
                route = self._route_plus_proche_voisin(start)
                route = self._2_opt(route)
                key = tuple(route.ordre)
                if key not in seen:
                    seen.add(key)
                    route.distance = self.graph.calcul_distance_route(route)
                    population.append(route)
        end_time = time.time()
        print(f"  - {nb_ppv} routes plus proche voisin créées en {end_time - start_time:.2f} secondes.")

        # 2) Farthest Insertion + amélioration 2-opt
        start_time_fi = time.time()
        for _ in range(nb_fi):
            route = self._farthest_insertion()
            route = self._2_opt(route)
            key = tuple(route.ordre)
            if key not in seen:
                seen.add(key)
                route.distance = self.graph.calcul_distance_route(route)
                population.append(route)
        end_time2 = time.time()
        print(f"  - {nb_fi} routes farthest insertion créées en {end_time2 - start_time_fi:.2f} secondes.")

        # 3) Grille
        start_time_grid = time.time()
        for _ in range(int(nb_grille)):
            route = self.construire_tour_grille()
            key = tuple(route.ordre)
            if key not in seen:
                seen.add(key)
                population.append(route)
        end_time3 = time.time()
        print(f"  - {int(nb_grille)} routes grille créées en {end_time3 - start_time_grid:.2f} secondes.")

        # 4) Routes aléatoires
        start_time_rand = time.time()
        while len(population) < taille_pop:
            ordre = [0] + random.sample(range(1, self.N), self.N - 1) + [0]
            key = tuple(ordre)
            if key not in seen:
                seen.add(key)
                population.append(self._new_route(ordre))
        end_time4 = time.time()
        print(f"  - {taille_pop - (nb_ppv + nb_fi + int(nb_grille))} routes aléatoires créées en {end_time4 - start_time_rand:.2f} secondes.")

        return population

    # Opérateurs génétiques
    def _tournoi(self, k: int = 3) -> Route:
        """Sélection par tournoi sur la population courante."""
        participants = random.sample(self.population, k)
        participants.sort()
        return participants[0]

    def _croisement_OX(self, parent1: Route, parent2: Route) -> list[int]:
        """exemple :  parent1 = [0 1 2|3 4 5|6 0]
                      parent2 = [0 4 5 6 1 2 3 0]
                      a=2, b=5
                      enfant = [0 _ _ 3 4 5 _ 0]  copie du segment de parent1
                      enfant = [0 6 1 3 4 5 2 0]  complété avec parent2 en conservant l'ordre
        """
        p1 = parent1.ordre[1:-1]
        p2 = parent2.ordre[1:-1]

        a, b = sorted(random.sample(range(len(p1)), 2))
        enfant = [None] * len(p1)

        # copie segment de p1
        enfant[a:b] = p1[a:b]
        lieux_utilises = set(p1[a:b])

        # complète avec p2 en conservant l'ordre
        idx_enfant = b % len(p1)
        for lieu in p2:
            if lieu in lieux_utilises:
                continue
            enfant[idx_enfant] = lieu
            lieux_utilises.add(lieu)
            idx_enfant = (idx_enfant + 1) % len(p1)

        # réinsère les 0 aux extrémités
        return [0] + enfant + [0]

    def _mutation(self, ordre: list[int]) -> None:
        """Applique une mutation choisie aléatoirement parmi plusieurs."""
        if self.N <= 3:
            return
        indices = range(1, self.N - 1)

        def swap(ordre) -> None:
            i, j = random.sample(indices, 2)
            ordre[i], ordre[j] = ordre[j], ordre[i]
        def inversion(ordre) -> None:
            i, j = sorted(random.sample(indices, 2))
            ordre[i:j] = reversed(ordre[i:j])
        def insertion(ordre) -> None:
            i, j = random.sample(indices, 2)
            lieu = ordre.pop(i)
            ordre.insert(j, lieu)
        def rotate(ordre) -> None:
            i, j = sorted(random.sample(indices, 2))
            segment = ordre[i:j]
            segment = segment[-1:] + segment[:-1]
            ordre[i:j] = segment

        random.choice([swap, inversion, insertion, rotate])(ordre)

    def _reproduction(self, k_tournoi: int = 2) -> list[Route]:
        """
        Crée la population d'enfants avec un croisement OX, en évitant les doublons.
        """
        enfants: list[Route] = []
        seen = set()  # ordres déjà présents parmi les enfants

        while len(enfants) < self.taille_pop_enfants:
            parent1 = self._tournoi(k_tournoi)
            parent2 = self._tournoi(k_tournoi)
            essais = 0
            while parent2 == parent1 and essais < 5:
                parent2 = self._tournoi(k_tournoi)
                essais += 1

            if essais == 5:
                ordre_enfant = parent1.ordre.copy()
                self._mutation(ordre_enfant)
            else:
                ordre_enfant = self._croisement_OX(parent1, parent2)

                if random.random() < self.prob_mutation:
                    self._mutation(ordre_enfant)

            key = tuple(ordre_enfant)
            if key in seen:
                continue
            seen.add(key)

            enfant = self._new_route(ordre_enfant.copy())
            enfants.append(enfant)

        return enfants

    def _selection(self, pop_enfants: list[Route], k_tournoi: int = 3) -> None:
        """
        Sélection nouvelle population :
        - fusion population + enfants
        - tri
        - élitisme (top 5 %)
        - remplissage par tournoi
        - évite de reprendre deux fois la même route.
        """
        routes_uniques = {}
        for r in (self.population + pop_enfants):
            routes_uniques[tuple(r.ordre)] = r

        population_totale = sorted(routes_uniques.values())

        # mise à jour du best global
        if population_totale[0].distance < self.best_route.distance:
            self.best_route = population_totale[0]

        nouvelle_population: list[Route] = []

        # Elitisme : top 5 %
        n_5_pourcents = max(1, int(self.taille_pop * 0.05))
        elite = population_totale[:n_5_pourcents]
        nouvelle_population.extend(elite)

        # pour éviter les doublons dans la nouvelle population
        seen = {tuple(r.ordre) for r in nouvelle_population}

        # reste des candidats
        reste = population_totale[n_5_pourcents:]

        # Tournois jusqu'à remplir la population
        while len(nouvelle_population) < self.taille_pop and reste:
            participants = random.sample(reste, min(k_tournoi, len(reste)))
            gagnant = min(participants, key=lambda r: r.distance)

            key = tuple(gagnant.ordre)
            if key not in seen:
                nouvelle_population.append(gagnant)
                seen.add(key)

            # on retire le gagnant du reste pour éviter de le resélectionner
            reste.remove(gagnant)

        self.population = nouvelle_population

    def step(self):
        """
        Effectue une génération :
        - reproduction -> enfants
        - sélection -> nouvelle population
        Retourne la population courante (pour l'IHM).
        """
        pop_enfants = self._reproduction(k_tournoi=2)
        self._selection(pop_enfants, k_tournoi=2)
        return self.population

    def resoudre(self) -> Route:
        """
        Boucle complète sur nb_generations (sans IHM).
        """
        for _ in range(self.nb_generations):
            pop_enfants = self._reproduction()
            self._selection(pop_enfants)
        return self.best_route


if __name__ == "__main__":
    # Exemple d'utilisation simple (sans IHM auto step-by-step)
    graph = Graph(csv_path='fichiers_csv_exemples/graph_20.csv')
    affichage = Affichage(graph)

    taille_pop = min(10, 2*graph.N) if graph.N < 500 else int(5*sqrt(graph.N))+900
    tsp_ga = TSP_GA(
        graph=graph,
        affichage=affichage,
        taille_pop=graph.N,
        taille_pop_enfants=int(graph.N * 0.7),
        prob_mutation=0.2,
        nb_generations=1000
    )

    meilleure_route = tsp_ga.resoudre()
    print("Meilleure route:")
    print(meilleure_route)
