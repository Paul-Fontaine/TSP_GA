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
        self.population: list[Route] = self._creer_pop_initiale(self.taille_pop)
        self.population.sort()
        self.best_route: Route = self.population[0]

    def _new_route(self, ordre: list[int]) -> Route:
        """Crée une Route à partir d'un ordre et calcule sa distance."""
        r = Route(ordre)
        r.distance = self.graph.calcul_distance_route(r)
        return r

    def _creer_pop_initiale(self, taille_pop: int) -> list[Route]:
        """
        Crée une population initiale:
        - une partie issue du plus proche voisin (départs différents)
        - le reste en routes aléatoires
        - pas de doublons (sur l'ordre de visite)
        """
        population: list[Route] = []
        seen = set()

        # 1) Routes "structurées" (plus proche voisin)
        nb_ppv = min(self.N, taille_pop // 2)
        for start in range(nb_ppv):
            r = self._route_plus_proche_voisin(start*2%self.N)
            key = tuple(r.ordre)
            if key in seen:
                continue
            r.distance = self.graph.calcul_distance_route(r)
            population.append(r)
            seen.add(key)

        # 2) Compléter avec des routes aléatoires si nécessaire
        while len(population) < taille_pop:
            ordre = [0] + random.sample(range(1, self.N), self.N - 1) + [0]
            key = tuple(ordre)
            if key in seen:
                continue
            r = self._new_route(ordre)
            population.append(r)
            seen.add(key)

        return population

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

    # Opérateurs génétiques
    def _tournoi(self, k: int = 3) -> Route:
        """Sélection par tournoi sur la population courante."""
        participants = random.sample(self.population, k)
        participants.sort()
        return participants[0]

    def _croisement_OX(self, parent1: Route, parent2: Route) -> list[int]:
        """
        OX (Order Crossover) adapté à un tour 0 ... 0.
        - on travaille sur les positions 1 .. N-1 (les villes, sans les 0)
        - on fixe enfant[0] = enfant[-1] = 0 ensuite
        """
        p1 = parent1.ordre[1:-1]
        p2 = parent2.ordre[1:-1]

        a, b = sorted(random.sample(range(len(p1)), 2))
        enfant = [None] * len(p1)

        # copie segment de p1
        enfant[a:b] = p1[a:b]
        lieux_utilises = set(p1[a:b])

        # complète avec p2 en conservant l'ordre
        idx_enfant = b
        for v in p2:
            if v in lieux_utilises:
                continue
            enfant[idx_enfant] = v
            lieux_utilises.add(v)
            idx_enfant = (idx_enfant + 1) % len(p1)

        # réinsère les 0 aux extrémités
        return [0] + enfant + [0]

    def _mutation(self, ordre: list[int]) -> None:
        """
        Mutation sur place : échange de 2 ou 4 villes (hors index 0 et -1).
        """
        if self.N <= 3:
            return
        indices = range(1, self.N - 1)
        if random.random() < 0.5:
            a, b = random.sample(indices, 2)
            ordre[a], ordre[b] = ordre[b], ordre[a]
        else:
            a, b, c, d = random.sample(indices, 4)
            ordre[a], ordre[b] = ordre[b], ordre[a]
            ordre[c], ordre[d] = ordre[d], ordre[c]

    def _reproduction(self, k_tournoi: int = 2) -> list[Route]:
        """
        Crée la population d'enfants, en évitant les doublons
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

            ordre_enfant = self._croisement_OX(parent1, parent2)

            if random.random() < self.prob_mutation:
                self._mutation(ordre_enfant)

            key = tuple(ordre_enfant)
            if key in seen:
                # on ne l'ajoute pas, on essaie un autre enfant
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
        pop_enfants = self._reproduction()
        self._selection(pop_enfants)
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

    tsp_ga = TSP_GA(
        graph=graph,
        affichage=affichage,
        taille_pop=graph.N,
        taille_pop_enfants=int(graph.N * 0.7),
        prob_mutation=0.25,
        nb_generations=100
    )

    meilleure_route = tsp_ga.resoudre()
    print("Meilleure route:")
    print(meilleure_route)