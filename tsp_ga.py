from tsp_graph_init import Graph, Affichage, Route, Lieu
import random
import numpy as np

# Paramètres
TAILLE_POPULATION = 100
PROB_MUTATION = 0.1
NOMBRE_GENERATIONS = 500


def nombre_doublons(liste):
    l1 = len(liste)
    l2 = len(list(set(liste)))
    return l1 - l2


class TSP_GA:
    def __init__(self, graph: Graph, affichage: Affichage,
                 taille_pop=TAILLE_POPULATION, taille_pop_enfants=int(TAILLE_POPULATION*0.7),
                 prob_mutation=PROB_MUTATION, nb_generations=NOMBRE_GENERATIONS
                 ):
        self.graph: Graph = graph
        self.affichage: Affichage = affichage
        self.taille_pop = taille_pop
        self.taille_pop_enfants = taille_pop_enfants
        self.prob_mutation = prob_mutation
        self.nb_generations = nb_generations

        self.population: list[Route] = self._creer_pop_initiale(taille_pop)
        self.population.sort()
        self.best_route: Route = self.population[0]

    def _creer_pop_initiale(self, taille_pop: int) -> list[Route]:
        population = []
        for i in range(0, taille_pop, 2):
            route = self._route_plus_proche_voisin(depart=i % self.graph.N)
            route.distance = self.graph.calcul_distance_route(route)
            population.append(route)

        population_unique = list(set(population))
        nb_routes_manquantes = taille_pop - len(population_unique)
        print("nombre de doublons : ", len(population) - len(population_unique))
        l = len(population)
        for j in range(nb_routes_manquantes):
            route = Route(self._mutation(population[j % l].ordre))
            route.distance = self.graph.calcul_distance_route(route)
            population_unique.append(route)

        return population_unique

    def _route_plus_proche_voisin(self, depart: int) -> Route:
        route = Route([depart])

        while len(route) < self.graph.N:
            dernier_lieu = route[-1]
            prochain_lieu = self.graph.plus_proche_voisin(dernier_lieu, set(route.ordre))
            route.ajouter_lieu(prochain_lieu)
        route.ajouter_lieu(depart)  # Retour au point de départ
        route.reordonner()
        return route

    def _tournoi(self, k: int = 3) -> Route:
        participants = random.sample(self.population, k)
        participants.sort()
        return participants[0]

    def _croisement_OX(self, parent1: Route, parent2: Route) -> list:
        a, b = sorted(random.sample(range(1, self.graph.N-1), 2))

        enfant = [None] * (self.graph.N+1)
        enfant[-1] = 0
        enfant[a:b] = parent1[a:b]  # copie une partie de parent1 dans enfant
        lieux_utilises = set(parent1[a:b])

        # place les villes restantes en préservant l'ordre de parent2 en évitant les doublons dans enfant
        i = 0
        saut = False
        for lieu in parent2:
            if lieu not in lieux_utilises:
                if i >= a and not saut:
                    i = b
                    saut = True
                enfant[i] = lieu
                lieux_utilises.add(lieu)
                i += 1

        return enfant

    def _mutation(self, enfant: list) -> list:
        if random.random() < 0.5:
            a, b = random.sample(range(1, self.graph.N - 1), 2)
            enfant[a], enfant[b] = enfant[b], enfant[a]
        else:
            a, b, c, d = random.sample(range(1, self.graph.N - 1), 4)
            enfant[a], enfant[b] = enfant[b], enfant[a]
            enfant[c], enfant[d] = enfant[d], enfant[c]
        return enfant

    def _reproduction(self, k_tournoi: int = 2) -> list[Route]:
        population_enfants = []

        while len(population_enfants) < self.taille_pop_enfants:
            parent1 = self._tournoi(k_tournoi)
            parent2 = self._tournoi(k_tournoi)
            essais = 0
            while parent2 == parent1 and essais < 10:
                parent2 = self._tournoi(k_tournoi)
                essais += 1

            enfant = self._croisement_OX(parent1, parent2)

            if random.random() < self.prob_mutation:
                if self.graph.calcul_distance_route(Route(enfant)) < self.best_route.distance:
                    self.best_route = enfant
                else:
                    enfant = self._mutation(enfant)

            enfant = Route(enfant)
            enfant.distance = self.graph.calcul_distance_route(enfant)
            population_enfants.append(enfant)

        population_enfants_unique = list(set(population_enfants))
        nb_doublons = self.taille_pop_enfants - len(population_enfants_unique)

        for j in range(nb_doublons):
            ordre = [0] + random.sample(range(1, self.graph.N), self.graph.N-1) + [0]
            route = Route(ordre)
            route.distance = self.graph.calcul_distance_route(route)
            population_enfants_unique.append(route)

        print("nombre de doublons enfants: ", nombre_doublons(population_enfants_unique))
        return population_enfants_unique

    def _selection(self, pop_enfants, k_tournoi: int = 3):
        population_totale = sorted(self.population + pop_enfants)
        if population_totale[0].distance < self.best_route.distance:
            self.best_route = population_totale[0]
        nouvelle_population = []

        # Elitisme : garde le top 5%
        n_5_pourcents = max(1, int(self.taille_pop * 0.05))
        nouvelle_population[:n_5_pourcents] = population_totale[:n_5_pourcents]
        del population_totale[:n_5_pourcents]

        # Selection par tournoi
        while len(nouvelle_population) < self.taille_pop:
            participants = random.sample(range(len(population_totale)), k_tournoi)
            gagnant_index = min(participants, key=lambda i: population_totale[i].distance)
            nouvelle_population.append(population_totale[gagnant_index])
            del population_totale[gagnant_index]

        print("nombre de doublons après selection : ", nombre_doublons(nouvelle_population))
        self.population = nouvelle_population

    def step(self):
        """
        Effectue une SEULE génération :
        - crée les enfants
        - fait la sélection
        - met à jour self.population et self.best_route
        Retourne (best_route, population).
        """
        print(self.population)

        pop_enfants = self._reproduction()
        self._selection(pop_enfants)
        # self.current_generation += 1
        return self.population
    
    def resoudre(self) -> Route:
        for i in range(self.nb_generations):
            print(self.population)
            pop_enfants = self._reproduction()
            self._selection(pop_enfants)
        return self.best_route
        



if __name__ == "__main__":
    # Initialisation du graphe et de l'affichage
    graph = Graph(csv_path='fichiers_csv_exemples/graph_20.csv')
    affichage = Affichage(graph)

    # Initialisation et exécution de l'algorithme génétique
    tsp_ga = TSP_GA(graph, affichage,
                    taille_pop=graph.N,
                    taille_pop_enfants=int(graph.N*0.7),
                    prob_mutation=0.25,
                    nb_generations=100)

    meilleure_route = tsp_ga.resoudre()
    print("Meilleure route:")
    print(meilleure_route)