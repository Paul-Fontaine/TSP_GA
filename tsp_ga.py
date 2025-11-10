from tsp_graph_init import Graph, Affichage, Route, Lieu

# Paramètres
TAILLE_POPULATION = 100
PROB_MUTATION = 0.2
PROB_CROISEMENT = 0.8
NOMBRE_GENERATIONS = 500


class TSP_GA:
    def __init__(self, graph: Graph, affichage: Affichage):
        self.graph: Graph = graph
        self.affichage: Affichage = affichage
        self.population: list[Route] = self._creer_pop_initiale(10)
        self.population.sort()
        self.best_route: Route = self.population[0]

    def _creer_pop_initiale(self, taille_pop: int) -> list[Route]:
        population = []
        for i in range(taille_pop):
            route = self.route_plus_proche_voisin(depart=i % self.graph.N)
            route.distance = self.graph.calcul_distance_route(route)
            population.append(route)
        return population

    def route_plus_proche_voisin(self, depart: int) -> Route:
        route = Route([depart])

        while len(route) < self.graph.N:
            dernier_lieu = route[-1]
            prochain_lieu = self.graph.plus_proche_voisin(dernier_lieu, set(route.ordre))
            route.ajouter_lieu(prochain_lieu)
        route.ajouter_lieu(depart)  # Retour au point de départ
        route.reordonner()
        return route


if __name__ == "__main__":
    # Initialisation du graphe et de l'affichage
    graph = Graph()
    affichage = Affichage(graph)

    # Initialisation et exécution de l'algorithme génétique
    tsp_ga = TSP_GA(graph, affichage)
    print(f"population initiale: {tsp_ga.population}")