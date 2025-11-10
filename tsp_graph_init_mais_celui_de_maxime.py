import numpy as np
import random
import time
import pandas as pd
import tkinter as tk
from tkinter import Canvas, Text, BOTH

# Constantes globales
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 5  # Peut être modifié selon le besoin


# Classe Lieu
class Lieu:
    def __init__(self, nom, x, y):
        self.nom = nom
        self.x = x
        self.y = y

    def distance(self, autre_lieu):
        """Calcule la distance euclidienne entre deux lieux"""
        return np.sqrt((self.x - autre_lieu.x) ** 2 + (self.y - autre_lieu.y) ** 2)

    def __repr__(self):
        return f"Lieu({self.nom}, x={self.x:.2f}, y={self.y:.2f})"


# Classe Graph
class Graph:
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None

    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX):
        """Génère des lieux avec des coordonnées aléatoires"""
        self.liste_lieux = [
            Lieu(i, random.uniform(0, LARGEUR), random.uniform(0, HAUTEUR))
            for i in range(nb_lieux)
        ]

    def charger_graph(self, chemin_csv):
        """Charge un graphe à partir d’un fichier CSV (colonnes x, y)"""
        data = pd.read_csv(chemin_csv)
        self.liste_lieux = [
            Lieu(i, float(row["x"]), float(row["y"])) for i, row in data.iterrows()
        ]

    def calcul_matrice_cout_od(self):
        """Calcule la matrice des distances entre tous les lieux"""
        n = len(self.liste_lieux)
        self.matrice_od = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.matrice_od[i, j] = self.liste_lieux[i].distance(self.liste_lieux[j])
        return self.matrice_od

    def plus_proche_voisin(self, indice_lieu):
        """Retourne l'indice du plus proche voisin d’un lieu donné"""
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        distances = self.matrice_od[indice_lieu]
        voisins = [(i, d) for i, d in enumerate(distances) if i != indice_lieu]
        return min(voisins, key=lambda x: x[1])[0]

    def calcul_distance_route(self, route):
        """Calcule la distance totale d'une route"""
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        distance_totale = 0.0
        for i in range(len(route.ordre) - 1):
            a, b = route.ordre[i], route.ordre[i + 1]
            distance_totale += self.matrice_od[a, b]
        return distance_totale


# Classe Route
class Route:
    def __init__(self, graph, ordre=None):
        self.graph = graph
        if ordre:
            self.ordre = ordre
        else:
            # Génère une route aléatoire qui commence et finit au lieu 0
            indices = list(range(1, len(graph.liste_lieux)))
            random.shuffle(indices)
            self.ordre = [0] + indices + [0]

    def distance_totale(self):
        """Retourne la distance totale de la route"""
        return self.graph.calcul_distance_route(self)

    def __repr__(self):
        return f"Route({self.ordre}) - distance={self.distance_totale():.2f}"


# Classe Affichage
class Affichage:
    def __init__(self, graph, route=None, nom_groupe="Groupe_X"):
        self.graph = graph
        self.route = route
        self.root = tk.Tk()
        self.root.title(f"TP Voyageur de Commerce - {nom_groupe}")

        self.canvas = Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="white")
        self.canvas.pack()

        self.zone_texte = Text(self.root, height=5)
        self.zone_texte.pack(fill=BOTH)

        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # Exemple : touche 'r' pour afficher la route
        self.root.bind("r", lambda e: self.afficher_route())

    def afficher_lieux(self):
        """Dessine les lieux sur le canvas"""
        rayon = 8
        for lieu in self.graph.liste_lieux:
            x, y = lieu.x, lieu.y
            self.canvas.create_oval(x - rayon, y - rayon, x + rayon, y + rayon, fill="lightblue")
            self.canvas.create_text(x, y, text=str(lieu.nom), fill="black")

    def afficher_route(self):
        """Affiche la route sous forme de ligne bleue"""
        if not self.route:
            return
        self.canvas.delete("all")
        self.afficher_lieux()

        for i in range(len(self.route.ordre) - 1):
            a = self.graph.liste_lieux[self.route.ordre[i]]
            b = self.graph.liste_lieux[self.route.ordre[i + 1]]
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill="blue", dash=(4, 2))
            # Affiche l’ordre au-dessus du lieu
            self.canvas.create_text(a.x, a.y - 15, text=str(i), fill="red")

        self.ajouter_texte(f"Distance totale : {self.route.distance_totale():.2f}")

    def ajouter_texte(self, texte):
        """Ajoute du texte dans la zone de messages"""
        self.zone_texte.insert(tk.END, texte + "\n")
        self.zone_texte.see(tk.END)

    def lancer(self):
        """Lance la boucle principale Tkinter"""
        self.afficher_lieux()
        self.root.mainloop()


# =======================
# Exemple d'utilisation
# =======================
if __name__ == "__main__":
    g = Graph()
    g.charger_graph("graph_5.csv")  # csv utilisé pour le chargement
    g.calcul_matrice_cout_od()

    route = Route(g)
    print(route)

    affichage = Affichage(g, route, nom_groupe="MonGroupe")
    affichage.lancer()
