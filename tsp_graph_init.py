# tsp_graph_init.py
# Dépendances autorisées uniquement
import csv
import random
import numpy as np
import tkinter as tk

# =========================
# Constantes
# =========================
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 10
RAYON_LIEU = 12
MARGE = 30
NOM_GROUPE = "GROUPE_10"  

# =========================
# Classe Lieu
# =========================
class Lieu:
    """Mémorise (x, y, nom) et calcule la distance euclidienne vers un autre Lieu."""
    def __init__(self, x: float, y: float, nom: str):
        self.x = float(x)
        self.y = float(y)
        self.nom = str(nom)

    def distance(self, autre: "Lieu") -> float:
        dx = self.x - autre.x
        dy = self.y - autre.y
        return float(np.sqrt(dx * dx + dy * dy))

    def __repr__(self):
        return f"Lieu(nom={self.nom}, x={self.x:.1f}, y={self.y:.1f})"


# =========================
# Classe Route
# =========================
class Route:
    """Représente un ordre de visite, avec contrainte 0 ... 0 (départ/arrivée)."""
    def __init__(self, ordre):
        self.ordre = list(ordre)
        self.distance = None  # Calculée ultérieurement via Graph.calcul_distance_route(route)
    
    def ajouter_lieu(self, lieu_index: int):
        """Ajoute un lieu avant le retour au point de départ (dernier élément)."""
        if len(self.ordre) < 2:
            # Cas d'une route vide ou mal initialisée
            self.ordre = [0, lieu_index, 0]
        else:
            # Insère avant le dernier 0 (retour)
            self.ordre.insert(-1, lieu_index)
        
    def inserer_lieu(self, index, position):
        """Insère un lieu à une position donnée."""
        if position < 0 or position > len(self.ordre) - 1:
            raise ValueError("Position hors limites.")
        self.ordre.insert(position, index)

    def supprimer_lieu(self, index):
        """Supprime un lieu (sauf le départ et l’arrivée)."""
        if index in self.ordre[1:-1]:
            self.ordre.remove(index)
    
    def __lt__ (self, other: "Route"):
        """Compare les distances des routes."""
        return self.distance < other.distance
    
    def __gt__ (self, other: "Route"):
        """Compare les distances des routes."""
        return self.distance > other.distance
    
    def __le__ (self, other: "Route"):
        """Compare les distances des routes."""
        return self.distance <= other.distance

    def __ge__ (self, other: "Route"):
        """Compare les distances des routes."""
        return self.distance >= other.distance  

    def __eq__ (self, other: "Route"):
        """si c'est les mêmes sommets mais dans un ordre différent"""
        return set(self.ordre) == set(other.ordre) and len(self.ordre) == len(other.ordre)

    def __neq__ (self, other: "Route"):
        """si ce n'est pas les mêmes sommets"""
        return not self.__eq__(other)
    
    def __repr__(self):
        return f"Route(ordre={self.ordre})"


# =========================
# Classe Graph
# =========================
class Graph:
    """
    - Mémorise la liste des lieux
    - Génère aléatoirement ou charge depuis CSV
    - Calcule la matrice des distances (matrice_od)
    - Fournit plus_proche_voisin(i, visites) (exigé par l'énoncé)
    """
    def __init__(self, nb_lieux=NB_LIEUX, csv_path=None, seed=None):
        if seed is not None:
            random.seed(seed)

        self.liste_lieux = []
        
        self.matrice_od = None

        if csv_path:
            self.charger_graph(csv_path)
        else:
            self._generer_aleatoire(nb_lieux)
        self.N = len(self.liste_lieux)

        self.calcul_matrice_cout_od()

    def _generer_aleatoire(self, nb_lieux):
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(MARGE, LARGEUR - MARGE)
            y = random.uniform(MARGE, HAUTEUR - MARGE)
            self.liste_lieux.append(Lieu(x, y, nom=str(i)))

    def charger_graph(self, csv_path: str):
        """
        CSV acceptés (même structure que Moodle) :
          - nom,x,y   (ou id,x,y)
          - x,y       (nom auto: index de ligne)
        """
        self.liste_lieux = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            sample = f.read(1024)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            reader = csv.reader(f, delimiter=",")

            headers = None
            if has_header:
                headers = [h.strip().lower() for h in next(reader)]

            idx_nom = idx_x = idx_y = None
            if headers:
                if "x" in headers and "y" in headers:
                    idx_x = headers.index("x")
                    idx_y = headers.index("y")
                if "nom" in headers:
                    idx_nom = headers.index("nom")
                elif "id" in headers:
                    idx_nom = headers.index("id")

            i_ligne = 0
            for row in reader:
                if not row:
                    continue
                if headers:
                    x = float(row[idx_x]) if idx_x is not None else float(row[0])
                    y = float(row[idx_y]) if idx_y is not None else float(row[1])
                    nom = row[idx_nom] if idx_nom is not None else str(i_ligne)
                else:
                    if len(row) >= 3:
                        x, y, nom = float(row[0]), float(row[1]), str(row[2])
                    else:
                        x, y, nom = float(row[0]), float(row[1]), str(i_ligne)
                self.liste_lieux.append(Lieu(x, y, nom))
                i_ligne += 1

        if not self.liste_lieux:
            raise ValueError("CSV vide ou illisible.")

    def calcul_matrice_cout_od(self):
        n = len(self.liste_lieux)
        self.matrice_od = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.matrice_od[i, j] = 0.0
                else:
                    self.matrice_od[i, j] = self.liste_lieux[i].distance(self.liste_lieux[j])

    def plus_proche_voisin(self, i: int, visites: set) -> int:
        """
        Renvoie l'indice du plus proche voisin non encore visité depuis i.
        (Conservée car demandée par l'énoncé, mais non utilisée ici.)
        """
        n = len(self.liste_lieux)
        best_j = None
        best_d = float("inf")
        for j in range(n):
            if j == i or j in visites:
                continue
            d = self.matrice_od[i, j]
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    def calcul_distance_route(self, route: Route) -> float:
        """Somme des distances le long de route.ordre (euclidienne)."""
        total = 0.0
        for a, b in zip(route.ordre[:-1], route.ordre[1:]):
            total += self.matrice_od[a, b]
        return float(total)


# =========================
# Classe Affichage
# =========================
class Affichage:
    """
    Affiche uniquement ce qu'on lui fournit :
      - les lieux du graph
      - une route courante (optionnelle) via set_route(Route)
      - une population (optionnelle) via set_population(list[Route])
    Raccourcis:
      P : toggle population (si définie)
      M : toggle matrice des coûts (texte)
      Échap : quitter
    """
    def __init__(self, graph: Graph, nom_groupe=NOM_GROUPE):
        self.graph = graph
        self.root = tk.Tk()
        self.root.title(f"TSP - {nom_groupe}")

        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.text = tk.Text(self.root, height=10, width=100)
        self.text.pack(fill="x")

        self._route = None
        self._population = []
        self._show_population = False
        self._show_matrix = False

        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Key-p>", self._toggle_population)
        self.root.bind("<Key-P>", self._toggle_population)
        self.root.bind("<Key-m>", self._toggle_matrix)
        self.root.bind("<Key-M>", self._toggle_matrix)

        self.redraw()

    # ------- API d'injection depuis tes algos -------
    def set_route(self, route: Route):
        self._route = route
        self._log(f"Route mise à jour : {route.ordre} | distance = {self.graph.calcul_distance_route(route):.2f}")
        self.redraw()

    def set_population(self, routes: list):
        self._population = list(routes) if routes else []
        self._log(f"Population mise à jour : {len(self._population)} routes")
        self.redraw()

    # ------- UI -------
    def _log(self, msg: str):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def _toggle_population(self, _evt=None):
        if not self._population:
            self._log("Aucune population définie (utilise set_population([...])).")
            return
        self._show_population = not self._show_population
        self._log(f"Affichage population = {self._show_population}")
        self.redraw()

    def _toggle_matrix(self, _evt=None):
        self._show_matrix = not self._show_matrix
        if self._show_matrix:
            self._afficher_matrice_od()
        else:
            self._log("Matrice masquée.")

    def _afficher_matrice_od(self):
        self._log("Matrice des coûts (distances euclidiennes):")
        mat = self.graph.matrice_od
        n = mat.shape[0]
        header = "     " + " ".join([f"{j:>7d}" for j in range(n)])
        self._log(header)
        for i in range(n):
            row_vals = " ".join([f"{mat[i, j]:7.1f}" for j in range(n)])
            self._log(f"{i:>3d}  {row_vals}")

    def _draw_lieu(self, x, y, label, order_idx=None):
        r = RAYON_LIEU
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="black", width=2)
        self.canvas.create_text(x, y, text=str(label), font=("Arial", 10, "bold"))
        if order_idx is not None:
            self.canvas.create_text(x, y - r - 10, text=str(order_idx), font=("Arial", 9))

    def _draw_route(self, route: Route, dash=(6, 4), color="blue", width=2):
        if not route or not route.ordre or len(route.ordre) < 2:
            return
        pts = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            pts.append((lieu.x, lieu.y))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            self.canvas.create_line(x1, y1, x2, y2, fill=color, dash=dash, width=width)

    def redraw(self):
        self.canvas.delete("all")

        # Population en fond (gris clair), si activée
        if self._show_population and self._population:
            for r in self._population:
                self._draw_route(r, dash=(), color="#CCCCCC", width=1)

        # Route principale (si fournie)
        if self._route:
            self._draw_route(self._route, dash=(6, 4), color="blue", width=2)
            # Indices d'ordre au-dessus des lieux selon la route
            ordre_index = {idx: k for k, idx in enumerate(self._route.ordre)}
        else:
            ordre_index = {}

        # Lieux
        for i, lieu in enumerate(self.graph.liste_lieux):
            order_idx = ordre_index.get(i, None)
            self._draw_lieu(lieu.x, lieu.y, i, order_idx=order_idx)

    def run(self):
        self.root.mainloop()


# =========================
# Exécution directe (démo sans route)
# =========================
if __name__ == "__main__":
    csv_path = "graph_5.csv" #nom du fichier csv à charger
    g = Graph(nb_lieux=NB_LIEUX, csv_path=csv_path, seed=42)
    ui = Affichage(g, nom_groupe=NOM_GROUPE)
    # Exemple d’injection ultérieure (depuis ta future classe d'algo) :
    #   r = Route([0,3,1,2,4,5,6,7,8,9,0])
    #   ui.set_route(r)
    #   ui.set_population([r1, r2, ...])
    ui.run()
