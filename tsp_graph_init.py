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

    def __len__(self):
        return len(self.ordre)

    def __getitem__(self, index):
        return self.ordre[index]
    
    def ajouter_lieu(self, lieu_index: int):
        """Ajoute un lieu à la fin de la route."""
        self.ordre.append(lieu_index)
        
    def inserer_lieu(self, index, position):
        """Insère un lieu à une position donnée."""
        if position < 0 or position > len(self.ordre) - 1:
            raise ValueError("Position hors limites.")
        self.ordre.insert(position, index)

    def supprimer_lieu(self, index):
        """Supprime un lieu (sauf le départ et l’arrivée)."""
        if index in self.ordre[1:-1]:
            self.ordre.remove(index)

    def reordonner(self):
        """Réorganise la route pour commencer et finir à 0."""
        if 0 not in self.ordre:
            raise ValueError("La route doit contenir le lieu de départ/arrivée (0).")
        idx_zero = self.ordre.index(0)
        nouvelle_ordre = self.ordre[idx_zero:] + self.ordre[1:idx_zero + 1]
        self.ordre = nouvelle_ordre
    
    def __lt__(self, other: "Route"):
        """inférieur à en fonction de la distance totale"""
        return self.distance < other.distance
    
    def __gt__(self, other: "Route"):
        """supérieur à en fonction de la distance totale"""
        return self.distance > other.distance
    
    def __le__(self, other: "Route"):
        """inférieur ou égal en fonction de la distance totale"""
        return self.distance <= other.distance

    def __ge__(self, other: "Route"):
        """supérieur ou égal en fonction de la distance totale"""
        return self.distance >= other.distance  

    def __eq__(self, other: "Route"):
        """deux routes sont égales si elles ont le même ordre de lieux"""
        return self.ordre == other.ordre

    def __neq__(self, other: "Route"):
        """si l'ordre des sommets n'est pas le même"""
        return not self.__eq__(other)
    
    def __repr__(self):
        return f"d={self.distance:.2f}, ordre={self.ordre})"


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
# Classe Affichage (style prof + step-by-step)
# =========================
class Affichage:
    """
    Affichage du TSP pour l'algorithme génétique.
    - Dessine les lieux (ville 0 en rouge)
    - Meilleure route en bleu pointillé
    - Les 5 routes suivantes en gris pointillé
    - Barre d'état en bas avec pourcentage et distance
    - Bouton 'Génération suivante' pour avancer étape par étape
    """

    def __init__(self, graph: Graph, titre="TSP - Algorithme génétique"):
        self.graph = graph
        self.tsp_ga = None          # sera fourni par set_ga(...)
        self.generation = 0
        self.nb_generations = 1     # sera mis à jour quand on a le GA

        self.best_route: Route | None = None
        self.population: list[Route] = []   # population courante

        # ----- Fenêtre -----
        self.root = tk.Tk()
        self.root.title(titre)

        # ----- Barre du haut (bouton) -----
        topbar = tk.Frame(self.root, bg="#f2f2f2")
        topbar.pack(side="top", fill="x")

        self.btn_next = tk.Button(
            topbar,
            text="Génération suivante",
            command=self._next_generation
        )
        self.btn_next.pack(side="left", padx=8, pady=5)

        # raccourci clavier : touche "n"
        self.root.bind("<Key-n>", lambda e: self._next_generation())
        # quitter avec Echap
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # ----- Canvas -----
        self.canvas_h = HAUTEUR - 20
        self.canvas = tk.Canvas(
            self.root,
            width=LARGEUR,
            height=self.canvas_h,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack(side="top", fill="both", expand=True)

        # ----- Barre d'état en bas -----
        status_frame = tk.Frame(self.root, height=20, bg="#e6e6e6")
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)

        self.lbl_status = tk.Label(
            status_frame,
            text="[0%] distance : -  trouvée en 0/0 générations",
            bg="#e6e6e6"
        )
        self.lbl_status.pack(side="left", padx=10)

        # Transform pour adapter les coordonnées au canvas
        self._transform = None
        self._compute_transform()

        # Premier dessin : fond + points
        self.redraw_base()

    # ----------------- Connexion avec le GA -----------------
    def set_ga(self, tsp_ga: "TSP_GA"):
        """Connecte l'algorithme génétique à l'affichage."""
        self.tsp_ga = tsp_ga
        self.nb_generations = tsp_ga.nb_generations
        self.generation = 0

        # récupération de la population initiale
        self.population = sorted(list(tsp_ga.population))
        self.best_route = self.population[0] if self.population else None

        self._update_status()
        self._draw_routes()

    # ----------------- Gestion des générations -----------------
    def _next_generation(self):
        """Appelé quand on clique sur le bouton ou appuie sur 'n'."""
        if self.tsp_ga is None:
            self.lbl_status.config(text="Pas d'algo (utilise set_ga(tsp_ga)).")
            return

        if self.generation >= self.nb_generations:
            # déjà au bout
            if self.best_route is not None:
                self.lbl_status.config(
                    text=f"[100%] distance : {self.best_route.distance:.3f}  "
                         f"trouvée en {self.generation}/{self.nb_generations} générations"
                )
            return

        # une génération de plus dans le GA
        # on se fiche du retour éventuel, on lit directement la population du GA
        self.tsp_ga.step()
        self.generation += 1

        # met à jour population + best_route
        self.population = sorted(list(self.tsp_ga.population))
        self.best_route = self.population[0] if self.population else None

        # redraw complet
        self.redraw_base()
        self._draw_routes()
        self._update_status()

    # ----------------- Géométrie & mapping -----------------
    def _compute_transform(self):
        xs = [lieu.x for lieu in self.graph.liste_lieux] or [0.0]
        ys = [lieu.y for lieu in self.graph.liste_lieux] or [0.0]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)

        W = LARGEUR
        H = self.canvas_h

        scale = min((W - 2 * MARGE) / w, (H - 2 * MARGE) / h)

        offx = MARGE - xmin * scale + (W - w * scale) / 2
        offy = MARGE - ymin * scale + (H - h * scale) / 2

        self._transform = (scale, offx, offy)

    def _to_canvas(self, x: float, y: float):
        if self._transform is None:
            self._compute_transform()
        s, ox, oy = self._transform
        return x * s + ox, y * s + oy

    # ----------------- Dessin -----------------
    def redraw_base(self):
        """Efface le canvas et redessine la grille + les points."""
        self.canvas.delete("all")
        self._draw_background()
        self._draw_points()

    def _draw_background(self):
        """Grille légère en fond."""
        W = LARGEUR
        H = self.canvas_h
        step = 50
        for x in range(0, W, step):
            self.canvas.create_line(x, 0, x, H, fill="#f0f0f0")
        for y in range(0, H, step):
            self.canvas.create_line(0, y, W, y, fill="#f0f0f0")

    def _draw_points(self):
        """Dessine les lieux, ville 0 en rouge, les autres en gris."""
        r = RAYON_LIEU
        ordre_index = {}
        if self.best_route is not None:
            # On prend l'ordre SANS le retour final à 0 pour numérotation
            for k, idx in enumerate(self.best_route.ordre[:-1]):
                ordre_index[idx] = k

        for i, lieu in enumerate(self.graph.liste_lieux):
            X, Y = self._to_canvas(lieu.x, lieu.y)

            if i == 0:
                fill = "#d64545"   # rouge
                outline = "#222222"
            else:
                fill = "#dddddd"   # gris clair
                outline = "#555555"

            self.canvas.create_oval(
                X - r, Y - r, X + r, Y + r,
                fill=fill, outline=outline, width=2
            )

            # index de la ville (dans le cercle)
            self.canvas.create_text(X, Y, text=str(i),
                                    font=("Arial", 10, "bold"))

            # rang dans la tournée (au-dessus du cercle), si connu
            if i in ordre_index:
                self.canvas.create_text(
                    X, Y - r - 8,
                    text=str(ordre_index[i]),
                    font=("Arial", 8),
                    fill="#333333"
                )

    def _draw_route(self, route: Route, color, dash, width):
        if route is None or not route.ordre or len(route.ordre) < 2:
            return

        pts = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            pts.append(self._to_canvas(lieu.x, lieu.y))

        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=width,
                dash=dash
            )

    def _draw_routes(self):
        """
        Dessine les 6 meilleures routes :
        - la meilleure en bleu
        - les 5 suivantes en gris
        """
        if not self.population:
            return

        pop_sorted = sorted(self.population)
        best = pop_sorted[0]
        others = pop_sorted[1:6]   # les 5 suivantes (ou moins si population < 6)
        print("others:", others)
        # Les 5 suivantes en gris pointillé
        for r in others:
            self._draw_route(r, color="#C91F1F", dash=(2, 4), width=1)

        # Meilleure actuelle en bleu pointillé
        self._draw_route(best, color="#0066cc", dash=(4, 3), width=2)

    # ----------------- Barre d'état -----------------
    def _update_status(self):
        if self.best_route is None or self.best_route.distance is None:
            txt = "[0%] distance : -  trouvée en 0/0 générations"
        else:
            pct = min(100, int(100 * self.generation / max(1, self.nb_generations)))
            txt = (
                f"[{pct}%] distance : {self.best_route.distance:.3f}  "
                f"trouvée en {self.generation}/{self.nb_generations} générations"
            )
        self.lbl_status.config(text=txt)

    # ----------------- Boucle principale -----------------
    def run(self):
        self.root.mainloop()



# =========================
# Exécution directe (démo sans route)
# =========================
# if __name__ == "__main__":
#     # g = Graph(csv_path="fichiers_csv_exemples/graph_20.csv",seed=2)   # <-- Chargement du csv
#     # ui = Affichage(g, nom_groupe="Groupe_10")  # <-- Affichage
#     # ui.run()

if __name__ == "__main__":
    from tsp_ga import TSP_GA  # adapte le nom du fichier si besoin
    graph = Graph(csv_path="fichiers_csv_exemples/graph_20.csv")

    affichage = Affichage(graph, titre="UI")  # si tu veux le même titre que sur la capture

    tsp_ga = TSP_GA(
        graph=graph,
        affichage=affichage,
        taille_pop=graph.N,
        taille_pop_enfants=int(graph.N * 0.7),
        prob_mutation=0.1,
        nb_generations=100,  
    )

    # La population initiale est déjà créée dans TSP_GA, et best_route aussi
    affichage.set_ga(tsp_ga)

    # Fenêtre Tkinter
    affichage.run()
