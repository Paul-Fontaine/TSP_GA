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
NB_LIEUX = 100
RAYON_LIEU = 12
MARGE = 10
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

    def __hash__(self):
        return hash((tuple(self.ordre), self.distance))

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
# Classe Affichage (auto-update sur amélioration)
# =========================
# =========================
# Classe Affichage (auto-update sur amélioration)
# =========================
class Affichage:
    """
    Affichage pour TSP_GA :
    - Dessine les lieux (0 en rouge) + grille si N <= 1000
    - Au-delà de 1000 lieux, n'affiche plus les points (uniquement les tracés)
    - Affiche la meilleure route en bleu (pointillé)
    - Affiche jusqu'à 5 routes secondaires en gris **UNIQUEMENT** si P est activé
    - Lance l'algorithme en continu et NE REDESSINE que lorsqu'une meilleure distance est trouvée
    """

    def __init__(self, graph: Graph, titre="TSP - Algorithme génétique (auto)"):
        self.graph = graph
        self.tsp_ga = None

        self.best_route: Route | None = None       # meilleure route affichée
        self.best_distance_affichee: float | None = None
        self.population: list[Route] = []

        self._show_population = False  # routes secondaires visibles ou non

        # ----- Fenêtre -----
        self.root = tk.Tk()
        self.root.title(titre)

        # ----- Canvas -----
        self.canvas_h = HAUTEUR - 24
        self.canvas = tk.Canvas(
            self.root, width=LARGEUR, height=self.canvas_h,
            bg="white", highlightthickness=0
        )
        self.canvas.pack(side="top", fill="both", expand=True)

        # ----- Barre d'état en bas -----
        status_frame = tk.Frame(self.root, height=24, bg="#e6e6e6")
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)

        self.lbl_status = tk.Label(
            status_frame,
            text="Prêt. (touche P : afficher/masquer routes secondaires)",
            bg="#e6e6e6"
        )
        self.lbl_status.pack(side="left", padx=10)

        # Raccourcis utiles
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Key-p>", self._toggle_population)
        self.root.bind("<Key-P>", self._toggle_population)

        # Transform pour adapter les coordonnées au canvas
        self._transform = None
        self._compute_transform()

        # Premier dessin (fond + éventuels points)
        self.redraw_base()

        # flag d'exécution auto
        self._auto_running = False
        self._delay_ms = 2  # délai entre itérations (ms)

    # --------------- Connexion avec le GA ---------------
    def set_ga(self, tsp_ga: "TSP_GA", start_auto: bool = True):
        """Attache le GA et (optionnel) lance la boucle auto."""
        self.tsp_ga = tsp_ga

        # récupère population & best initiaux
        self.population = sorted(list(tsp_ga.population))
        self.best_route = self.population[0] if self.population else None
        self.best_distance_affichee = (
            self.best_route.distance if self.best_route and self.best_route.distance is not None else None
        )

        # affiche l'état initial
        self.redraw_base()
        self._draw_routes()
        self._update_status(gen=0, nb_gen=tsp_ga.nb_generations)

        if start_auto:
            self.start_auto()

    # --------------- Boucle auto ---------------
    def start_auto(self):
        """Démarre la boucle qui appelle step() en continu et ne redessine que si amélioration."""
        if self.tsp_ga is None or self._auto_running:
            return
        self._auto_running = True
        self._auto_loop()

    def stop_auto(self):
        self._auto_running = False

    def _auto_loop(self):
        if not self._auto_running or self.tsp_ga is None:
            return

        # Une génération
        self.tsp_ga.step()  # met self.population à jour en interne
        self.population = sorted(list(self.tsp_ga.population))
        current_best = self.population[0] if self.population else None
        current_best_dist = current_best.distance if current_best else None

        improved = (
            current_best_dist is not None and
            (self.best_distance_affichee is None or current_best_dist < self.best_distance_affichee)
        )

        if improved:
            self.best_route = current_best
            self.best_distance_affichee = current_best_dist
            # Redessine uniquement si on a mieux
            self.redraw_base()
            self._draw_routes()
            self._update_status(gen=None, nb_gen=self.tsp_ga.nb_generations)

        # Replanifie la prochaine itération
        self.root.after(self._delay_ms, self._auto_loop)

    # --------------- Toggle population (touche P) ---------------
    def _toggle_population(self, _evt=None):
        self._show_population = not self._show_population
        # on redessine simplement en prenant en compte le nouveau flag
        self.redraw_base()
        self._draw_routes()

    # --------------- Géométrie & mapping ---------------
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

    # --------------- Dessin ---------------
    def redraw_base(self):
        """
        Efface et redessine le fond.
        Si le nombre de lieux > 1000 : on ne dessine **pas** les points (uniquement les tracés plus tard).
        Sinon : on dessine la grille + les points.
        """
        self.canvas.delete("all")
        self._draw_background()
        if len(self.graph.liste_lieux) <= 1000:
            self._draw_points()

    def _draw_background(self):
        W = LARGEUR
        H = self.canvas_h
        step = 50
        for x in range(0, W, step):
            self.canvas.create_line(x, 0, x, H, fill="#f0f0f0")
        for y in range(0, H, step):
            self.canvas.create_line(0, y, W, y, fill="#f0f0f0")

    def _draw_points(self):
        """
        Dessine les lieux uniquement si on a <= 1000 points (filtré par redraw_base).
        """
        r = RAYON_LIEU
        ordre_index = {}
        if self.best_route is not None:
            for k, idx in enumerate(self.best_route.ordre[:-1]):
                ordre_index[idx] = k

        for i, lieu in enumerate(self.graph.liste_lieux):
            X, Y = self._to_canvas(lieu.x, lieu.y)
            fill = "#d64545" if i == 0 else "#dddddd"
            outline = "#222222" if i == 0 else "#555555"

            self.canvas.create_oval(X - r, Y - r, X + r, Y + r,
                                    fill=fill, outline=outline, width=2)
            self.canvas.create_text(X, Y, text=str(i), font=("Arial", 10, "bold"))

            if i in ordre_index:
                self.canvas.create_text(X, Y - r - 8, text=str(ordre_index[i]),
                                        font=("Arial", 8), fill="#333333")

    def _draw_route(self, route: Route, color, dash, width):
        if route is None or not route.ordre or len(route.ordre) < 2:
            return
        pts = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            pts.append(self._to_canvas(lieu.x, lieu.y))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash)

    def _draw_routes(self):
        if not self.population:
            return
        pop_sorted = self.population
        best = pop_sorted[0]
        others = pop_sorted[1:6]  # jusqu'à 5 suivantes

        # routes secondaires en gris UNIQUEMENT si _show_population = True
        if self._show_population:
            for r in others:
                self._draw_route(r, color="#c0c0c0", dash=(2, 4), width=1)

        # meilleure route toujours affichée
        self._draw_route(best, color="#0066cc", dash=(4, 3), width=2)

    # --------------- Status ---------------
    def _update_status(self, gen: int | None, nb_gen: int | None):
        if self.best_route is None or self.best_route.distance is None:
            self.lbl_status.config(text="Distance : -")
            return
        if gen is None or nb_gen is None:
            self.lbl_status.config(text=f"Distance : {self.best_route.distance:.3f}")
        else:
            pct = min(100, int(100 * gen / max(1, nb_gen)))
            self.lbl_status.config(text=f"[{pct}%] distance : {self.best_route.distance:.3f}")

    # --------------- Boucle Tk ---------------
    def run(self):
        self.root.mainloop()


# =========================
# Exécution directe
# =========================
if __name__ == "__main__":
    from tsp_ga import TSP_GA  # adapte le nom du fichier si besoin

    graph = Graph(50)  # ou csv_path="fichiers_csv_exemples/graph_20.csv"
    affichage = Affichage(graph, titre="UI")

    tsp_ga = TSP_GA(
        graph=graph,
        affichage=affichage,
        taille_pop=graph.N,
        taille_pop_enfants=int(graph.N * 0.7),
        prob_mutation=0.2,
        nb_generations=300,
    )

    affichage.set_ga(tsp_ga)
    affichage.run()
