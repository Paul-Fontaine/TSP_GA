# tsp_graph_init.py
# Dépendances autorisées uniquement
import csv
import random
import numpy as np
import tkinter as tk
import time

# =========================
# Constantes
# =========================
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 100000
RAYON_LIEU = 12
MARGE = 10
NOM_GROUPE = "GROUPE_10"

# Seuils d'affichage
MAX_POINTS = 500        # au-delà, on n'affiche plus les points
MAX_SEGMENTS = 3000     # au-delà, on ne trace pas plus de 3000 segments par route


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
    - Peut générer aléatoirement ou charger depuis CSV
    - Peut calculer une matrice des distances (format triangulaire compact)
    - Bascule automatiquement en mode "calcul à la demande" si trop gros
    - Fournit distance_ij, plus_proche_voisin, calcul_distance_route
    """

    def __init__(self, nb_lieux=NB_LIEUX, csv_path=None, seed=None):
        if seed is not None:
            random.seed(seed)

        self.liste_lieux = []
        self.matrice_od = None
        self._tri_index = None
        self.use_matrix = False   # sera ajusté automatiquement

        # Chargement ou génération
        if csv_path:
            self.charger_graph(csv_path)
        else:
            self._generer_aleatoire(nb_lieux)

        self.N = len(self.liste_lieux)

        #-------------------------------------------------------
        # Limite stricte : 5 Go pour la matrice distances
        # ------------------------------------------------------
        max_bytes = int(5 * 1024 * 1024 * 1024)  # 5 Go
        needed_bytes = ((self.N * (self.N - 1)) // 2) * 4  # float32

        if needed_bytes <= max_bytes:
            self.use_matrix = True
            print(f"[INFO] Matrice distances permise ({needed_bytes/1e9:.2f} Go).")
            self.calcul_matrice_cout_od()
        else:
            self.use_matrix = False
            print(f"[INFO] Matrice distances trop grosse ({needed_bytes/1e9:.2f} Go) : calcul direct.")


        # Stockage vectorisé des coordonnées (utilisé même en mode matrice)
        self.coords = np.array(
            [(lieu.x, lieu.y) for lieu in self.liste_lieux],
            dtype=np.float32
        )

    # ----------------------------------------------------------------------
    # Génération / Chargement
    # ----------------------------------------------------------------------
    def _generer_aleatoire(self, nb_lieux):
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(MARGE, LARGEUR - MARGE)
            y = random.uniform(MARGE, HAUTEUR - MARGE)
            self.liste_lieux.append(Lieu(x, y, nom=str(i)))

    def charger_graph(self, csv_path: str):
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

    # ----------------------------------------------------------------------
    # Matrice triangulaire compacte
    # ----------------------------------------------------------------------
    def calcul_matrice_cout_od(self):
        n = self.N

        xs = np.array([lieu.x for lieu in self.liste_lieux], dtype=np.float32)
        ys = np.array([lieu.y for lieu in self.liste_lieux], dtype=np.float32)

        I, J = np.triu_indices(n, k=1)

        dx = xs[I] - xs[J]
        dy = ys[I] - ys[J]
        dists = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        self.matrice_od = dists

        def index(i, j):
            if i == j:
                return None
            if j < i:
                i, j = j, i
            return i * n - (i * (i + 1)) // 2 + (j - i - 1)

        self._tri_index = index

    # ----------------------------------------------------------------------
    # Distance
    # ----------------------------------------------------------------------
    def distance_ij(self, i, j):
        """Distance entre i et j, avec fallback en mode calcul direct."""
        if i == j:
            return 0.0

        # ---------- MODE MATRICE (rapide) ----------
        if self.use_matrix:
            idx = self._tri_index(i, j)
            return float(self.matrice_od[idx])

        # ---------- MODE FALLBACK (distance à la demande) ----------
        dx = self.coords[i, 0] - self.coords[j, 0]
        dy = self.coords[i, 1] - self.coords[j, 1]
        return float(np.sqrt(dx*dx + dy*dy))

    # ----------------------------------------------------------------------
    # Fonctions TSP
    # ----------------------------------------------------------------------
    def plus_proche_voisin(self, i: int, visites: set) -> int:
        n = self.N
        best_j = None
        best_d = float("inf")
        for j in range(n):
            if j == i or j in visites:
                continue
            d = self.distance_ij(i, j)
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    def calcul_distance_route(self, route: Route) -> float:
        total = 0.0
        for a, b in zip(route.ordre[:-1], route.ordre[1:]):
            total += self.distance_ij(a, b)
        return float(total)
    

# =========================
# Classe Affichage (auto-update sur amélioration + CLI friendly)
# =========================
# =========================
# Classe Affichage (auto-update sur amélioration + CLI friendly)
# =========================
class Affichage:
    """
    Affichage pour TSP_GA :
    - N <= 500 : lieux + grille + meilleure route + routes secondaires (si P)
    - 500 < N <= 10000 : pas de points, grille + routes (limitées en segments)
    - N > 10000 : mode ultra simple :
        * pas de grille (fond blanc)
        * pas de points
        * uniquement la meilleure route, fortement sous-échantillonnée
    - Limite le nombre de segments tracés à MAX_SEGMENTS par route si N est grand
    - NE REDESSINE que lorsqu'une meilleure distance est trouvée
    """

    def __init__(self, graph: Graph, titre="TSP - Algorithme génétique (auto)"):
        self.graph = graph
        self.tsp_ga = None

        self.best_route: Route | None = None       # meilleure route affichée
        self.best_distance_affichee: float | None = None
        self.population: list[Route] = []

        self._show_population = False  # routes secondaires visibles ou non
        self._current_gen = 0          # itération courante
        self._best_gen = 0             # itération où la meilleure route actuelle a été trouvée

        # ----- Modes d'affichage selon N -----
        self.N = self.graph.N
        if self.N <= 500:
            self._mode = "full"       # points + grille + routes + population P
        elif self.N <= 10000:
            self._mode = "medium"     # pas de points, grille + routes
        else:
            self._mode = "ultra"      # ultra simplifié : pas de grille, pas de points, meilleure route simplifiée

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

        # label gauche : itération + distance
        self.lbl_status_left = tk.Label(
            status_frame,
            text="Itération : 0 / 0   |   Distance : -",
            bg="#e6e6e6",
            anchor="w"
        )
        self.lbl_status_left.pack(side="left", padx=10, fill="x", expand=True)

        # label droite : aide P
        self.lbl_status_right = tk.Label(
            status_frame,
            text="Appuyer sur P pour afficher top N" if self.N <= 10000 else "Mode ultra simple (N>10000)",
            bg="#e6e6e6",
            anchor="e"
        )
        self.lbl_status_right.pack(side="right", padx=10)

        # Raccourcis utiles
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        # En mode ultra, la touche P n'a plus d'effet
        if self._mode != "ultra":
            self.root.bind("<Key-p>", self._toggle_population)
            self.root.bind("<Key-P>", self._toggle_population)

        # Transform pour adapter les coordonnées au canvas
        self._transform = None
        self._compute_transform()

        # Premier dessin (fond uniquement, points seront redessinés avec les routes)
        self.redraw_base()

        # flag d'exécution auto
        self._auto_running = False
        self._delay_ms = 2  # délai entre itérations (ms)

    # --------------- Connexion avec le GA ---------------
    def set_ga(self, tsp_ga: "TSP_GA", start_auto: bool = True):
        """Attache le GA et (optionnel) lance la boucle auto."""
        self.tsp_ga = tsp_ga

        # reset compteur d'itérations
        self._current_gen = 0
        self._best_gen = 0

        # récupère population & best initiaux
        self.population = sorted(list(tsp_ga.population))
        self.best_route = self.population[0] if self.population else None
        self.best_distance_affichee = (
            self.best_route.distance if self.best_route and self.best_route.distance is not None else None
        )

        # affiche l'état initial
        self.redraw_base()
        self._draw_routes()            # d'abord les traits
        self._draw_points_if_needed()  # puis les points par-dessus (si mode le permet)
        self._update_status(gen=self._current_gen, nb_gen=tsp_ga.nb_generations)

        if start_auto and self.graph.N<=5000:
            self.start_auto()
        elif start_auto:
            time.sleep(5)
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

        # stop si on a déjà atteint le nb maximal de générations
        if self._current_gen >= self.tsp_ga.nb_generations:
            return

        # Une génération
        self.tsp_ga.step()  # met self.population à jour en interne
        self._current_gen += 1

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
            self._best_gen = self._current_gen  # nouvelle meilleure trouvée à cette itération

            # Redessine uniquement si on a mieux
            self.redraw_base()
            self._draw_routes()
            self._draw_points_if_needed()

        # Met à jour la barre (itération + distance + best_gen) à chaque step
        self._update_status(gen=self._current_gen, nb_gen=self.tsp_ga.nb_generations)

        # Replanifie la prochaine itération
        self.root.after(self._delay_ms, self._auto_loop)

    # --------------- Toggle population (touche P) ---------------
    def _toggle_population(self, _evt=None):
        # En mode ultra, la population n'est jamais affichée
        if self._mode == "ultra":
            return
        self._show_population = not self._show_population
        # on redessine simplement en prenant en compte le nouveau flag
        self.redraw_base()
        self._draw_routes()
        self._draw_points_if_needed()

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
        - full / medium : grille
        - ultra : fond blanc uniquement (pas de grille)
        """
        self.canvas.delete("all")
        if self._mode in ("full", "medium"):
            self._draw_background()
        else:
            # mode ultra : simple fond blanc (le bg du canvas suffit)
            pass

    def _draw_background(self):
        W = LARGEUR
        H = self.canvas_h
        step = 50
        for x in range(0, W, step):
            self.canvas.create_line(x, 0, x, H, fill="#f0f0f0")
        for y in range(0, H, step):
            self.canvas.create_line(0, y, W, y, fill="#f0f0f0")

    def _draw_points_if_needed(self):
        """
        Dessine les lieux uniquement si :
        - mode "full"
        - et N <= MAX_POINTS
        Pas de points en mode medium/ultra.
        """
        if self._mode == "full" and len(self.graph.liste_lieux) <= MAX_POINTS:
            self._draw_points()

    def _draw_points(self):
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
        """
        Trace une route.
        - N <= 10000 : on limite à MAX_SEGMENTS segments.
        - N > 10000 (mode ultra) :
            * sous-échantillonnage fort de la route ( ~1000 points max )
            * segments également limités.
        """
        if route is None or not route.ordre or len(route.ordre) < 2:
            return

        indices = route.ordre

        # --- Mode ultra : gros sous-échantillonnage ---
        if self._mode == "ultra":
            # On vise environ 1000 points max
            max_pts = 1000
            step = max(1, len(indices) // max_pts)
            if step > 1:
                # on garde 0, step, 2*step, ... + dernier
                indices = indices[::step]
                if indices[-1] != route.ordre[-1]:
                    indices = indices + [route.ordre[-1]]

        pts = []
        for idx in indices:
            lieu = self.graph.liste_lieux[idx]
            pts.append(self._to_canvas(lieu.x, lieu.y))

        nb_segments = len(pts) - 1
        if nb_segments <= 0:
            return

        # Limitation du nombre de segments
        if self._mode == "ultra":
            # on peut encore réduire un peu pour être sûr d'être léger
            max_segments = min(MAX_SEGMENTS, nb_segments)
        else:
            max_segments = min(MAX_SEGMENTS, nb_segments)

        for k in range(max_segments):
            (x1, y1) = pts[k]
            (x2, y2) = pts[k + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash)

    def _draw_routes(self):
        """
        Dessine les routes (traits) UNIQUEMENT.
        Les points seront dessinés ensuite par _draw_points_if_needed()
        pour rester au-dessus des traits.
        """
        if not self.population:
            return
        pop_sorted = self.population
        best = pop_sorted[0]
        others = pop_sorted[1:6]  # jusqu'à 5 suivantes

        # En mode ultra : uniquement la meilleure route, en bleu simple
        if self._mode == "ultra":
            self._draw_route(best, color="#0066cc", dash=None, width=1)
            return

        # modes full / medium : comportement d'origine
        if self._show_population:
            for r in others:
                self._draw_route(r, color="#c0c0c0", dash=(2, 4), width=1)

        # meilleure route toujours affichée
        self._draw_route(best, color="#0066cc", dash=(4, 3), width=2)

    # --------------- Status ---------------
    def _update_status(self, gen: int | None, nb_gen: int | None):
        if self.best_route is None or self.best_route.distance is None or gen is None or nb_gen is None:
            self.lbl_status_left.config(text="Itération : 0 / 0   |   Distance : -")
            return

        txt = (
            f"Itération : {gen} / {nb_gen}   |   Distance : {self.best_route.distance:.3f} "
            f"(meilleure trouvée à l’itération {self._best_gen})"
        )
        self.lbl_status_left.config(text=txt)

    # --------------- Boucle Tk ---------------
    def run(self):
        self.root.mainloop()


# =========================
# Exécution directe avec arguments CLI
# =========================
if __name__ == "__main__":
    import argparse
    from math import sqrt
    from tsp_ga import TSP_GA

    parser = argparse.ArgumentParser(description="Run TSP GA with optional CSV or number of cities")
    parser.add_argument("--csv", dest="csv_path", help="Path to CSV file with cities", default=None)
    parser.add_argument("-n", dest="nb_lieux", type=int, help="Number of cities to generate (ignored if --csv provided)",
                        default=200)
    args = parser.parse_args()

    # Build graph from CSV if provided, otherwise generate nb cities
    if args.csv_path:
        graph = Graph(csv_path=args.csv_path)
    else:
        graph = Graph(nb_lieux=args.nb_lieux)

    affichage = Affichage(graph, titre="UI")

    taille_pop = max(10, 2 * graph.N) if graph.N < 500 else int(5 * sqrt(graph.N)) + 900
    print(f"Initialisation GA avec taille_pop={taille_pop} pour N={graph.N}")
    tsp_ga = TSP_GA(
        graph=graph,
        affichage=affichage,
        taille_pop=taille_pop,
        taille_pop_enfants=int(taille_pop * 0.7),
        prob_mutation=0.2,
        nb_generations=10000
    )

    # Lancer
    affichage.set_ga(tsp_ga)
    affichage.run()