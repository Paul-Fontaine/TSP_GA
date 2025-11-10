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

    def __len__(self):
        return len(self.ordre)

    def __getitem__(self, index):
        return self.ordre[index]
    
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

    def reordonner(self):
        """Réorganise la route pour commencer et finir à 0."""
        if 0 not in self.ordre:
            raise ValueError("La route doit contenir le lieu de départ/arrivée (0).")
        idx_zero = self.ordre.index(0)
        nouvelle_ordre = self.ordre[idx_zero:] + self.ordre[1:idx_zero + 1]
        self.ordre = nouvelle_ordre
        

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
ZONE_TEXTE_H = 35   # hauteur fixe en pixels de la barre du bas
MARGE_CANVAS = 28   # marge interne autour du nuage de points (en px)
COULEUR_BG = "#fafafa"
COULEUR_GRILLE = "#eeeeee"

class Affichage:
    """
    Canvas dimensionné à (LARGEUR, HAUTEUR - ZONE_TEXTE_H) + barre de log fixe.
    - Toolbar en haut (Charger CSV / Aléatoire / Exécuter / Effacer)
    - Zoom-to-fit : on mappe automatiquement les coordonnées dans le canvas
      pour que rien ne soit caché par la barre (ni rogné).
    - set_run_callback(fn): fn(graph) -> Route | (Route, population:list[Route])
    """
    def __init__(self, graph: Graph, nom_groupe=NOM_GROUPE):
        import tkinter.filedialog as fd  # stdlib (autorisé via tkinter)
        self.fd = fd

        self.graph = graph
        self.root = tk.Tk()
        self.root.title(f"TSP - {nom_groupe}")

        # Hauteur utile du canvas (au-dessus de la barre)
        self.canvas_h = max(100, HAUTEUR - ZONE_TEXTE_H)

        # ---------- Toolbar (haut) ----------
        topbar = tk.Frame(self.root, bg="#f2f2f2")
        topbar.pack(side="top", fill="x")
        self.btn_csv = tk.Button(topbar, text="Charger CSV…", command=self._ui_charger_csv)
        self.btn_csv.pack(side="left", padx=6, pady=6)
        self.btn_rand = tk.Button(topbar, text="Aléatoire", command=self._ui_regenerer_aleatoire)
        self.btn_rand.pack(side="left", padx=6, pady=6)
        self.btn_run = tk.Button(topbar, text="Exécuter", command=self._ui_executer)
        self.btn_run.pack(side="left", padx=6, pady=6)
        self.btn_clear = tk.Button(topbar, text="Effacer route", command=self._ui_effacer)
        self.btn_clear.pack(side="left", padx=6, pady=6)

        self.lbl_info = tk.Label(topbar, text="Prêt.", bg="#f2f2f2")
        self.lbl_info.pack(side="right", padx=8)

        # ---------- Zone graphique (milieu) ----------
        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=self.canvas_h,
                                bg=COULEUR_BG, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=False)

        # ---------- Zone de texte (bas) en hauteur *pixels* fixée ----------
        text_frame = tk.Frame(self.root, height=ZONE_TEXTE_H)
        text_frame.pack(side="bottom", fill="x")
        text_frame.pack_propagate(False)
        self.text = tk.Text(text_frame, bd=0)
        self.text.pack(fill="both", expand=True)

        # états
        self._route = None
        self._population = []
        self._show_population = False
        self._show_matrix = False
        self._run_callback = None  # à brancher par set_run_callback

        # raccourcis
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Key-p>", self._toggle_population)
        self.root.bind("<Key-P>", self._toggle_population)
        self.root.bind("<Key-m>", self._toggle_matrix)
        self.root.bind("<Key-M>", self._toggle_matrix)

        # transform (zoom-to-fit) recalculée à chaque redraw
        self._transform = None  # (scale, offset_x, offset_y)

        self.redraw()

    # ===== API externe =====
    def set_route(self, route: Route):
        self._route = route
        self._log(f"Route mise à jour : {route.ordre} | distance = {self.graph.calcul_distance_route(route):.2f}")
        self.lbl_info.config(text=f"Distance: {self.graph.calcul_distance_route(route):.2f}")
        self.redraw()

    def set_population(self, routes: list):
        self._population = list(routes) if routes else []
        self._log(f"Population mise à jour : {len(self._population)} routes")
        self.redraw()

    def set_run_callback(self, fn):
        """fn(graph) -> Route | (Route, population:list[Route])"""
        self._run_callback = fn

    # ===== UI handlers =====
    def _ui_charger_csv(self):
        path = self.fd.askopenfilename(
            title="Choisir un CSV (nom,x,y ou x,y)",
            filetypes=[("CSV", "*.csv"), ("Tous", "*.*")]
        )
        if not path:
            return
        try:
            g = Graph(csv_path=path)
            self.graph = g
            self._route = None
            self._population = []
            self.lbl_info.config(text=f"CSV chargé: {path.split('/')[-1]}")
            self._log(f"Graphe chargé depuis CSV: {path}")
            self.redraw()
        except Exception as e:
            self._log(f"[ERREUR] {e}")
            self.lbl_info.config(text="Erreur CSV")

    def _ui_regenerer_aleatoire(self):
        self.graph = Graph(nb_lieux=len(self.graph.liste_lieux))
        self._route = None
        self._population = []
        self.lbl_info.config(text="Graphe aléatoire régénéré")
        self._log("Graphe aléatoire régénéré.")
        self.redraw()

    def _ui_executer(self):
        if not self._run_callback:
            self._log("Aucun callback d'exécution défini (utilise set_run_callback).")
            self.lbl_info.config(text="Pas d'algo")
            return
        res = self._run_callback(self.graph)
        if res is None:
            return
        if isinstance(res, tuple) and len(res) == 2:
            route, pop = res
            self.set_population(pop)
            self.set_route(route)
        else:
            self.set_route(res)

    def _ui_effacer(self):
        self._route = None
        self._population = []
        self.lbl_info.config(text="Route effacée")
        self.redraw()

    # ===== logging / toggles =====
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

    # ===== géométrie & dessin =====
    def _compute_transform(self):
        """Calcule (scale, offx, offy) pour zoom-to-fit avec marge."""
        xs = [lieu.x for lieu in self.graph.liste_lieux] or [0]
        ys = [lieu.y for lieu in self.graph.liste_lieux] or [0]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)

        W = max(2*MARGE_CANVAS + 2*RAYON_LIEU, self.canvas.winfo_width() or LARGEUR)
        H = max(2*MARGE_CANVAS + 2*RAYON_LIEU, self.canvas_h)

        # scale uniforme pour garder les proportions
        scale = min(
            (W - 2*MARGE_CANVAS - 2*RAYON_LIEU) / w,
            (H - 2*MARGE_CANVAS - 2*RAYON_LIEU) / h
        )

        offx = MARGE_CANVAS + RAYON_LIEU - xmin*scale + (W - 2*MARGE_CANVAS - 2*RAYON_LIEU - w*scale)/2
        offy = MARGE_CANVAS + RAYON_LIEU - ymin*scale + (H - 2*MARGE_CANVAS - 2*RAYON_LIEU - h*scale)/2
        self._transform = (scale, offx, offy)

    def _map_to_canvas(self, x, y):
        if self._transform is None:
            self._compute_transform()
        s, ox, oy = self._transform
        X = x*s + ox
        Y = y*s + oy
        # garde une petite marge pour les disques
        X = max(RAYON_LIEU+2, min((self.canvas.winfo_width() or LARGEUR) - RAYON_LIEU-2, X))
        Y = max(RAYON_LIEU+2, min(self.canvas_h - RAYON_LIEU-2, Y))
        return X, Y

    def _draw_background(self):
        # grille légère
        self.canvas.delete("bg-grid")
        step = 50
        W = self.canvas.winfo_width() or LARGEUR
        H = self.canvas_h
        for x in range(0, W, step):
            self.canvas.create_line(x, 0, x, H, fill=COULEUR_GRILLE, tags="bg-grid")
        for y in range(0, H, step):
            self.canvas.create_line(0, y, W, y, fill=COULEUR_GRILLE, tags="bg-grid")

    def _draw_lieu(self, x, y, label, order_idx=None):
        x, y = self._map_to_canvas(x, y)
        r = RAYON_LIEU
        fill = "#ff5a5a" if str(label) == "0" else "#f0f0f0"
        outline = "#222222" if str(label) == "0" else "#555555"
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
        self.canvas.create_text(x, y, text=str(label), font=("Arial", 10, "bold"))
        if order_idx is not None:
            self.canvas.create_text(x, y - r - 10, text=str(order_idx), font=("Arial", 9), fill="#333333")

    def _draw_route(self, route: Route, dash=(6, 4), color="blue", width=2):
        if not route or not route.ordre or len(route.ordre) < 2:
            return
        pts = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            pts.append(self._map_to_canvas(lieu.x, lieu.y))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            self.canvas.create_line(x1, y1, x2, y2, fill=color, dash=dash, width=width)

    def redraw(self):
        self.canvas.delete("all")
        self._compute_transform()
        self._draw_background()

        # Population en fond
        if self._show_population and self._population:
            for r in self._population:
                self._draw_route(r, dash=(), color="#cfcfcf", width=1)

        # Route principale
        if self._route:
            self._draw_route(self._route, dash=(6, 4), color="#1e70ff", width=2)
            ordre_index = {idx: k for k, idx in enumerate(self._route.ordre)}
        else:
            ordre_index = {}

        # Lieux
        for i, lieu in enumerate(self.graph.liste_lieux):
            self._draw_lieu(lieu.x, lieu.y, i, order_idx=ordre_index.get(i, None))

    def run(self):
        # recalcule le zoom quand la fenêtre change
        def _on_resize(_evt):
            self.redraw()
        self.canvas.bind("<Configure>", _on_resize)
        self.root.mainloop()
        


# =========================
# Exécution directe (démo sans route)
# =========================
if __name__ == "__main__":
    g = Graph(csv_path="fichiers_csv_exemples/graph_20.csv",seed=2)   # <-- Chargement du csv
    ui = Affichage(g, nom_groupe="Groupe_10")  # <-- Affichage
    ui.run()
