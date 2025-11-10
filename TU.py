# test_tsp_graph_init.py
# Tests unitaires pour Lieu, Route, Graph
# Dépendances: unittest (stdlib), csv (autorisé), numpy (autorisé)
import os
import csv
import unittest
import math
import numpy as np

# importe les classes et constantes depuis ton module
from tsp_graph_init import (
    Lieu, Route, Graph,
    LARGEUR, HAUTEUR, MARGE
)

class TestLieu(unittest.TestCase):
    def test_distance_zero(self):
        a = Lieu(10, 20, "A")
        self.assertEqual(a.distance(a), 0.0)

    def test_distance_3_4_5(self):
        a = Lieu(0, 0, "A")
        b = Lieu(3, 4, "B")
        self.assertAlmostEqual(a.distance(b), 5.0, places=7)
        self.assertAlmostEqual(b.distance(a), 5.0, places=7)  # symétrie

    def test_repr_format(self):
        a = Lieu(1, 2, "A")
        s = repr(a)
        self.assertIn("Lieu(nom=A", s)
        self.assertIn("x=1.0", s)
        self.assertIn("y=2.0", s)


class TestRoute(unittest.TestCase):
    def test_route_container(self):
        r = Route([0, 2, 1, 0])
        self.assertEqual(r.ordre, [0, 2, 1, 0])
        self.assertIn("Route(ordre=", repr(r))


class TestGraphGeneration(unittest.TestCase):
    def test_generation_aleatoire_nombre_et_bornes(self):
        g = Graph(nb_lieux=25, csv_path=None, seed=123)
        self.assertEqual(len(g.liste_lieux), 25)
        # vérifie que tous les points respectent les marges et dimensions
        for p in g.liste_lieux:
            self.assertGreaterEqual(p.x, MARGE - 1e-9)
            self.assertLessEqual(p.x, LARGEUR - MARGE + 1e-9)
            self.assertGreaterEqual(p.y, MARGE - 1e-9)
            self.assertLessEqual(p.y, HAUTEUR - MARGE + 1e-9)

    def test_matrice_distances_symetrie_et_diagonale(self):
        g = Graph(nb_lieux=12, csv_path=None, seed=7)
        mat = g.matrice_od
        self.assertEqual(mat.shape, (12, 12))
        # diagonale nulle
        self.assertTrue(np.allclose(np.diag(mat), 0.0))
        # symétrie
        self.assertTrue(np.allclose(mat, mat.T, atol=1e-9))
        # cohérence avec distance directe pour quelques paires
        for (i, j) in [(0, 1), (3, 7), (10, 5)]:
            self.assertAlmostEqual(
                mat[i, j],
                g.liste_lieux[i].distance(g.liste_lieux[j]),
                places=7,
            )


class TestGraphCSV(unittest.TestCase):
    CSV_PATH = "fichiers_csv_exemples/graph_5.csv"

    def tearDown(self):
        # nettoie le fichier créé si présent
        if os.path.exists(self.CSV_PATH):
            os.remove(self.CSV_PATH)

    def _write_csv(self, rows, header=None):
        with open(self.CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            for r in rows:
                w.writerow(r)

    def test_chargement_csv_sans_nom_headers_xy(self):
        # CSV avec en-tête x,y seulement => nom = index de ligne
        self._write_csv(
            rows=[
                [100.0, 100.0],
                [200.0, 250.0],
                [300.0, 400.0],
            ],
            header=["x", "y"],
        )
        g = Graph(csv_path=self.CSV_PATH)
        self.assertEqual(len(g.liste_lieux), 3)
        self.assertEqual(g.liste_lieux[0].nom, "0")
        self.assertEqual(g.liste_lieux[1].nom, "1")

        # distances connues
        d01 = g.liste_lieux[0].distance(g.liste_lieux[1])
        self.assertAlmostEqual(d01, math.hypot(100.0, 150.0), places=7)

        # matrice cohérente
        self.assertTrue(np.allclose(np.diag(g.matrice_od), 0.0))
        self.assertTrue(np.allclose(g.matrice_od, g.matrice_od.T, atol=1e-9))

    def test_chargement_csv_avec_nom(self):
        # CSV avec nom,x,y
        self._write_csv(
            rows=[
                ["A", 10, 10],
                ["B", 13, 14],  # dist 5 avec A
            ],
            header=["nom", "x", "y"],
        )
        g = Graph(csv_path=self.CSV_PATH)
        self.assertEqual([p.nom for p in g.liste_lieux], ["A", "B"])
        self.assertAlmostEqual(g.matrice_od[0, 1], 5.0, places=7)


class TestFonctionsGraph(unittest.TestCase):
    def setUp(self):
        # petit graphe déterministe à la main
        # 0 --(1)--> 1 --(2)--> 2 --(3)--> 3 --(4)--> 0 (distances ne sont pas linéaires mais on force la géométrie)
        self.csv_path = "test_small.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["nom", "x", "y"])
            w.writerow([0, 0.0, 0.0])   # 0
            w.writerow([1, 1.0, 0.0])   # 1  dist 1 avec 0
            w.writerow([2, 1.0, 2.0])   # 2  dist 2 avec 1
            w.writerow([3, 4.0, 2.0])   # 3  dist 3 avec 2 ; dist 4 avec 0 (3-0,2-0) -> hypot(4,2)=~4.472.. pas 4 pile
        self.g = Graph(csv_path=self.csv_path)

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_plus_proche_voisin(self):
        # depuis 0, le plus proche est 1
        j = self.g.plus_proche_voisin(0, visites={0})
        self.assertEqual(j, 1)
        # si 1 déjà visité, depuis 0 le suivant le plus proche devrait être 2 ou 3 selon distances
        j2 = self.g.plus_proche_voisin(0, visites={0, 1})
        # calcule manuellement la plus petite distance vers {2,3}
        d02 = self.g.matrice_od[0, 2]
        d03 = self.g.matrice_od[0, 3]
        attendu = 2 if d02 < d03 else 3
        self.assertEqual(j2, attendu)

    def test_calcul_distance_route(self):
        # route 0->1->2->3->0
        r = Route([0, 1, 2, 3, 0])
        d = self.g.calcul_distance_route(r)

        # somme des segments depuis la matrice
        m = self.g.matrice_od
        attendu = m[0, 1] + m[1, 2] + m[2, 3] + m[3, 0]
        self.assertAlmostEqual(d, attendu, places=7)

    def test_matrice_coherente_avec_lieux(self):
        # vérifie pour plusieurs paires au hasard
        pairs = [(0,1), (0,2), (1,3), (2,0), (3,1)]
        for i, j in pairs:
            self.assertAlmostEqual(
                self.g.matrice_od[i, j],
                self.g.liste_lieux[i].distance(self.g.liste_lieux[j]),
                places=7
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
