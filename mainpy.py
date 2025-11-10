import numpy as np
import random
import time
import pandas as pd
import tkinter as tk

class Route:
    """Représente un ordre de visite, avec contrainte 0 ... 0 (départ/arrivée)."""
    def __init__(self, ordre):
        self.ordre = list(ordre)
    
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
        

    def __repr__(self):
        return f"Route(ordre={self.ordre})"


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