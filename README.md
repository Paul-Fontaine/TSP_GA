# Résolution du problème du voyageur de commerce avec un algorithme génétique

## Paramètres de l'algorithme
- nombre de lieux
- taille de la population
- taille de la population enfant (nombre d'enfants générés)
- probabilité de mutation
- nombre de générations

## Population initiale
La moitié est remplie par des routes générées avec l'heuristique du plus proche voisin.  
Les éventuels doublons sont supprimés et le reste de la population est remplie avec des routes aléatoires pour ajouter de la diversité.

## Reproduction
Deux parents sont choisis avec un tournoi de taille 2. 
Un enfant est créé avec un croisement OX pour préserver l'ordre, 
cet enfant a une probabilité de subir une mutation qui échange deux lieux.

## Sélection
La sélection se fait sur la population des parents et la population des enfants. 
Les doublons sont supprimés puis la population totale est triée.
Le top 5% est conservé (élitisme) et le reste de la population est selectionné 
avec des tournois de taille 2 sans remise du gagnant. 
Un tournoi de taille 2 permet d'avoir une pression selective assez faible pour garder plus de diversité génétique.

