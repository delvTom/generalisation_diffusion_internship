# Masque Wavelet Coarse-to-Fine

## Objectif

Cette expérience teste si une closed-form empirique de Flow Matching peut
approcher la vitesse d'un U-Net préentraîné en utilisant un masque multiscale
simple et interprétable. Elle est réalisée sur CIFAR-10 avec

```text
x_t = (1-t) epsilon + t x_1,
v = x_1 - epsilon,
```

où `t=0` correspond au bruit et `t=1` aux données. Une DWT `db2`, niveau 3,
mode `periodization`, est appliquée séparément aux canaux RGB.

Les sous-bandes sont regroupées en trois régions :

```text
C = {A3, H3, V3, D3}   grossier
M = {H2, V2, D2}       intermédiaire
F = {H1, V1, D1}       fin
```

## Construction du masque

Pour prédire un site wavelet `q`, le masque sélectionne des sites dans les
sous-bandes autorisées. Chaque site sélectionné donne accès à ses trois
coefficients RGB, et `q` est toujours inclus.

- `C(q)` : voisinage périodique `3x3` dans chacune des bandes `H3`, `V3` et
  `D3`, plus le site spatialement aligné dans `A3` ;
- `M(q)` : voisinage périodique `3x3` dans chacune des bandes `H2`, `V2` et
  `D2` ;
- `F(q)` : voisinage périodique `3x3` dans chacune des bandes `H1`, `V1` et
  `D1`.

La fenêtre se déplace progressivement vers les hautes fréquences :

| Région de `q` | `0 <= t < 1/3` | `1/3 <= t < 2/3` | `2/3 <= t <= 1` |
|---|---|---|---|
| `q in C` | `q + C(q)` | `q + C(q)` | `q + C(q)` |
| `q in M` | `q + C(q)` | `q + C(q) + M(q)` | `q + M(q)` |
| `q in F` | `q + C(q)` | `q + C(q) + M(q)` | `q + M(q) + F(q)` |

Contrairement à un masque cumulatif, les bandes grossières sont donc retirées
aux temps tardifs pour les sorties intermédiaires et fines.

## Cardinalités exactes

Dans les colonnes `Hj/Vj/Dj`, le nombre indiqué vaut pour **chacune** des trois
orientations. Un voisinage `3x3` contient le centre et ses huit voisins.

| Région de `q` | Régime | `A3` | `H3/V3/D3` | `H2/V2/D2` | `H1/V1/D1` | Sites uniques | Entrées RGB |
|---|---|---:|---:|---:|---:|---:|---:|
| `C` | tous les `t` | 1 | 9 chacune | 0 | 0 | **28** | **84** |
| `M` | `t < 1/3` | 1 | 9 chacune | `q` seul | 0 | **29** | **87** |
| `M` | `1/3 <= t < 2/3` | 1 | 9 chacune | 9 chacune | 0 | **55** | **165** |
| `M` | `t >= 2/3` | 0 | 0 | 9 chacune | 0 | **27** | **81** |
| `F` | `t < 1/3` | 1 | 9 chacune | 0 | `q` seul | **29** | **87** |
| `F` | `1/3 <= t < 2/3` | 1 | 9 chacune | 9 chacune | `q` seul | **56** | **168** |
| `F` | `t >= 2/3` | 0 | 0 | 9 chacune | 9 chacune | **54** | **162** |

Ici, `q seul` signifie un site dans la sous-bande contenant `q` et aucun dans
les deux autres orientations de la même échelle. Les totaux comptent les sites
uniques : lorsque `q` appartient déjà à un voisinage `3x3`, il n'est pas compté
deux fois.

## Comparaisons et sorties

La baseline est comparée au U-Net FM, au masque adaptatif same-band (`Mask A`)
et au meilleur estimateur affine de Wiener. Le script sauvegarde :

- les MSE et similarités cosinus globales et par région `C/M/F` ;
- les générations avec le même bruit initial ;
- les trajectoires autour de `t=1/3` et `t=2/3` ;
- l'énergie wavelet par région et les plus proches voisins d'entraînement.

Lancement sur GPU :

```bash
python3 -m wavelet_coarse_to_fine.run_cifar10_baseline
```

Les résultats sont écrits dans
`wavelet_coarse_to_fine/results/cifar10_baseline/`.
