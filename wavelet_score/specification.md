# Specification mathematique et numerique

Ce document fixe les conventions de la reimplementation auditee de
*Where the Score Lives: A Wavelet View of Diffusion*. Toute implementation
doit respecter ces conventions ou documenter explicitement pourquoi elle y
deroge.

## 1. Perimetre initial

La premiere phase utilise :

- MNIST redimensionne en images grayscale de taille `32 x 32` ;
- une transformee en ondelettes 2D separable ;
- les ondelettes Haar pour les tests pedagogiques, puis db2 ;
- des conditions aux bords periodiques (`mode="periodization"`) ;
- une DWT critique, orthonormale et exactement inversible ;
- des calculs de reference en precision `float64` sur CPU.

Les images RGB, le Flow Matching et la generation sont hors du perimetre de
cette premiere phase.

## 2. Images et dimensions

On note :

- `N` : nombre d'images du batch ;
- `H`, `W` : hauteur et largeur ;
- `d = H * W` : dimension d'une image vectorisee ;
- `x_t` : une image a l'instant ou au niveau de bruit `t` ;
- `X_t` : un batch d'images vectorisees.

Les representations utilisees sont :

| Objet | Shape | Description |
|---|---:|---|
| `images_t` | `[N, H, W]` | Batch d'images 2D |
| `X_t` | `[N, d]` | Meme batch aplati ligne par ligne |
| `x_t` | `[d]` | Une image vectorisee |

Pour MNIST 32 x 32, `H = W = 32` et `d = 1024`.

L'ordre d'aplatissement des pixels est l'ordre C, donc l'indice du pixel
`(u1, u2)` est

$$
q = u_1 W + u_2.
$$

## 3. Operateur d'ondelettes

Soit

$$
B \in \mathbb{R}^{d\times d}
$$

la matrice conceptuelle dont chaque ligne est un atome de la base d'ondelettes
vectorise dans le meme ordre que l'image. La matrice ne sera pas necessairement
materialisee dans le code : PyWavelets applique implicitement cet operateur.

Pour une image representee comme vecteur colonne :

$$
y_t = Bx_t,
\qquad
x_t = B^\top y_t.
$$

Pour un batch stocke ligne par ligne :

$$
Y_t = X_t B^\top,
\qquad
X_t = Y_t B.
$$


Dans le cas orthonormal :

$$
B^\top B = BB^\top = I_d
$$

et Parseval donne

$$
\lVert x_t\rVert_2^2 = \lVert y_t\rVert_2^2.
$$

## 4. Convention des niveaux

Supposons `H = W = 2^J`. Pour une image 32 x 32, `J = 5`.

Dans la convention mathematique, `j` augmente des echelles grossieres vers les
echelles fines. Une approximation a l'echelle `J0` possede

$$
2^{J_0}\times 2^{J_0}
$$

coefficients d'echelle.

PyWavelets utilise une convention differente : `level=1` est la premiere
decomposition depuis les pixels et contient les details les plus fins. Pour une
DWT de profondeur `L`, la correspondance critique est

$$
j = J - \texttt{pywt\_level}.
$$

Avec une image 32 x 32 et `L = 3` :

| Sous-bande PyWavelets | Shape | Echelle mathematique |
|---|---:|---:|
| `cA3` | `[4, 4]` | `J0 = 2` |
| `cH3`, `cV3`, `cD3` | `[4, 4]` | `j = 2` |
| `cH2`, `cV2`, `cD2` | `[8, 8]` | `j = 3` |
| `cH1`, `cV1`, `cD1` | `[16, 16]` | `j = 4` |

La decomposition contient exactement

$$
4^2 + 3\left(4^2 + 8^2 + 16^2\right) = 1024
$$

coefficients.

Le papier annonce par ailleurs une base echantillonnee approximativement
orthonormale et une troncature des echelles pour limiter les effets de bord.
Cette implementation n'est pas identique a la DWT critique definie ici. Elle
sera etudiee separement sous le nom `paper_sampled_basis` apres validation de
la version `critical_dwt`.

## 5. Convention des orientations

Chaque sous-bande de detail possede une orientation parmi :

- `H` : sous-bande retournee comme `cH` par PyWavelets ;
- `V` : sous-bande retournee comme `cV` par PyWavelets ;
- `D` : sous-bande diagonale `cD`.

Les noms `H` et `V` de la bibliotheque ne suffisent pas pour conclure sur
l'orientation geometrique du bord detecte. Leur interpretation sera fixee par
des tests sur une image constante, un bord vertical et un bord horizontal.

## 6. Aplatissement des coefficients

PyWavelets retourne une structure imbriquee :

```text
[cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]
```

Notre vecteur aplati `y_t` suit obligatoirement cet ordre :

1. approximation `cA_L` ;
2. details du niveau `L`, dans l'ordre `H`, `V`, `D` ;
3. niveaux suivants du plus grossier au plus fin ;
4. chaque sous-bande est aplatie en ordre C.

Ainsi, pour MNIST 32 x 32 et `L = 3`, les intervalles sont :

| Intervalle d'indices | Sous-bande |
|---:|---|
| `[0, 16)` | `cA3` |
| `[16, 32)` | `cH3` |
| `[32, 48)` | `cV3` |
| `[48, 64)` | `cD3` |
| `[64, 128)` | `cH2` |
| `[128, 192)` | `cV2` |
| `[192, 256)` | `cD2` |
| `[256, 512)` | `cH1` |
| `[512, 768)` | `cV1` |
| `[768, 1024)` | `cD1` |

L'operation inverse doit reconstruire exactement la structure attendue par
PyWavelets sans deduire implicitement les niveaux depuis des offsets disperses
dans le code.

## 7. Metadonnees d'un coefficient

Chaque indice aplati `i` est associe a une metadonnee immuable contenant :

```text
flat_index
coefficient_type   # "scaling" ou "detail"
pywt_level         # L, ..., 1
math_scale_j       # j = J - pywt_level
orientation        # None, "H", "V" ou "D"
k1
k2
subband_height
subband_width
```

Pour un coefficient d'echelle, `pywt_level = L`, `math_scale_j = J0` et
`orientation = None`.

La correspondance entre `flat_index` et cette metadonnee doit etre construite
une seule fois par une fonction centralisee. Le flatten et l'unflatten doivent
utiliser le meme objet d'indexation.

## 8. Bruitage VE et coefficients

Les experiences initiales du papier utilisent le bruitage variance exploding

$$
x_t = x_0 + \sigma z,
\qquad
z\sim\mathcal{N}(0,I_d),
\qquad
\sigma\in\{1,2,3,4\}.
$$

Dans une base orthonormale :

$$
y_t = y_0 + \sigma\eta,
\qquad
\eta=Bz\sim\mathcal{N}(0,I_d).
$$

Cette propriete sera verifiee empiriquement globalement et par sous-bande.

## 9. Denoiseur et score

Une regression de `y_0` depuis des features de `y_t` produit un estimateur des
coefficients propres :

$$
\widehat y_0(y_t,\sigma).
$$

La reconstruction propre correspondante est

$$
\widehat x_0 = B^\top \widehat y_0.
$$

Sous la convention VE standard, la formule de Tweedie donne

$$
\widehat s(x_t,\sigma)
=
\frac{\widehat x_0(x_t,\sigma)-x_t}{\sigma^2}.
$$

Le code distinguera obligatoirement les fonctions de prediction du propre et
de prediction du score. Le signe exact utilise dans l'implementation du papier
sera audite avant de reproduire ses experiences.

## 10. Criteres de validation de la premiere etape

La transformee et son indexation ne seront considerees comme validees que si
les tests suivants passent :

1. exactement `H * W` coefficients pour des images 8 x 8 et 32 x 32 ;
2. round-trip `coeffs -> flatten -> unflatten -> coeffs` exact a la tolerance ;
3. reconstruction relative

   $$
   \frac{\lVert x-\widehat x\rVert_2}{\lVert x\rVert_2}
   < 10^{-10}
   $$

   en `float64`, et inferieure a `1e-6` en `float32` ;
4. erreur relative de Parseval inferieure aux memes tolerances ;
5. bijection entre les `H * W` indices et les `H * W` metadonnees ;
6. verification manuelle des premieres et dernieres metadonnees ;
7. orientation validee sur des images synthetiques ;
8. assertions explicites sur toutes les dimensions d'entree et de sortie.

## 11. Questions volontairement reportees

Les points suivants ne doivent pas etre resolus dans la premiere implementation :

- reproduction de la base echantillonnee surcomplete du papier ;
- choix des features independantes, band-coupled ou local-coupled ;
- regression ridge ;
- comparaison entre la regression contrainte et l'equation (23) dense ;
- conversion vers Flow Matching ;
- images RGB et interactions entre canaux ;
- generation.
