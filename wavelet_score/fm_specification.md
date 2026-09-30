# Specification Flow Matching dans la DWT critique

Ce document complete `specification.md` sans modifier les conventions de
transformee, d'aplatissement ou d'indexation deja validees.

## 1. Convention temporelle

Le chemin Flow Matching standard est

$$
X_t=(1-t)\varepsilon+tX_0,\qquad \varepsilon\sim\mathcal N(0,I),
\qquad t\in[0,1].
$$

Ainsi, `t=0` correspond au bruit et `t=1` aux donnees. Sa velocite
conditionnelle est constante le long du chemin :

$$
V^X=\partial_tX_t=X_0-\varepsilon.
$$

Pour un batch de `N` images `H x W`, `X_0`, `epsilon`, `X_t` et `V_X` sont
stockes sous forme `[N,H,W]` dans le code. La dimension vectorisee vaut
`d=H*W`.

## 2. Chemin dans la base d'ondelettes

Pour la DWT critique orthonormale `W`, on note

$$
Y_0=\mathcal W(X_0),\qquad \xi=\mathcal W(\varepsilon).
$$

Par linearite,

$$
Y_t=\mathcal W(X_t)=(1-t)\xi+tY_0,
$$

et la cible de velocite devient

$$
V^Y=Y_0-\xi=\mathcal W(V^X).
$$

Les objets `Y_0`, `xi`, `Y_t` et `V_Y` ont la shape `[N,d]`. La matrice de
la transformee n'est jamais materialisee : analyse et synthese utilisent
`WaveletTransform2D` et le `WaveletLayout` existants.

## 3. Objet appris

Le premier modele apprend directement `V_Y` depuis `Y_t`, a un instant `t`
fixe. Il ne predit ni un score, ni une image propre :

$$
\widehat V_i^Y(Y_t,t)=u_{t,i}^{\top}
[1,Y_{t,i},\ldots,Y_{t,i}^{D}]^{\top}.
$$

Chaque sortie `i` depend uniquement de l'entree `i`. Les poids ont la shape
`[d,D+1]` et sont obtenus par les petites equations normales ridge
independantes. Une feature map globale suivie d'une matrice dense appartient a
une autre famille, nommee `global_additive_dense`, hors de ce jalon.

La velocite pixel est reconstruite par

$$
\widehat V^X=\mathcal W^{-1}(\widehat V^Y).
$$

Aucun clipping, aucune formule de Tweedie et aucune conversion depuis un score
ne sont appliques a une velocite.

## 4. Distinction des taches

- denoiseur : `Y_t -> Y_0` ;
- score : gradient du log de la densite bruitee ;
- velocite FM : `Y_t -> V_Y`.

Ces trois objets peuvent reutiliser un meme solveur ridge generique, mais leurs
wrappers, cibles et methodes de reconstruction doivent rester explicitement
distincts.
