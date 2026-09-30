# Flow Matching empirique masque dans l'espace des ondelettes

Cette experience est distincte des regressions polynomiales existantes.
Pour le chemin `Y_t=(1-t)xi+tY_0`, elle estime, groupe par groupe,

$$
\widehat Y_{0,O_g}(y,t)=\sum_n w_{n,g}(y,t)Y^{(n)}_{0,O_g},
$$

avec

$$
w_{n,g}(y,t)\propto\exp\left(
-\frac{\|y_{I_g}-tY^{(n)}_{0,I_g}\|_2^2}{2(1-t)^2}
\right),
\qquad
\widehat v_{O_g}(y,t)=
\frac{\widehat Y_{0,O_g}(y,t)-y_{O_g}}{1-t}.
$$

`I_g` et `O_g` sont respectivement les indices observes et predits par un
`WaveletMaskGroup`. Chaque coefficient de sortie appartient exactement a un
groupe. Le centre est toujours observe.

Familles :

- `independent` : un coefficient observe et predit seul ;
- `band_tied` : les orientations H/V/D au meme niveau et a la meme position ;
- `local` : voisinage periodique dans une seule sous-bande ;
- `local_band_tied` : voisinage periodique et trois orientations partageant
  les memes poids.

L'approximation `cA` est independante pour `band_tied` et locale pour
`local_band_tied`. Aucune dependance entre echelles n'est introduite.

Le softmax est exact et accumule par chunks de references. A `t=0`, la moyenne
empirique est utilisee exactement. La formule n'est jamais evaluee a `t=1`.
