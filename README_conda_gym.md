
Ci-dessous:
* pour Mac et Linux, les commandes sont à faire dans un terminal classique. 
* Pour Windows, il faut utiliser **Anaconda prompt** et pas un terminal de commande classique (taper "Anaconda Prompt" dans la barre de recherche Windows). 

1. Activer l'environnement utilisé au tp précédent `tpdeep2026`:

```
conda activate tpdeep2026
```

A ce niveau, votre ligne de commande doit ressembler à : `(tpdeep2026) <User>: `. 

`(tpdeep2026)` indique que l'environnement créé est actif.

Vous devez maintenant installer le package *gymnasium* dans cet environnement.


2. Installation de gymnasium
-  Sur __Windows__:
```
pip install swig
```
Ensuite aller sur https://visualstudio.microsoft.com/visual-cpp-build-tools/
 -> cliquer sur" Télécharger Build tools", puis lancer l'installer installé. Lors du choix, sélectionner "Desktop Development with C++"
Une fois installé:
```
pip install gymnasium[box2d]
```

- Sur __Linux__: 
```
conda install conda-forge::gymnasium
conda install conda-forge::gymnasium-box2d
```
- Sur __Mac__:
```
conda install -c conda-forge gymnasium
conda install swig
conda install -c conda-forge gym-box2d
```


3. Vous pouvez maintenant  commencer à compléter le notebook `TPDQN.ipynb` pour faire votre TP, soit avec `jupyter-lab`, ou (conseillé) avec l'[extension Jupyter de VisualStudio](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) qui vous permet de debugger votre notebook.


## Sources

- si besoin, utiliser ([google colab](https://colab.research.google.com/?hl=fr)), version cloud de jupyter notebook qui  permet d'accéder gratuitement à des ressources informatiques, dont des GPU (limité).
- Un tutoriel sur les [Jupyter notebook](https://python.sdv.univ-paris-diderot.fr/18_jupyter/)
- Vous pouvez lister les environnements conda installés :
```
conda env list
```
- Vous pouvez lister les packages installés dans l'environnement actif :
```
conda list
```

