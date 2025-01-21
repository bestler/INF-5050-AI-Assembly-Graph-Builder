# ai-lecture-project

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).
If you wish, you can upgrade to a higher Python version. 

## Überlegungen zur Implementierung

### Implmentierung eines FFN (Feed Forward Network)
Eine der ersten KI-Ansätze war es ein Feed Forward Network zu bauen, das als Eingabe einen Vektor der Part_IDs hat und als Ausgabe eine Adjazenz-Matrix des resultierenden Graphen. Wenn man sich vorher etwas Gedanken über die Größenordnung gemacht hätte, hätte man direkt erkennen können, dass dieser Ansatz nicht zielführend ist, da die Matrix mit 1089x1089 deutlich zu groß ist und viel zu viel Speicher benötigt.
Darüber hinaus wurde mir währenddessen bewusst, dass eine reine Adjazenz-Matrix mit Mutilabel Classification auch schon allein deswegen nicht funktinoniert, da es Teile mehrfach geben kann (und deswegen man das Ergebnis sowieso noch weiterverarbieten müsste.)
Nach einem Experiement wurde, diese Erkenntnis in der Praxis erfahren, sodass ein neuer Ansatz gewählt werden musst. Anstatt zu versuchen den ganzen Graph direkt vorherzusagen, wird nun versucht für jede Knoten, die wahrscheinlichste Kante vorherzusagen.

### Stacking (Kanten-Vorhersage + Graph-Aufbau)



## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.
`conda create -n ai-project python=3.8`

2. Activate the environment:
`conda activate ai-project`

3. Install the dependencies:
`conda install -c conda-forge --file requirements.txt`


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html).
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This automatically searches all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.
