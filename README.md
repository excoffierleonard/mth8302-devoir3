# MTH8302 - Devoir 3: Régression Linéaire Multiple

Devoir 3 du cours MTH8302 (Hiver 2026) - Polytechnique Montréal.

## Structure

```txt
src/           Scripts Python
report/        Source LaTeX
output/        Sorties générées (figures, .txt)
lessons/       Diapositives du cours (référence)
data/          Le Dataset utilisé
run.sh         Script d'exécution complet
```

## Prérequis

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- LaTeX (`pdflatex`) avec les paquets standards (`amsmath`, `babel`, `listings`, etc.)

## Installation

```bash
git clone https://github.com/excoffierleonard/mth8302-devoir3.git
cd mth8302-devoir3
uv sync
```

## Exécution

Tout en une commande :

```bash
./run.sh
```

Cela exécute tous les scripts Python (génère figures et fichiers texte dans `output/`), puis compile le rapport LaTeX. Le PDF final est généré à la racine du projet sous `MTH8302_ExcoffierLeonard_2085276_Devoir3.pdf`.

## Visualisation

Le rapport compilé se trouve dans [`MTH8302_ExcoffierLeonard_2085276_Devoir3.pdf`](MTH8302_ExcoffierLeonard_2085276_Devoir3.pdf) à la racine du projet après exécution de `run.sh`.
