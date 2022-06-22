#!/bin/sh

python3 plot_graph.py --trial 6
python3 main.py -l 1 --trial 6 --seed 0
python3 main.py -l 1 --trial 6 --seed 1
python3 main.py -l 1 --trial 6 --seed 2
python3 main.py -l 1 --trial 6 --seed 3
python3 main.py -l 1 --trial 6 --seed 4