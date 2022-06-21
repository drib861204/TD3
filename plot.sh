#!/bin/sh

python3 plot_graph.py --trial 3
python3 main.py -l 1 --trial 3 --seed 0
python3 main.py -l 1 --trial 3 --seed 1
python3 main.py -l 1 --trial 3 --seed 2
python3 main.py -l 1 --trial 3 --seed 3
python3 main.py -l 1 --trial 3 --seed 4

python3 plot_graph.py --trial 4
python3 main.py -l 1 --trial 4 --seed 0
python3 main.py -l 1 --trial 4 --seed 1
python3 main.py -l 1 --trial 4 --seed 2
python3 main.py -l 1 --trial 4 --seed 3
python3 main.py -l 1 --trial 4 --seed 4

python3 plot_graph.py --trial 5
python3 main.py -l 1 --trial 5 --seed 0
python3 main.py -l 1 --trial 5 --seed 1
python3 main.py -l 1 --trial 5 --seed 2
python3 main.py -l 1 --trial 5 --seed 3
python3 main.py -l 1 --trial 5 --seed 4
