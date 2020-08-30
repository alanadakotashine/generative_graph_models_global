# generative_graph_models_global
Generative graph models subject to global similarity code base with accompanying thesis

To run netgan:
mkdir results 
python netgan/demo_l16.py 3 walk 1 results 64 40 30 model_best football

To run cut fix generation with random walk generation as input:
mkdir results
cd graph_cuts
python3 graph_cuts_demo.py football ../results/ std_stats.json football_params.json num_walk_algs.json 0

To run spectral generation:
see intructions in spectral_generation/README.md
