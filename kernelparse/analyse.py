from numpy import mean
from scipy.stats import tsem as stderr
import json

def analyse(all_results):
    analysis = {}
    for parser_name, results in all_results.items():
        scores = []
        for result in results:
            scores.append(result['score'])
        analysis[parser_name] = (mean(scores), stderr(scores))
    return analysis

def analyse_results_file(results_path):
    with open(results_path) as results_file:
        results = json.load(results_file)
        analysis = analyse(results)
        print analysis

if __name__ == "__main__":
    import sys
    analyse_results_file(sys.argv[1])
