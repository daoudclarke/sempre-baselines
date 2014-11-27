from numpy import mean
from scipy.stats import tsem as stderr


def analyse(all_results):
    analysis = {}
    for parser_name, results in all_results.items():
        scores = []
        for result in results:
            scores.append(result['score'])
        analysis[parser_name] = (mean(scores), stderr(scores))
    return analysis
