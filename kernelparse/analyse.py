import json
from numpy import mean
from scipy.stats import tsem as stderr


def analyse(filepath):
    results_json = open(filepath)
    scores = []
    for result_json in results_json:
        result = json.loads(result_json)
        scores.append(result['score'])
    print "Mean score: %f +/ %f" % (
        mean(scores),
        stderr(scores),
        )
        



if __name__ == "__main__":
    analyse('results.json')
