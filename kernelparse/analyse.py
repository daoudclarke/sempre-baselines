import json
from numpy import mean
from scipy.stats import tsem as stderr


def analyse(results):
    scores = []
    for result in results:
        scores.append(result['score'])
    return  mean(scores), stderr(scores)
