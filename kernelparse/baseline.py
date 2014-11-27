import json
from sklearn.cross_validation import KFold, cross_val_score
from itertools import islice, groupby
from operator import itemgetter
import sys
import os
from random import Random
from analyse import analyse

from kernelparse.experiment import Experiment
from kernelparse.log import logger
from kernelparse.tensor import TensorParser

def output_results(results, results_path):
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)

if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json'
    results_path = 'results.json'

    with open(dataset_path) as dataset_file:
        dataset = islice(dataset_file, 10000)
        dataset = [json.loads(row) for row in dataset if len(row.strip()) > 0]

    parser = TensorParser()
    experiment = Experiment()
    results = experiment.run(dataset, parser)
    output_results(results, results_path)
    print analyse(results)
    
