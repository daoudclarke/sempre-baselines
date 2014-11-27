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

def get_examples(self, filename):
    full_path = os.path.join(self.dataset_path, filename)
    example_file = gzip.open(full_path)
    for row in example_file:
        yield json.loads(row)

def output_results(results, results_path):
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)

if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json'
    results_path = 'results.json'

    with open(dataset_path) as dataset_file:
        #dataset = islice(dataset_file, 100000)
        dataset = [json.loads(row) for row in dataset_file if len(row.strip()) > 0]

    experiment = Experiment(dataset)
    experiment.train()
    results = experiment.test()
    output_results(results, results_path)
    print analyse(results)
    
