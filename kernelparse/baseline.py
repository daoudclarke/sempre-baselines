"""
Script to run experiments on semantic parsing paraphrase datasets.
"""

import json
from itertools import islice
from analyse import analyse

from kernelparse.experiment import Experiment
from kernelparse.tensor import TensorParser
from kernelparse.oracle import OracleParser

PARSERS = {repr(parser): parser for parser in 
           [TensorParser(), OracleParser()] }

def output_results(results, results_path):
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)


def run(dataset_path, results_path, parser_names):
    with open(dataset_path) as dataset_file:
        dataset = islice(dataset_file, 10000)
        dataset = [json.loads(row) for row in dataset if len(row.strip()) > 0]

    experiment = Experiment()
    
    parsers = [PARSERS[name] for name in parser_names]
    results = experiment.run(dataset, parsers)
    output_results(results, results_path)
    print analyse(results)
    

if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument('dataset_path', type=str, help='Path to .json file containing the dataset')
    argument_parser.add_argument('results_path', type=str, help='Path in which to save the results in .json format')
    argument_parser.add_argument('parser_names', type=str, nargs='+', help='Which parser. Valid parsers are %s' % ', '.join(PARSERS))

    #dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json'
    #results_path = 'results/results.json'

    args = argument_parser.parse_args()
    run(**vars(args))
