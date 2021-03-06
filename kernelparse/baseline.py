"""
Script to run experiments on semantic parsing paraphrase datasets.
"""

import json
from analyse import analyse

from kernelparse.experiment import Experiment
from kernelparse.tensor import TensorParser
from kernelparse.oracle import OracleParser
from kernelparse.randomparser import RandomParser
from kernelparse.bigramtensor import BigramTensorParser
from kernelparse.log import logger

PARSERS = {repr(parser): parser for parser in 
           [TensorParser(), BigramTensorParser(), OracleParser(), RandomParser()] }

def output_results(results, results_path):
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)


def run(dataset_path, results_path, parser_names, limit):
    parsers = [PARSERS[name] for name in parser_names]

    logger.info("Starting experiment run - reading dataset")
    with open(dataset_path) as dataset_file:
        dataset = [json.loads(row) for row in dataset_file if len(row.strip()) > 0]

    experiment = Experiment(limit)
    
    logger.info("Running experiment")
    results = experiment.run(dataset, parsers)
    output_results(results, results_path)

    logger.info("Experiment complete")
    print analyse(results)
    

if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument('dataset_path', type=str, help='Path to .json file containing the dataset')
    argument_parser.add_argument('results_path', type=str, help='Path in which to save the results in .json format')
    argument_parser.add_argument('parser_names', type=str, nargs='+', help='Which parser. Valid parsers are %s' % ', '.join(PARSERS))
    argument_parser.add_argument('--limit', type=int, default=None, help="Limit the number of training examples to this value.")
    
    args = argument_parser.parse_args()
    logger.debug("Requested arguments: %r", vars(args))
    run(**vars(args))
