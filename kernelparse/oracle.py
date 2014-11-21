from gensim.utils import simple_preprocess as tokenize
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from itertools import islice, groupby
from operator import itemgetter
import sys
import os
import numpy as np
from random import Random
from lru import lru_cache
from analyse import analyse

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Experiment(object):
    def __init__(self, dataset_path, results_path):
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.random = Random(1)

    def get_examples(self, filename):
        full_path = os.path.join(self.dataset_path, filename)
        example_file = gzip.open(full_path)
        for row in example_file:
            yield json.loads(row)

    def test(self):
        test_path = join(self.dataset_path, 'examples-test.json.gz')
        examples = self.get_examples(test_path)

        source_groups = groupby(examples, itemgetter('source'))
        logger.info("Evaluation on test set")
        count = 0
        with open(self.results_path, 'w') as results_file:
            for source, group in source_groups:
                group = list(group)
                self.random.shuffle(group)

                scores = [item['score'] for item in group]
                best_index = np.argmax(scores)
                results_file.write(json.dumps(group[best_index]) + '\n')
                count += 1
                if count % 100 == 0:
                    logger.info("Processed %d items", count)
                    #logger.debug("Cache info: %r", preprocess.cache_info())

    def run_experiment(self):
        self.test()
        analyse(self.results_path)

if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/'
    results_path = 'oracle.json'

    experiment = Experiment(dataset_path, results_path)
    experiment.run_experiment()
    
