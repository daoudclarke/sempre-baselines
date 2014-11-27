from itertools import groupby
from operator import itemgetter
from random import Random

from kernelparse.log import logger


class Experiment(object):
    """
    Class to run experiments on Sempre paraphrase datasets.
    """

    def __init__(self, train_limit=None):
        self.train_limit = train_limit

    def run(self, dataset, parsers):
        logger.info("Loading dataset")
        random = Random(1)        

        dataset = [(source, list(group)) for source, group in
                    groupby(dataset, itemgetter('source'))]

        if len(dataset) != len(set(map(itemgetter(0), dataset))):
            raise ValueError("Dataset contains non-contiguous source examples")

        # Shuffle the dataset and each group within the dataset
        random.shuffle(dataset)
        for source, group in dataset:
            random.shuffle(group)

        length = len(dataset)
        assert length > 1

        train_length = int(0.8*length)
        train_limit = self.train_limit or train_length
        train_set = dataset[:min(train_length, train_limit)]
        test_set = dataset[train_length:]

        logger.info("Training with %d items", len(train_set))

        results = {}
        for parser in parsers:
            logger.info("Training parser %r", parser)
            parser.train(train_set)
            logger.info("Evaluating on test set")
            parser_results = parser.test(test_set)
            results[repr(parser)] = parser_results
        return results
        


