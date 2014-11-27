from itertools import groupby
from operator import itemgetter
from random import Random

from kernelparse.log import logger


class Experiment(object):
    """
    Class to run experiments on Sempre paraphrase datasets.
    """

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
        train_set = dataset[:train_length]
        test_set = dataset[train_length:]

        results = {}
        for parser in parsers:
            parser.train(train_set)
            parser_results = parser.test(test_set)
            results[repr(parser)] = parser_results
        return results
        


