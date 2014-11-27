import json
from itertools import groupby
from operator import itemgetter
import sys
import os
import numpy as np
from random import Random
from analyse import analyse

from kernelparse.log import logger

class OracleParser(object):
    """
    Parser that can see the answers and just chooses the best one
    available. This gives us an upper bound on performance.
    """
    def train(self, train_set):
        pass

    def test(self, test_set):
        results = []
        for source, group in test_set:
            scores = [item['score'] for item in group]
            best_index = np.argmax(scores)
            results.append(group[best_index])
        return results

