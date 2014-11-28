from random import Random

class RandomParser(object):
    """
    Baseline parser that simply chooses a target sentence at random.
    """

    def __init__(self):
        self.random = Random(1)

    def train(self, train_set):
        pass

    def test(self, test_set):
        results = []
        for source, group in test_set:
            results.append(self.random.choice(group))
        return results

    def __repr__(self):
        return type(self).__name__
