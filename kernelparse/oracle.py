import numpy as np

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

    def __repr__(self):
        return type(self).__name__

