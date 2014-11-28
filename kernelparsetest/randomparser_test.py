from kernelparse.analyse import analyse
from kernelparse.experiment import Experiment
from kernelparse.randomparser import RandomParser

from kernelparsetest.experiment_test import data
assert data   # To make pyflakes happy

def test_random_parser(data):
    parser = RandomParser()
    experiment = Experiment()

    results = experiment.run(data, [parser])

    analysis = analyse(results)
    mean, error = analysis.values()[0]

    print "Mean: %f +/- %f" % (mean, error)
    assert mean > 0.0 and mean < 0.8


