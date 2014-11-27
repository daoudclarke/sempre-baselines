from kernelparse.analyse import analyse
from kernelparse.experiment import Experiment
from kernelparse.oracle import OracleParser

from kernelparsetest.experiment_test import data
assert data   # To make pyflakes happy

def test_oracle(data):
    parser = OracleParser()
    experiment = Experiment()

    results = experiment.run(data, [parser])

    analysis = analyse(results)
    mean, error = analysis.values()[0]

    print "Mean: %f +/- %f" % (mean, error)
    assert abs(mean - 0.8) < 1e-5

