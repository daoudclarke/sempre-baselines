import pytest
from kernelparsetest.randomdata import RandomData
from kernelparse.analyse import analyse
from kernelparse.experiment import Experiment
from kernelparse.oracle import OracleParser

from kernelparsetest.experiment_test import data

def test_oracle(data):
    parser = OracleParser()
    experiment = Experiment()

    results = experiment.run(data, parser)

    mean, error = analyse(results)
    print "Mean: %f +/- %f" % (mean, error)
    assert abs(mean - 0.8) < 1e-5

