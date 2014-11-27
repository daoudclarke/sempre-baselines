# Bismillahi-r-Rahmani-r-Rahim

import pytest
import os
from tempfile import mkdtemp
from kernelparsetest.randomdata import RandomData
from kernelparse.analyse import analyse
from kernelparse.baseline import Experiment

@pytest.fixture
def data():
    random = RandomData()
    return random.get_data()

def test_ordered_data_run(data):
    experiment = Experiment(data)

    experiment.train()
    results = experiment.test()

    mean, error = analyse(results)
    print "Mean: %f +/- %f" % (mean, error)
    assert abs(mean - 0.8) < 1e-5


def test_that_test_data_in_training_set_errors(data):
    with pytest.raises(ValueError):
        experiment = Experiment(list(data)*2)
