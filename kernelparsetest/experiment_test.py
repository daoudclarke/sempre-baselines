# Bismillahi-r-Rahmani-r-Rahim

import pytest
from kernelparsetest.randomdata import RandomData
from kernelparse.analyse import analyse
from kernelparse.experiment import Experiment
from kernelparse.tensor import TensorParser

@pytest.fixture
def data():
    random = RandomData()
    return random.get_data()

def test_ordered_data_run(data):
    parser = TensorParser()
    experiment = Experiment()

    results = experiment.run(data, parser)

    mean, error = analyse(results)
    print "Mean: %f +/- %f" % (mean, error)
    assert abs(mean - 0.8) < 1e-5


def test_that_test_data_in_training_set_errors(data):
    with pytest.raises(ValueError):
        experiment = Experiment()
        parser = TensorParser()
        experiment.run(list(data)*2, parser)
