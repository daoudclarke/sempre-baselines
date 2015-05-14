"""
Class to analyse errors produced by the tensor-based classifier.
"""

from kernelparse.tensor import TensorParser
from kernelparse.oracle import OracleParser

class TensorErrorParser(TensorParser):
    def test(self, test_set):
        oracle = OracleParser()
        oracle_results = oracle.test(test_set)

        tensor_results = super(TensorErrorParser, self).test(test_set)

        for tensor_result, oracle_result in zip(tensor_results, oracle_results):
            if oracle_result['score'] == 0:
                print "Oracle zero: ", oracle_result
                tensor_result['error'] = 'oracle'
            else:
                score = tensor_result['score']
                if  score == 0.0:
                    oracle_features = list(self.get_example_features(oracle_result))
                    oracle_feature_scores = {feature: self.feature_scores.get(feature, None)
                                             for feature in oracle_features}
                    tensor_result['oracle_target'] = oracle_result['target']
                    tensor_result['oracle_features'] = oracle_feature_scores

                    tensor_features = list(self.get_example_features(tensor_result))
                    tensor_feature_scores = {feature: self.feature_scores.get(feature, None)
                                             for feature in tensor_features}
                    tensor_result['tensor_features'] = tensor_feature_scores
                    tensor_result['error'] = 'features'
                else:
                    print "Tensor success", score
                    tensor_result['error'] = 'none'
        return tensor_results
