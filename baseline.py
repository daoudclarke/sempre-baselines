from gensim.utils import simple_preprocess as tokenize
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from itertools import islice, groupby
from operator import itemgetter
import sys
import os
import numpy as np
from random import Random
from lru import lru_cache
from analyse import analyse

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

STOPWORDS = {'what', 'is', 'the', 'of'}

#@lru_cache(maxsize=32)
def preprocess(sentence):
    tokens = set(tokenize(sentence))
    return tokens - STOPWORDS

def get_example_features(example):
    source_tokens = preprocess(example['source'])
    target_tokens = preprocess(example['target'])
    #print source_tokens, target_tokens
    shared = source_tokens & target_tokens
    source_only = source_tokens - target_tokens
    target_only = target_tokens - source_tokens

    #features = {}
    features = {'s:' + token: 1.0 for token in source_only}
    features.update({'t:' + token: 1.0 for token in target_only})
    features.update({'b:' + token: 1.0 for token in shared})

    # features = {'s:' + token: 1.0 for token in source_tokens}
    # features.update({'t:' + token: 1.0 for token in target_tokens})

    return features

def get_features(examples):
    for example in examples:
        features = get_example_features(example)
        #yield features, int(example['score']*5)
        yield features, example['score'] > 0.0

class Experiment(object):
    def __init__(self, dataset_path, results_path):
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.random = Random(1)

    def get_examples(self, filename):
        full_path = os.path.join(self.dataset_path, filename)
        example_file = gzip.open(full_path)
        for row in example_file:
            yield json.loads(row)

    def train(self):
        examples = self.get_examples('examples-train.json.gz')
        features = get_features(examples)

        features = islice(features, 50000)
        features, values = zip(*list(features))

        #print features
        #print values

        logger.info("Training - building vectors")
        self.vectorizer = DictVectorizer()
        vectors = self.vectorizer.fit_transform(features)

        logger.info("Training classifier")
        svm = LinearSVC()

        parameters = {'C': [0.1, 1.0, 10.0, 100.0]}
        self.classifier = GridSearchCV(svm, parameters, scoring='mean_absolute_error')
        #self.classifier = svm

        self.classifier.fit(vectors, values)
        self.classifier = self.classifier.best_estimator_

        logger.info("SVM classes: %r", self.classifier.classes_)

        # feature_scores = self.vectorizer.inverse_transform(self.classifier.coef_)
        # best_features = sorted(feature_scores[0].iteritems(), key=itemgetter(1), reverse=True)        
        # logger.debug("Top SVM parameters: %r", best_features[:100])
        # logger.debug("Top negative SVM parameters: %r", best_features[::-1][:100])

        logger.info("Finished training")


    def test(self):
        test_path = join(self.dataset_path, 'examples-test.json.gz')
        examples = self.get_examples(test_path)

        source_groups = groupby(examples, itemgetter('source'))
        logger.info("Evaluation on test set")
        count = 0
        with open(self.results_path, 'w') as results_file:
            for source, group in source_groups:
                group = list(group)
                self.random.shuffle(group)
                data = get_features(group)
                features, values = zip(*list(data))

                vectors = self.vectorizer.transform(features)
                predictions = self.classifier.decision_function(vectors)
                #predictions = self.classifier.predict(vectors)
                #print sorted(zip(predictions, group), reverse=True)
                best_index = np.argmax(predictions)
                results_file.write(json.dumps(group[best_index]) + '\n')
                count += 1
                if count % 100 == 0:
                    logger.info("Processed %d items", count)
                    #logger.debug("Cache info: %r", preprocess.cache_info())

    def run_experiment(self):
        self.train()
        self.test()
        analyse(self.results_path)

if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/'
    results_path = 'results.json'

    experiment = Experiment(dataset_path, results_path)
    experiment.run_experiment()
    
