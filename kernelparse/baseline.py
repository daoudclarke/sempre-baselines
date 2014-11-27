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
#STOPWORDS = set()

def preprocess(sentence):
    tokens = set(tokenize(sentence))
    return tokens - STOPWORDS

def get_example_features(example):
    source_tokens = preprocess(example['source'])
    target_tokens = preprocess(example['target'])
    shared = source_tokens & target_tokens
    source_only = source_tokens - target_tokens
    target_only = target_tokens - source_tokens

    features = []
    for source in source_tokens:
        for target in target_tokens:
            features.append(source + ':' + target)
    return {f: 1.0 for f in features}


def get_features(examples):
    for example in examples:
        features = get_example_features(example)
        yield features, example['score'] > 0.0

def get_examples(self, filename):
    full_path = os.path.join(self.dataset_path, filename)
    example_file = gzip.open(full_path)
    for row in example_file:
        yield json.loads(row)


class Experiment(object):
    num_folds = 3

    def __init__(self, dataset):
        dataset = [(source, list(group)) for source, group in
                    groupby(dataset, itemgetter('source'))]
        if len(dataset) != len(set(map(itemgetter(0), dataset))):
            raise ValueError("Dataset contains non-contiguous source examples")

        length = len(dataset)
        logger.info("Total number of instances: %d", length)
        assert length > 1

        self.random = Random(1)
        self.random.shuffle(dataset)

        train_length = int(0.8*length)
        self.train_set = dataset[:train_length]
        self.test_set = dataset[train_length:]

        
        

    def train(self):
        logger.info("Converting features to list")
        features = []
        values = []
        for source, examples in self.train_set:
            group_features = list(get_features(examples))
            values += [feature[1] for feature in group_features]
            group_features = [feature[0] for feature in group_features]
            features += group_features
        assert len(features) == len(values)


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

        feature_scores = self.vectorizer.inverse_transform(self.classifier.coef_)
        best_features = sorted(feature_scores[0].iteritems(), key=itemgetter(1), reverse=True)        
        logger.debug("Top SVM parameters: %r", best_features[:100])
        logger.debug("Top negative SVM parameters: %r", best_features[::-1][:100])

        logger.info("Finished training")


    def test(self):
        source_groups = self.test_set
        logger.info("Evaluation on test set")
        count = 0
        results = []
        for source, group in source_groups:
            self.random.shuffle(group)
            data = get_features(group)
            features, values = zip(*list(data))

            vectors = self.vectorizer.transform(features)
            predictions = self.classifier.decision_function(vectors)
            best_index = np.argmax(predictions)
            results.append(group[best_index])
            count += 1
            if count % 100 == 0:
                logger.info("Processed %d items", count)
        return results


def output_results(results, results_path):
    with open(results_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)

if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json'
    results_path = 'results.json'

    with open(dataset_path) as dataset_file:
        #dataset = islice(dataset_file, 100000)
        dataset = [json.loads(row) for row in dataset_file if len(row.strip()) > 0]

    experiment = Experiment(dataset)
    experiment.train()
    results = experiment.test()
    output_results(results, results_path)
    print analyse(results)
    
