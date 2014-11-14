from gensim.utils import simple_preprocess as tokenize
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.svm import LinearSVC
from itertools import islice, groupby
from operator import itemgetter
import sys
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

STOPWORDS = {'what', 'is', 'the', 'of'}

def preprocess(sentence):
    tokens = set(tokenize(sentence))
    return tokens - STOPWORDS

# def get_tokens_length(tokens):
#     return float(len(''.join(tokens)))

def get_example_features(example):
    source_tokens = preprocess(example['source'])
    target_tokens = preprocess(example['target'])
    #print source_tokens, target_tokens
    shared = source_tokens & target_tokens
    source_only = source_tokens - target_tokens
    target_only = target_tokens - source_tokens

    features = {'s:' + token:1 for token in source_only}
    features.update({'t:' + token:1 for token in target_only})
    features.update({'b:' + token:1 for token in shared})

    return features

def get_features(examples):
    for example in examples:
        features = get_example_features(example)
        yield features, example['score']

class Experiment(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_examples(self, filename):
        full_path = os.path.join(self.dataset_path, filename)
        example_file = gzip.open(full_path)
        for row in example_file:
            yield json.loads(row)

    def train(self):
        examples = self.get_examples('examples-train.json.gz')
        features = get_features(examples)

        features = islice(features, 1000)
        features, values = zip(*list(features))

        #print features
        #print values

        logger.info("Training - building vectors")
        self.vectorizer = DictVectorizer()
        vectors = self.vectorizer.fit_transform(features)

        logger.info("Training classifier")
        self.classifier = LinearSVC()
        self.classifier.fit(vectors, values)
        logger.info("Finished training")


    def test(self):
        test_path = join(self.dataset_path, 'examples-test.json.gz')
        examples = self.get_examples(test_path)

        source_groups = groupby(examples, itemgetter('source'))
        logger.info("Evaluation on test set")
        count = 0
        for source, group in source_groups:
            group = list(group)
            data = get_features(group)
            features, values = zip(*list(data))
        
            vectors = self.vectorizer.transform(features)
            predictions = self.classifier.predict(vectors)
            best_index = np.argmax(predictions)
            print group[best_index]
            count += 1
            if count % 100 == 0:
                logger.info("Processed %d items", count)

    def run_experiment(self):
        self.train()
        self.test()


if __name__ == "__main__":
    import gzip
    from os.path import join

    dataset_path = '/home/dc/Experiments/sempre-paraphrase-dataset/'
    experiment = Experiment(dataset_path)
    experiment.run_experiment()
    
