from operator import itemgetter
from collections import defaultdict

import numpy as np

from gensim.utils import simple_preprocess as tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

from kernelparse.tensor import TensorParser
from kernelparse.log import logger

STOPWORDS = {'what', 'is', 'the', 'of'}

def get_precision_recall_f1(actual, predicted):
    """
    Return the F1 score, given two sets of entities: the correct set,
    and those predicted.
    """
    intersection = float(len(set(actual) & set(predicted)))
    if intersection == 0.0:
        return 0.0, 0.0, 0.0

    precision = intersection/len(predicted)
    recall = intersection/len(actual)
    f1_score = 2*precision*recall/(precision + recall)
    return precision, recall, f1_score


def get_f1_score(actual, predicted):
    _, _, f1 = get_precision_recall_f1(actual, predicted)
    return f1


class EntityParser(TensorParser):
    def train(self, train_set):
        logger.info("Converting features to list")
        all_features = []
        all_values = []
        for source, examples in train_set:
            logger.info("Processing query: %r", source)
            entity_features = self.get_entity_features(examples)

            positive_entities = set(examples[0]['gold'])
            for entity, features in entity_features.items():
                #logger.debug("Entity: %r, features: %r", entity, features)
                all_features.append(features)
                all_values.append(entity in positive_entities)

        self.build_model(all_features, all_values)

    def test(self, test_set):
        logger.info("Evaluation on test set")
        count = 0
        results = []
        for source, examples in test_set:
            logger.info("Processing query: %r", source)
            entity_features = self.get_entity_features(examples)
            all_features = []
            for entity, features in entity_features.items():
                all_features.append(features)
            vectors = self.vectorizer.transform(all_features)
            predictions = self.classifier.decision_function(vectors)
            best_index = np.argmax(predictions)
            best_entity = entity_features.keys()[best_index]
            logger.debug("Chosen entity: %r", best_entity)
            score = get_f1_score([best_entity], examples[0]['gold'])
            results.append({'score': score})

        return results

    def get_entity_features(self, examples):
        entity_features = defaultdict(dict)
        for example in examples:
            example_features = self.get_example_features(example)
            entities = example['value']
            for entity in entities:
                entity_features[entity].update(example_features)
            entity_features[entity].update({entity: 1.0 for entity in entities})
        return entity_features

    def __repr__(self):
        return type(self).__name__
