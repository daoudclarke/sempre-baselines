from operator import itemgetter

import numpy as np

from gensim.utils import simple_preprocess as tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

from kernelparse.log import logger

STOPWORDS = {'what', 'is', 'the', 'of'}

class TensorParser(object):
    def train(self, train_set):
        logger.info("Converting features to list")
        features = []
        values = []
        for source, examples in train_set:
            group_features = list(self.get_features(examples))
            values += [feature[1] for feature in group_features]
            group_features = [feature[0] for feature in group_features]
            features += group_features
        logger.info("Total number of instances: %d", len(features))
        assert len(features) == len(values)


        logger.info("Training - building vectors")
        self.vectorizer = DictVectorizer()
        vectors = self.vectorizer.fit_transform(features)


        logger.info("Training classifier")
        svm = LinearSVC()

        parameters = {'C': [0.1, 1.0, 10.0, 100.0]}
        self.classifier = GridSearchCV(svm, parameters, scoring='f1')

        self.classifier.fit(vectors, values)
        self.classifier = self.classifier.best_estimator_

        logger.info("SVM classes: %r", self.classifier.classes_)

        feature_scores = self.vectorizer.inverse_transform(self.classifier.coef_)
        best_features = sorted(feature_scores[0].iteritems(), key=itemgetter(1), reverse=True)        
        logger.debug("Top SVM parameters: %r", best_features[:100])
        logger.debug("Top negative SVM parameters: %r", best_features[::-1][:100])

        logger.info("Finished training")

    def test(self, test_set):
        logger.info("Evaluation on test set")
        count = 0
        results = []
        for source, group in test_set:
            data = self.get_features(group)
            features, values = zip(*list(data))

            vectors = self.vectorizer.transform(features)
            predictions = self.classifier.decision_function(vectors)
            best_index = np.argmax(predictions)
            results.append(group[best_index])
            count += 1
            if count % 100 == 0:
                logger.info("Processed %d items", count)
        return results

    def get_features(self, examples):
        for example in examples:
            features = self.get_example_features(example)
            yield features, example['score'] > 0.0

    def get_example_features(self, example):
        source_tokens = self.get_sentence_features(example['source'])
        target_tokens = self.get_sentence_features(example['target'])
        features = []
        for source in source_tokens:
            for target in target_tokens:
                features.append(source + ':' + target)
        return {f: 1.0 for f in features}

    def get_sentence_features(self, sentence):
        tokens = tokenize(sentence)
        return [token for token in tokens if token not in STOPWORDS]

    def __repr__(self):
        return type(self).__name__
