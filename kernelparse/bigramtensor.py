from kernelparse.tensor import TensorParser


def get_bigrams(tokens):
    """
    Get the bigrams from a sequence of tokens.
    """
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(tuple(tokens[i:i+2]))
    return bigrams

class BigramTensorParser(TensorParser):
    def get_sentence_features(self, sentence):
        tokens = super(BigramTensorParser, self).get_sentence_features(sentence)
        bigrams = ["%s#%s" % pair for pair in get_bigrams(tokens)]
        return tokens + bigrams
