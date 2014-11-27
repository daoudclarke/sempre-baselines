from kernelparse.bigramtensor import BigramTensorParser, get_bigrams

def test_get_bigrams():
    tokens = "a nice long sentence"
    bigrams = get_bigrams(tokens.split())
    
    assert bigrams == [("a", "nice"), ("nice", "long"), ("long", "sentence")]

def test_bigrams_unigrams():
    sentence = "nice short sentence"
    parser = BigramTensorParser()
    features = parser.get_sentence_features(sentence)

    assert features == ["nice", "short", "sentence",
                        "nice#short", "short#sentence"]
