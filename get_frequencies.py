from gensim.utils import simple_preprocess as tokenize
import json
from collections import Counter

STOPWORDS = {'what', 'is', 'the', 'of'}


def get_source_sentences(examples):
    previous = None
    for example_json in examples:
        example = json.loads(example_json)
        source = example['source']
        if source != previous:
            yield source
            previous = source

def get_frequencies(examples):
    counter = Counter()
    for source in get_source_sentences(examples):
        tokens = tokenize(source)
        counter.update(tokens)
        #print counter.most_common(100)
    return counter

def save_frequencies(frequencies, path):
    output_file = open(path, 'w')
    json.dump(dict(frequencies), output_file)

if __name__ == "__main__":
    import gzip
    input_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json.gz'
    input_file = gzip.open(input_path)
    frequencies = get_frequencies(input_file)
    save_frequencies(frequencies, 'frequencies.json')
