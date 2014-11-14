from gensim.utils import simple_preprocess as tokenize
import json

STOPWORDS = {'what', 'is', 'the', 'of'}

def preprocess(sentence):
    tokens = set(tokenize(sentence))
    return tokens - STOPWORDS

def get_tokens_length(tokens):
    return float(len(''.join(tokens)))

def get_overlap(example):
    source_tokens = preprocess(example['source'])
    target_tokens = preprocess(example['target'])
    #print source_tokens, target_tokens
    overlap = (get_tokens_length(source_tokens & target_tokens) /
               get_tokens_length(source_tokens | target_tokens))
    return overlap

def process_examples(example_file):
    for example_json in example_file:
        example = json.loads(example_json)
        overlap = get_overlap(example)
        print overlap, example





if __name__ == "__main__":
    import gzip
    input_path = '/home/dc/Experiments/sempre-paraphrase-dataset/examples.json.gz'
    input_file = gzip.open(input_path)
    process_examples(input_file)
    
