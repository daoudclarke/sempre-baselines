from random import Random


class RandomData(object):
    def __init__(self):
        self.random = Random(1)

    choices = [
        'banana',
        'apple',
        'fish',
        'socks',
        'orangutan',
        'organism',
        'cheetah',
        'mixer',
        'potato',
        'john',
        'example',
        'wichita',
        'of'
        'large'
        'really',
        'very']

    def get_random_words(self):
        words = [self.random.choice(self.choices)
                 for i in range(3)]
        return ' '.join(words)
        # self.random.shuffle(word)
        # return ''.join(word)

    def get_data(self):
        random_words = [self.get_random_words() for i in range(100)]
        random_words.sort()
        for word1 in random_words:
            source = 'what is a %s like on tuesday?' % word1
            yield {
                'source':  source,
                'target': 'what on earth could a %s be?' % word1,
                'score': 0.8,
                }
            yield {
                'source': source,
                'target': 'what do you think a %s is?' % word1,
                'score': 0.8,
                }
            yield {
                'source': source,
                'target': 'Do you like %s?' % word1,
                'score': 0.0,
                }
            word2 = self.get_random_words()
            yield {
                'source':  source,
                'target': 'what on earth could a %s be?' % word2,
                'score': 0.0,
                }

