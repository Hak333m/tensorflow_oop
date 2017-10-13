"""
Bag of words.
"""

import numpy as np
import collections
import operator

from tensorflow_oop.compatibility_imports import *

class TFBagOfWords(object):

    """
    Bag of words model.

    Attributes:
        size                   Dictionary length.
        non_chars              String with chars for excluding.
        lower_case             Indicator of case insensitive mode.
        digit_as_zero          Indicator of converting all digits to zero.
        min_count              Filter minimum words count for adding to dictionary.
        max_count              Filter maximum words count for adding to dictionary.
        max_dictionary_size    Maximal dictionary size, other words marked as unknown_word.
        unknown_word           Word for replace rare words if dictionary size greater then max_dictionary_size.
        all_words_counter      Counter object of all words in texts.
        rare_words_counter     Counter object of too rare words in texts.
        freq_words_counter     Counter object of too frequent words in texts.
        valid_words_counter    Counter object of valid words in texts.
        dictionary             Dict object of correct words in texts.

    """

    __slots__ = ['size',
                 'non_chars', 'lower_case', 'digit_as_zero',
                 'min_count', 'max_count',
                 'words_counter', 'dictionary']

    def __init__(self,
                 texts,
                 non_chars=None,
                 lower_case=True,
                 digit_as_zero=True,
                 min_count=1,
                 max_count=np.inf,
                 max_dictionary_size=np.inf,
                 rare_word='UNKRARE'
                 freq_word='UNKFREQ'):
        """Constructor.

        Arguments:
            texts                  List of texts for building dictionary.
            non_chars              String with chars for excluding.
            lower_case             Indicator of case insensitive mode.
            digit_as_zero          Indicator of converting all digits to zero.
            min_count              Filter minimum words count for adding to dictionary.
            max_count              Filter maximum words count for adding to dictionary.
            max_dictionary_size    Maximal dictionary size, other words marked as unknown_word.
            unknown_word           Word for replace rare words if dictionary size greater then max_dictionary_size.

        """

        assert min_count > 0, \
            '''Minimum count should be greater then zero:
               min_count = %s''' % min_count
        assert max_count >= min_count, \
            '''Maximum count should not be less then minimum count:
               min_count = %s, max_count = %s''' % (min_count, max_count)
        assert max_dictionary_size > 0, \
            '''Maximum dictionary size should be greater then zero:
               max_dictionary_size = %s''' % max_dictionary_size
        assert isinstance(rare_word, str), \
            '''Rare word should be string:
               type(rare_word) == %s''' % type(rare_word)
        assert isinstance(freq_word, str), \
            '''Frequent word should be string:
               type(freq_word) == %s''' % type(freq_word)

        # Save properties
        if non_chars is not None:
            assert isinstance(non_chars, str), \
                '''Non char symbols should be string:
                   type(non_chars) == %s''' % type(non_chars)
            self.non_chars = non_chars
        else:
            self.non_chars = '/.,!?()_-";:*=&|%<>@\'\t\n\r'
        self.lower_case = lower_case
        self.digit_as_zero = digit_as_zero
        self.min_count = min_count
        self.max_count = max_count
        self.rare_word = rare_word
        self.freq_word = freq_word

        # Calculate statistic
        words = self.list_of_words(' '.join(texts))
        self.all_words_counter = collections.Counter(words)

        # Calculate dictionary
        self.dictionary = {}
        for word in sorted(self.words_counter):
            if self.min_count <= self.words_counter[word] <= self.max_count:
                self.dictionary[word] = len(self.dictionary)
        self.size = len(self.dictionary)
        
        # Check dictionary size
        if self.size > max_dictionary_size:
            # Check if should be replaced unknown words
            if unknown_word is None:
                # Get known words
                known_words = set(collections.Counter(self.dictionary).most_common(max_dictionary_size))

                # Recalculate dictionary
                self.dictionary = {unknown_word: 0}
                for word in known_words:
                    self.dictionary[word] = len(self.dictionary)
                self.size = len(self.dictionary)

            else:
                # Get known words
                known_words = set(collections.Counter(self.dictionary).most_common(max_dictionary_size - 1))

                assert unknown_word not in known_words, \
                    '''Unknown word string '%s' can't be in known words.''' % unknown_word

                # Calculate unknown words count
                for word in self.dictionary:
                    if word not in known_words:
                        self.unknown_count += 1

                # Recalculate dictionary
                self.dictionary = {unknown_word: 0}
                for word in known_words:
                    self.dictionary[word] = len(self.dictionary)
                self.size = len(self.dictionary)

    def list_of_words(self, text):
        """Get list of standart words from text.

        Arguments:
            text        Text in string format.

        Return:
            words       List of extracted words.

        """
        words = self.preprocessing(text).split(' ')
        if '' in words:
            words.remove('')
        return words

    def vectorize(self, text, binary):
        """Calculate vector by text.

        Arguments:
            text        Conversation text.
            binary      Indicator of using only {0, 1} instead rational from [0, 1].

        Return:
            vector      Numeric representation vector.

        """

        # Calculate statistic
        words = self.list_of_words(text)
        vector = np.zeros(self.size)
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
                if binary:
                    vector[index] = 1.
                else:
                    vector[index] += 1.

        # Validate data
        valid_count = np.sum(vector)
        assert valid_count > 0, \
            '''Valid words count should be greater then zero:
            valid_count = %s''' % valid_count

        # Normalize if necessary
        if not binary:
            vector /= valid_count
        return vector

    def preprocessing(self, old_text):
        """Standartize text to one format.

        Arguments:
            old_text    Text to preprocessing in string format.

        Return:
            new_text    Processed text.

        """
        if self.lower_case:
            new_text = old_text.lower()
        for non_char in self.non_chars:
            new_text = new_text.replace(non_char, ' ')
        if self.digit_as_zero:
            for i in range(1, 10):
                new_text = new_text.replace(str(i), '0')
        while new_text.find('  ') >= 0:
            new_text = new_text.replace('  ', ' ')
        new_text = new_text.strip()
        return new_text

    def __len__(self):
        """Unique words count in dictionary."""
        return self.size

    def __str__(self):
        """String formatting."""
        string = 'TFBagOfWords object:\n'
        for attr in self.__slots__:
            if attr != 'words_counter' and attr != 'dictionary':
                string += '%20s: %s\n' % (attr, getattr(self, attr))
        sorted_words_counter = sorted(self.words_counter.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
        string += '%20s:\n%s\n...\n%s\n' % ('words_counter',
            '\n'.join([str(elem) for elem in sorted_words_counter[:10]]),
            '\n'.join([str(elem) for elem in sorted_words_counter[-10:]]))
        sorted_dictionary = sorted(self.dictionary.items(),
                                   key=operator.itemgetter(0),
                                   reverse=False)
        string += '%20s:\n%s\n...\n%s\n' % ('dictionary',
            '\n'.join([str(elem) for elem in sorted_dictionary[:10]]),
            '\n'.join([str(elem) for elem in sorted_dictionary[-10:]]))
        return string[:-1]
