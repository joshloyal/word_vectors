import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from word_vectors import WordVectors, WordVectorProcessor

#vectors = WordVectors.from_stanford('pretrained/glove.6B.100d.txt', dtype=np.float32)
#sequence = [vectors.word_id(word, allow_oov=True) for word in 'my name is josh'.split(' ')]

raw_documents = ['hi my name is josh.',
                 'what is your name?']

sklearn_analyzer = CountVectorizer().build_analyzer()
processor = WordVectorProcessor(filename='pretrained/glove.6B.100d.txt',
                                w2v_type='stanford',
                                analyzer=sklearn_analyzer,
                                dtype=np.float32)

sequence = processor.fit_transform(raw_documents)
