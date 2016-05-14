import array
import io
import re

import numpy as np
from sklearn.base import TransformerMixin


module_rng = np.random.RandomState(1234)


TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)


def tokenizer(document):
    return TOKENIZER_RE.findall(document)


def new_embedding(no_components, dtype):
    return (module_rng.uniform(-1, 1, no_components).astype(dtype),)


class WordVectors(object):
    def __init__(self, vectors, vocabulary):
        self.embeddings_ = vectors
        self.vocabulary_ = vocabulary

    @property
    def no_components(self):
        return int(self.embeddings_.shape[1])

    @property
    def no_vectors(self):
        return len(self.vocabulary_)

    def word_id(self, word, allow_oov=False):
        oov_id = self.no_vectors if allow_oov else None
        return self.vocabulary_.get(word, oov_id)

    def word_vector(self, word):
        word_id = self.word_id(word)
        if word_id:
            return self.embedding_[word_id, :]

    @classmethod
    def from_stanford(cls, filename, dtype=np.float64):
        """
        Load model from the output files generated by
        the C code from http://nlp.stanford.edu/projects/glove/.

        The entries of the word dictionary will of type
        unicode in Python 2 and str in Python 3.
        """
        vocabulary = {}
        vectors = array.array('f' if dtype == np.float32 else 'd')

        with io.open(filename, 'r', encoding='utf-8') as savefile:
            # read this in multiple processes. I hate waiting...
            for i, line in enumerate(savefile):
                tokens = line.strip().split(' ')

                word = tokens[0]
                entries = tokens[1:]

                vocabulary[word] = i
                vectors.extend(dtype(x) for x in entries)

        # Infer word vectors dimension
        no_components = len(entries)
        no_vectors = len(vocabulary)

        return cls(np.array(vectors, dtype=dtype).reshape(no_vectors, no_components),
                   vocabulary)


class WordVectorProcessor(TransformerMixin):
    def __init__(self, filename, w2v_type, max_document_length=100, allow_oov=True,
                 analyzer=None, dtype=np.float64):
        self.filename = filename
        self.w2v_type = w2v_type
        self.dtype = dtype
        self.analyzer = analyzer
        self.allow_oov = allow_oov
        self.max_document_length = max_document_length

    @property
    def embeddings_(self):
        extra_embeddings = new_embedding(self.vectors.no_components, self.dtype)
        if self.allow_oov:
            extra_embeddings += new_embedding(self.vectors.no_components, self.dtype)
        return np.vstack((self.vectors.embeddings_,) + extra_embeddings)

    def build_analyzer(self):
        return self.analyzer if self.analyzer else tokenizer

    def fit(self, raw_documents):
        self.vectors = WordVectors.from_stanford(self.filename, dtype=self.dtype)
        return self

    def transform(self, raw_documents):
        analyzer = self.build_analyzer()
        for doc in raw_documents:
            sequence_length = 0
            word_ids = np.repeat(self.vectors.no_vectors + 1,
                                 self.max_document_length)
            for word in analyzer(doc):
                if sequence_length >= self.max_document_length:
                    break
                word_id = self.vectors.word_id(word, allow_oov=self.allow_oov)
                if word_id:
                    word_ids[sequence_length] = word_id
                    sequence_length += 1
            yield word_ids
