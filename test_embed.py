#!/usr/bin/env python

"""Tests for the embed module."""

# pylint: disable=missing-function-docstring

import os
import pickle
import re
from typing import Any
import unittest

import numpy as np
from parameterized import parameterized, parameterized_class

from _test_helpers import lazy_if, cache_by
from embed import embed_one, embed_many, embed_one_eu, embed_many_eu


_SHOULD_CACHE_RE = re.compile(r'\A\s*(?:yes|true|\+?0*[1-9][0-9]*)\s*\Z', re.I)
"""Regular expression that _should_cache matches to an environment variable."""


def _should_cache():
    """
    Decide if tests in this module should cache calls to embedding functions.

    This is done if the TESTS_CACHE_EMBEDDING_CALLS environment variable exists
    and holds "yes" or "true" (case-insensitively), or a positive integer.
    """
    value = os.environ.get('TESTS_CACHE_EMBEDDING_CALLS', default='')
    return _SHOULD_CACHE_RE.match(value)


_maybe_cache = lazy_if(_should_cache, cache_by(pickle.dumps))
"""
Decorator that adds caching if ``TESTS_CACHE_EMBEDDING_CALLS`` says to do so.

This happens dynamically and lazily, so it can be changed after the module is
imported, any time before the first test-case call to a wrapper function. That
is, even though ``_maybe_cache`` is called at import time due to its presence
in ``@parameterized_class`` decorations, whether caching happens is not decided
until the function returned by ``_maybe_cache`` is called for the first time.
"""


@parameterized_class(('name', 'func'), [
    (embed_one.__name__, staticmethod(_maybe_cache(embed_one))),
    (embed_one_eu.__name__, staticmethod(_maybe_cache(embed_one_eu))),
])
class TestEmbedOne(unittest.TestCase):
    """Tests for embed_one and embed_one_eu."""

    func: Any

    def test_returns_numpy_array(self):
        result = self.func("Your text string goes here")
        with self.subTest('ndarray'):
            self.assertIsInstance(result, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(result[0], np.float32)

    def test_shape_is_model_dimension(self):
        result = self.func("Your text string goes here")
        self.assertEqual(result.shape, (1536,))

    @parameterized.expand([
        ("catrun", "The cat runs.", "El gato corre."),
        ("dogwalk", "The dog walks.", "El perro camina."),
        ("lionsleep", "The lion sleeps.", "El león duerme."),
    ])
    def test_en_and_es_sentence_are_very_similar(
            self, _name, text_en, text_es):
        embedding_en = self.func(text_en)
        embedding_es = self.func(text_es)
        result = np.dot(embedding_en, embedding_es)
        self.assertGreaterEqual(result, 0.9)

    def test_different_meanings_are_dissimilar(self):
        sentence_one = self.func("Your text string goes here")
        sentence_two = self.func("The cat runs.")
        result = np.dot(sentence_one, sentence_two)
        self.assertLess(result, 0.8)


@parameterized_class(('name', 'func'), [
    (embed_many.__name__, staticmethod(_maybe_cache(embed_many))),
    (embed_many_eu.__name__, staticmethod(_maybe_cache(embed_many_eu))),
])
class TestEmbedMany(unittest.TestCase):
    """Tests for embed_many and embed_many_eu."""

    func: Any

    def setUp(self):
        self._many = self.func([
            "Your text string goes here",
            "The cat runs.",
            "El gato corre.",
            "The dog walks.",
            "El perro camina.",
        ])

    def test_returns_numpy_array(self):
        with self.subTest('ndarray'):
            self.assertIsInstance(self._many, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(self._many[0][0], np.float32)

    def test_shape_is_model_dimension(self):
        self.assertEqual(self._many.shape, (5, 1536))

    def test_en_and_es_sentences_are_very_similar(self):
        with self.subTest('catrun'):
            result = np.dot(self._many[1], self._many[2])
            self.assertGreaterEqual(result, 0.9)
        with self.subTest('dogwalk'):
            result = np.dot(self._many[3], self._many[4])
            self.assertGreaterEqual(result, 0.9)

    def test_different_meanings_are_dissimilar(self):
        result = np.dot(self._many[0], self._many[1])
        self.assertLess(result, 0.8)


if __name__ == '__main__':
    unittest.main()
