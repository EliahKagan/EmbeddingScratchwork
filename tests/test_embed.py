#!/usr/bin/env python

"""Tests for the embedding functions in the ``embed`` module."""

# pylint: disable=missing-function-docstring

import os
from typing import Any
import threading
import unittest

import numpy as np
from parameterized import parameterized, parameterized_class

from embed import (
    embed_one,
    embed_many,
    embed_one_eu,
    embed_many_eu,
    embed_one_req,
    embed_many_req,
)
from . import _helpers

_helpers.configure_logging()
_maybe_cache = _helpers.get_maybe_caching_decorator()


@parameterized_class(('name', 'func'), [
    (embed_one.__name__, staticmethod(_maybe_cache(embed_one))),
    (embed_one_eu.__name__, staticmethod(_maybe_cache(embed_one_eu))),
    (embed_one_req.__name__, staticmethod(_maybe_cache(embed_one_req))),
])
class TestEmbedOne(unittest.TestCase):
    """Tests for ``embed_one`` and ``embed_one_eu``."""

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
    (embed_many_req.__name__, staticmethod(_maybe_cache(embed_many_req))),

])
class TestEmbedMany(unittest.TestCase):
    """Tests for ``embed_many`` and ``embed_many_eu``."""

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


# NOTE: Manually enable this briefly if needed, but otherwise keep it skipped.
@unittest.skip("No need to regularly slam OpenAI's servers. Also: very slow.")
class TestBackoff(unittest.TestCase):
    """
    Tests that backoff works in the ``requests`` version.

    This is hard to check for if one's OpenAI account is not subject to reduced
    rate limits, which happens only in the trial period and shortly thereafter.
    But occasionally it may be valuable to check this explicitly. So this sends
    a lot of requests to the OpenAI embeddings endpoint in a short time.
    """

    def setUp(self):
        """Reduce the risk of accidentally running this on CI."""
        if os.getenv('CI') is not None:
            raise Exception(  # pylint: disable=broad-exception-raised
                "These tests shouldn't run via continuous integration.")

    def test_embed_one_req_backs_off(self):
        def run(thread_index):
            for loop_index in range(75):
                text = f'Testing rate limiting. {thread_index=} {loop_index=}'
                embed_one_req(text)

        threads = [
            threading.Thread(target=run, args=(thread_index,))
            for thread_index in range(75)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # FIXME: Assert that the expected "INFO" log message(s) were created.


if __name__ == '__main__':
    unittest.main()
