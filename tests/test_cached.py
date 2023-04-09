#!/usr/bin/env python

"""
Tests for embedding functions in the ``embed.cached`` submodule.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

import json
import pathlib
import sys
import tempfile
from typing import Any
import unittest
import unittest.mock

from parameterized import parameterized_class

import embed
from embed import cached
from tests import _helpers

_helpers.configure_logging()

_HOLA_FILENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b.json'
)
"""Filename that would be generated from the input ``'hola'``."""


@parameterized_class(('name', 'func'), [
    (cached.embed_one.__name__, staticmethod(cached.embed_one)),
    (cached.embed_one_eu.__name__, staticmethod(cached.embed_one_eu)),
    (cached.embed_one_req.__name__, staticmethod(cached.embed_one_req)),
])
@_helpers.maybe_cache_embeddings_in_memory
class TestDiskCachedEmbedOne(unittest.TestCase):
    """Tests of ``embed.cached.embed_one*`` functions, which cache to disk."""

    name: Any
    func: Any

    def setUp(self):
        """Create a temporary directory."""
        # pylint: disable=consider-using-with  # tearDown cleans this up.
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._dir_path = pathlib.Path(self._temporary_directory.name)

    def tearDown(self):
        """Delete the temporary directory."""
        self._temporary_directory.cleanup()

    # Test returned embeddings could plausibly be correct

    # Test delegation to the non-caching version
    def test_calls_same_name_non_caching_version_if_not_cached(self):
        with self._patch_underlying_embedder() as mock:
            self.func('hola', data_dir=self._dir_path)

        mock.assert_called_once_with('hola')

    # Test saving new files
    def test_saves_file_if_not_cached(self):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:embed.cached:{self.name}: saved: {path}'

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    # Test loading existing files
    def test_loads_file_if_cached(self):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:embed.cached:{self.name}: loaded: {path}'
        fake_data = [1.0] + [0.0] * (embed.DIMENSION - 1)  # Normalized vector.

        with open(file=path, mode='w', encoding='utf-8') as file:
            json.dump(obj=fake_data, fp=file)

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    # Test different functions access existing files
    def test_saves_file_that_any_implementation_can_load(self):
        path = self._dir_path / _HOLA_FILENAME
        self.func('hola', data_dir=self._dir_path)

        for load_func in (cached.embed_one,
                          cached.embed_one_eu,
                          cached.embed_one_req):
            with self.subTest(load_func=load_func):
                expected_message = (
                    f'INFO:embed.cached:{load_func.__name__}: loaded: {path}'
                )

                with self.assertLogs(logger=cached.__name__) as log_context:
                    load_func('hola', data_dir=self._dir_path)

                self.assertEqual(log_context.output, [expected_message])

    # Test for load auditing event
    def test_for_load_audit_event(self):
        path = self._dir_path / _HOLA_FILENAME
        fake_data = [1.0] + [0.0] * (embed.DIMENSION - 1)  # Normalized vector.
        with open(file=path, mode='w', encoding='utf-8') as file:
            json.dump(obj=fake_data, fp=file)

        events = []

        expected = (str(path), 'r')

        def hook_open(event, args):
            if event == 'open' and record_open:
                path, mode, _ = args
                events.append((path, mode))

        record_open = True
        sys.addaudithook(hook_open)
        try:
            self.func('hola', data_dir=self._dir_path)
        finally:
            record_open = False

        self.assertIn(expected, events)

    # Test for save auditing event

    # Test file is created when should save
    def test_saved_embedding_exists(self):
        path = self._dir_path / _HOLA_FILENAME
        self.func('hola', data_dir=self._dir_path)
        self.assertTrue(path.is_file())

    # Test even when data_dir is not passed

    def _patch_underlying_embedder(self):
        """Patch the same-named function in ``embed``, to examine its calls."""
        return unittest.mock.patch(
            target=f'{embed.__name__}.{self.name}',
            wraps=getattr(embed, self.name),
        )


# FIXME: Finish writing most or all of this module's tests. Roughly speaking:
#
#   (1) For each remaining "# Test ..." comment in the above class, either
#       write a test, or convert it to a fixme if deferring it for later.
#
#       "Test returned embeddings could plausibly be correct" may make sense to
#       defer, since reorganizing the test suite (with or without inheritance)
#       could allow the logic in test_embed to be reused to do this robustly.
#
#   (2) Either make a second class here or modify and further parameterize the
#       above class, so that the embed.cached.embed_many* functions are tested.
#
#   (3) Remove the "# Test ..." comments (or convert them to docstrings).


# FIXME: Consider ways to reorganize the whole test suite (not just this file).
#
#   Reorganization may include how the suite is broken up into modules, but
#   also whether inheritance would better express the code reuse and fixture
#   logic currently done with @parameterized_class and helpers. In particular:
#
#   (a) Overriding an abstract func property would automatically play well with
#       monkey-patching, because expressions in property getters are evaluated
#       when properties are accessed: during the tests, while any patching is
#       active. This would eliminate the need for _helpers.Caller.
#
#   (b) Inheritance could reuse test logic from classes in test_embed for the
#       disk-caching versions in this module, to test behaviors expected both
#       of them and the non-caching versions. Both hit and miss scenarios
#       (which are alike in all the guaranteed behaviors they also share with
#       non-caching versions) could be covered, by using different fixures.
#
#       We could instead do this by expanding the existing @parameterized_class
#       decoration. But the resulting test classes would be in test_embed, and
#       the supporting fixture logic might be tricky to read and understand.
#
#   (c) One or more base classes could handle all the fixture logic currently
#       expressed with helper functions including our custom decorators. Most
#       significantly, this could handle conditionally patching embed.embed_*
#       to do in-memory caching, replacing @maybe_cache_embeddings_in_memory.


if __name__ == '__main__':
    unittest.main()
