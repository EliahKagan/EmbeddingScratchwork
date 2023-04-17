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
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from parameterized import parameterized_class

import embed
from embed import cached
from tests import _audit, _helpers

_HOLA_FILENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b.json'
)
"""Filename that would be generated from the input ``'hola'``."""

_HOLA_HELLO_FILENAME = (
    '4a77f419587b08963e94105b8b9272531e53ade9621b613fda175aa0a96cd839.json'
)
"""Filename that would be generated from the input ``['hola', 'hello']``."""

_helpers.configure_logging()


def _patch_non_disk_caching_embedder(name):
    """Patch a function in ``embed`` to examine its calls."""
    embedder = getattr(embed, name)
    return patch(
        target=f'{embed.__name__}.{name}',
        wraps=embedder,
        __name__=embedder.__name__,
    )


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

    # FIXME: Test returned embeddings could plausibly be correct.

    def test_calls_same_name_non_caching_version_if_not_cached(self):
        with _patch_non_disk_caching_embedder(self.name) as mock:
            self.func('hola', data_dir=self._dir_path)

        mock.assert_called_once_with('hola')

    def test_saves_file_if_not_cached(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_loads_file_if_cached(self):
        self._write_fake_data_file()

        expected_message = 'INFO:embed.cached:{name}: loaded: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_saves_file_that_any_implementation_can_load(self):
        self.func('hola', data_dir=self._dir_path)
        message_format = 'INFO:embed.cached:{name}: loaded: {path}'

        for load_func in (cached.embed_one,
                          cached.embed_one_eu,
                          cached.embed_one_req):
            with self.subTest(load_func=load_func):
                expected_message = message_format.format(
                    name=load_func.__name__,
                    path=self._path,
                )

                with self.assertLogs(logger=cached.__name__) as log_context:
                    load_func('hola', data_dir=self._dir_path)

                self.assertEqual(log_context.output, [expected_message])

    @_audit.skip_if_unavailable
    def test_load_confirmed_by_audit_event(self):
        self._write_fake_data_file()
        expected_open_event = _audit.OpenEvent(str(self._path), 'r')

        with _audit.listening_for_open() as open_events:
            self.func('hola', data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    @_audit.skip_if_unavailable
    def test_save_confirmed_by_audit_event(self):
        # TODO: Decide whether to keep allowing just 'x', or if 'w' is OK too.
        expected_open_event = _audit.OpenEvent(str(self._path), 'x')

        with _audit.listening_for_open() as open_events:
            self.func('hola', data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    def test_saved_embedding_exists(self):
        self.func('hola', data_dir=self._dir_path)
        self.assertTrue(self._path.is_file())

    def test_uses_default_data_dir_if_not_passed(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with patch(f'{cached.__name__}.DEFAULT_DATA_DIR', self._dir_path):
            with self.assertLogs(logger=cached.__name__) as log_context:
                self.func('hola')

        self.assertEqual(
            log_context.output, [expected_message],
            'DEFAULT_DATA_DIR should be used',
        )

    @property
    def _path(self):
        """Path of temporary test file."""
        return self._dir_path / _HOLA_FILENAME

    def _write_fake_data_file(self):
        """Create a file containing a fake embedding."""
        fake_data = [1.0] + [0.0] * (embed.DIMENSION - 1)  # Normalized vector.
        with open(file=self._path, mode='w', encoding='utf-8') as file:
            json.dump(obj=fake_data, fp=file)


@parameterized_class(('name', 'func'), [
    (cached.embed_many.__name__, staticmethod(cached.embed_many)),
    (cached.embed_many_eu.__name__, staticmethod(cached.embed_many_eu)),
    (cached.embed_many_req.__name__, staticmethod(cached.embed_many_req)),
])
@_helpers.maybe_cache_embeddings_in_memory
class TestDiskCachedEmbedMany(unittest.TestCase):
    """Tests of ``embed.cached.embed_many*`` functions, which cache to disk."""

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

    # FIXME: Test returned embeddings could plausibly be correct

    def test_calls_same_name_non_caching_version_if_not_cached(self):
        with _patch_non_disk_caching_embedder(self.name) as mock:
            self.func(['hola', 'hello'], data_dir=self._dir_path)

        mock.assert_called_once_with(['hola', 'hello'])

    def test_saves_file_if_not_cached(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func(['hola', 'hello'], data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_loads_file_if_cached(self):
        self._write_fake_data_file()

        expected_message = 'INFO:embed.cached:{name}: loaded: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func(['hola', 'hello'], data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_saves_file_that_any_implementation_can_load(self):
        self.func(['hola', 'hello'], data_dir=self._dir_path)
        message_format = 'INFO:embed.cached:{name}: loaded: {path}'

        for load_func in (cached.embed_many,
                          cached.embed_many_eu,
                          cached.embed_many_req):
            with self.subTest(load_func=load_func):
                expected_message = message_format.format(
                    name=load_func.__name__,
                    path=self._path,
                )

                with self.assertLogs(logger=cached.__name__) as log_context:
                    load_func(['hola', 'hello'], data_dir=self._dir_path)

                self.assertEqual(log_context.output, [expected_message])

    @_audit.skip_if_unavailable
    def test_load_confirmed_by_audit_event(self):
        self._write_fake_data_file()
        expected_open_event = _audit.OpenEvent(str(self._path), 'r')

        with _audit.listening_for_open() as open_events:
            self.func(['hola', 'hello'], data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    @_audit.skip_if_unavailable
    def test_save_confirmed_by_audit_event(self):
        # TODO: Decide whether to keep allowing just 'x', or if 'w' is OK too.
        expected_open_event = _audit.OpenEvent(str(self._path), 'x')

        with _audit.listening_for_open() as open_events:
            self.func(['hola', 'hello'], data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    def test_saved_embedding_exists(self):
        self.func(['hola', 'hello'], data_dir=self._dir_path)
        self.assertTrue(self._path.is_file())

    def test_uses_default_data_dir_if_not_passed(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self.name,
            path=self._path,
        )

        with patch(f'{cached.__name__}.DEFAULT_DATA_DIR', self._dir_path):
            with self.assertLogs(logger=cached.__name__) as log_context:
                self.func(['hola', 'hello'])

        self.assertEqual(
            log_context.output, [expected_message],
            'DEFAULT_DATA_DIR should be used',
        )

    @property
    def _path(self):
        """Path of temporary test file."""
        return self._dir_path / _HOLA_HELLO_FILENAME

    def _write_fake_data_file(self):
        """Create a file containing a fake embedding."""
        fake_data = [1.0] + [0.0] * (embed.DIMENSION - 1)  # Normalized vector.
        with open(file=self._path, mode='w', encoding='utf-8') as file:
            json.dump(obj=fake_data, fp=file)


if __name__ == '__main__':
    unittest.main()
