"""Name algorithms from their code."""

__all__ = [
    'examples',
    'get_known_names',
    'same_names',
    'generate_definition',
    'define_names',
    'get_code_text',
    'compute_similarities',
    'show_top',
]

import concurrent.futures
import datetime
import inspect
import logging
from pathlib import Path
import pprint
import re

import attrs
import backoff
import bs4
import more_itertools
import openai
import orjson
import wikipediaapi

from embed import cached
from embed.demos.whatalgo import examples

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule (``embed.demos.whatalgo``)."""


@attrs.frozen
class _Rule:
    """Rule for collecting algorithm or data structure names."""

    article_title = attrs.field()
    """The title of the article."""

    leaf_only = attrs.field()
    """Whether interest is limited names in list items WITHOUT sublists."""


_ARTICLE_RULES = [
    _Rule(article_title='List of algorithms', leaf_only=True),
    _Rule(article_title='List of data structures', leaf_only=False),
]
"""Information on Wikipedia articles listing algorithms and data structures."""

_SKIP_SECTIONS = {'See also', 'External links'}
"""Titles of sections not listing specific algorithms and data structures."""

_NAME_PATTERN = re.compile(
    r"""
    \S  # The first character can be anything that isn't whitespace.
    (?:  # Pattern for each character after the first:
        (?= [^\n])  # It is NOT allowed to be a newline, that much we know.
        (?:
            # It can be anything besides a colon, comma, or whitespace.
            [^:,\s]

            # It can even be a comma that is NOT followed by whitespace.
          | ,(?!\s)

            # It can even be whitespace, but NOT surrounding a dash.
          | \s(?![-\N{EN DASH}\N{EM DASH}]\s)
        )
    )*
    """,
    flags=(re.VERBOSE | re.UNICODE),
)
"""Regex for the "name" part of "name: summary" or several similar forms."""

_ORJSON_SAVE_OPTIONS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
"""Options for ``orjson.dumps`` when it is called to save algorithm names."""

_COMPLETION_REQUESTS_TIMEOUT = datetime.timedelta(seconds=60)
"""Connection timeout for completion-model requests."""

_COMPLETION_JOBS = 15
"""The maximum number of completion requests in progress at any one time."""


def _parse_section(section):
    """Parse the HTML of a Wikipedia article section with Beautiful Soup."""
    return bs4.BeautifulSoup(section.full_text(), features='html.parser')


def _fetch_known_names():
    """Retrieve some names of algorithms and data structures from Wikipedia."""
    wiki = wikipediaapi.Wikipedia('en', wikipediaapi.ExtractFormat.HTML)

    return sorted({
        _NAME_PATTERN.search(element.text).group()
        for rule in _ARTICLE_RULES
        for section in wiki.article(rule.article_title).sections
        if section.title not in _SKIP_SECTIONS
        for element in _parse_section(section).find_all('li', recursive=True)
        if not (rule.leaf_only and element.find('ul'))
    })


def get_known_names(data_dir=None):
    """Load names of "known" algorithms, or fetch from Wikipedia and save."""
    if data_dir is None:
        data_dir = cached.DEFAULT_DATA_DIR

    path = Path(data_dir, 'whatalgo-known-names.json')
    try:
        json_bytes = path.read_bytes()
    except FileNotFoundError:
        names = _fetch_known_names()
        path.write_bytes(orjson.dumps(names, option=_ORJSON_SAVE_OPTIONS))
        _logger.info('%s: saved: %s', get_known_names.__name__, path)
    else:
        names = orjson.loads(json_bytes)
        _logger.info('%s: loaded: %s', get_known_names.__name__, path)

    return names


def same_names(names, *, data_dir=None):
    """
    Identity function. Returns the names passed in, unchanged.

    Use this to make ``compute_similarities`` compare code to names.
    """
    del data_dir  # No need to cache anything just to pass names through.
    return names


@backoff.on_exception(
    backoff.expo,
    openai.error.ServiceUnavailableError,
    max_tries=6,  # Compare to embeddings_utils, where ALL exceptions retry 6x.
)
@backoff.on_exception(
    backoff.expo,
    openai.error.Timeout,
    max_tries=10,  # Eventually fail, in case the timeout is due to outage.
)
@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
)
def generate_definition(name):
    """
    Request a completion from ``gpt-3.5-turbo-0613`` to attempt a definition.

    This is public because it makes sense to call it to check that the output
    appears suitable. But it serves chiefly as a helper for ``define_names``.
    See the ``define_names`` docstring for an important note on accuracy.
    """
    openai_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=[
            {
                'role': 'system',
                'content': 'Define the algorithm or data structure.',
            },
            {
                'role': 'user',
                'content': name,
            },
        ],
        temperature=0,
        max_tokens=650,
        request_timeout=_COMPLETION_REQUESTS_TIMEOUT.total_seconds(),
    )
    content = openai_response.choices[0].message.content
    _logger.info('%s: received: %r', generate_definition.__name__, name)
    return content


def _generate_definition_named(name):
    """Generate a definition and return it with the name."""
    return name, generate_definition(name)


def _load_definitions(path):
    """Load saved definitions, if any, as a dict. Write a log entry."""
    try:
        json_bytes = path.read_bytes()
    except FileNotFoundError:
        old_definitions = {}
        _logger.info('%s: no file: %s', _load_definitions.__name__, path)
    else:
        old_definitions = orjson.loads(json_bytes)
        _logger.info('%s: read file: %s', _load_definitions.__name__, path)

    return old_definitions


def _save_definitions(path, definitions):
    """Save definitions, overwriting any existing ones. Write a log entry."""
    path.write_bytes(orjson.dumps(definitions, option=_ORJSON_SAVE_OPTIONS))
    _logger.info('%s: wrote file: %s', _save_definitions.__name__, path)


def _do_define_names(names, path):
    """Helper function to do most of the work for ``define_names``."""
    old_defs = _load_definitions(path)
    needed_generator = (name for name in names if name not in old_defs)
    needed_unique = list(more_itertools.unique_everseen(needed_generator))

    with concurrent.futures.ThreadPoolExecutor(_COMPLETION_JOBS) as executor:
        new_defs_map = executor.map(_generate_definition_named, needed_unique)
        new_defs = dict(new_defs_map)

    if len(new_defs) == 1:
        _logger.info('%s: %d definition generated',
                     _do_define_names.__name__, len(new_defs))
    else:
        _logger.info('%s: %d definitions generated',
                     _do_define_names.__name__, len(new_defs))

    all_defs = old_defs | new_defs

    if new_defs:
        _save_definitions(path, all_defs)
        _logger.info('%s: saved: %s', _do_define_names.__name__, path)

    return [all_defs[name] for name in names]


def define_names(names, *, data_dir=None):
    """
    Get output generated by a completion model prompted to define each name.

    Use this to make ``compute_similarities`` compare code to definitions. Note
    that the definitions may not always be correct or otherwise informative.
    The search task (see ``compute_similarities``) may tolerate some
    inaccuracies. We have not tested this combination of prompt, model, and
    inference parameters for the distinct task of generating edifying text.
    """
    if data_dir is None:
        data_dir = cached.DEFAULT_DATA_DIR
    path = Path(data_dir, 'whatalgo-definitions.json')
    return _do_define_names(list(names), path)


def get_code_text(implementation):
    """
    Get the source code text of an implementation.

    ``implementation`` may be:

    - A module object, whose ``.py`` file is to be found and read.
    - A ``Path``, representing the path to a ``.py`` file.
    - A string, holding the actual source code (not a filename).
    """
    if inspect.ismodule(implementation):
        # The code could instead be gotten from the module itself, but we want
        # whatever is in the actual file (and to fail if there isn't one).
        return Path(implementation.__file__).read_text('utf-8')

    if isinstance(implementation, Path):
        return implementation.read_text('utf-8')

    if not isinstance(implementation, str):
        raise TypeError('impl must be module, Path, or str, '
                        f'not {type(implementation)!r}')

    if '\n' not in implementation:  # This is probably a bug in the caller.
        raise ValueError(f"implementation {implementation!r} doesn't look like"
                         ' code (pass paths as Path)')

    return implementation


# TODO: Make name_describer optional, once a good choice for it is known.
def compute_similarities(*, name_describer, implementations, data_dir=None):
    """
    Compute similarities between pieces of source code and descriptions.

    Names of known algorithms and data structures are obtained by calling
    ``get_known_names``. The ``name_describer`` argument is expected to be a
    function that, when called with these names, returns suitable descriptions
    associated with them. (See ``same_names`` and ``define_names``.)

    Implementations may be provided in any way supported by ``get_code_text``.

    A matrix with a row for each implementation is returned. Each row's
    elements are similarities of that implementation to each description.
    """
    # Get the descriptions to search in, and the code to search with.
    names = get_known_names(data_dir=data_dir)
    descriptions = name_describer(names, data_dir=data_dir)
    codes = [get_code_text(impl) for impl in implementations]

    # Compute embeddings for all the descriptions and all the code.
    description_embeddings = cached.embed_many(descriptions, data_dir=data_dir)
    code_embeddings = cached.embed_many(codes, data_dir=data_dir)

    # Return a matrix of all (description, code) similarities.
    return code_embeddings @ description_embeddings.transpose()


def show_top(*, labels, similarities, count=5, data_dir=None):
    """Display the top ``count`` results from ``compute_similarities``."""
    if len(labels) != len(similarities):
        raise ValueError(f'number of labels ({len(labels)}) unequal to '
                         f'number of similarities ({len(similarities)})')

    names = get_known_names(data_dir=data_dir)

    for label, row in zip(labels, similarities):
        print(f'\n{label} example:\n')
        sorted_row = row.argsort(kind='stable')
        end_slice = slice(None, -(count + 1), -1)
        top_picks = [names[index] for index in sorted_row[end_slice]]
        pprint.pp(top_picks)
