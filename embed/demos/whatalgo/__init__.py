"""Name algorithms from their code."""

__all__ = [
    'examples',
    'get_known_names',
    'same_names',
    'define_names',
    'get_code_text',
    'compute_similarities',
]

import inspect
import logging
from pathlib import Path
import re

import attrs
import bs4
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


def same_names(names):
    """
    Identity function. Returns the names passed in, unchanged.

    Use this to make ``compute_similarities`` compare code to names.
    """
    return names


def define_names(names):
    """
    Retrieve output from a completion model prompted to define each name.

    Use this to make ``compute_similarities`` compare code to definitions. Note
    that the definitions may not always be correct or otherwise informative.
    The search task (see ``compute_similarities``) may tolerate some
    inaccuracies. We have not tested this combination of prompt, model, and
    inference parameters for the distinct task of generating edifying text.
    """
    raise NotImplementedError  # FIXME: Implement this.


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
    descriptions = name_describer(get_known_names(data_dir=data_dir))
    codes = [get_code_text(impl) for impl in implementations]

    # Compute embeddings for all the descriptions and all the code.
    description_embeddings = cached.embed_many(descriptions, data_dir=data_dir)
    code_embeddings = cached.embed_many(codes, data_dir=data_dir)

    # Return a matrxix of all (description, code) similarities.
    return code_embeddings @ description_embeddings.transpose()
