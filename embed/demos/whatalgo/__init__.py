"""Name algorithms from their code."""

__all__ = ['examples', 'get_known_names', 'compute_similarities']

import inspect
import json
import logging
from pathlib import Path
import re

import attrs
import bs4
import openai
import wikipediaapi

from embed import cached
from . import examples

_logger = logging.getLogger(__name__)


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

_NAME_PATTERN = re.compile(  # TODO: Simplify this regular expression.
    r'\S(?:(?=[^\n])(?:[^:,\s]|,(?!\s)|\s(?![-\N{EN DASH}\N{EM DASH}]\s)))*',
)
"""Regex for the "name" part of "name: summary" or several similar forms."""


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


def get_known_names(dir=None):
    """Load names of "known" algorithms, or fetch from Wikipedia and save."""
    if dir is None:
        dir = cached.DEFAULT_DATA_DIR

    path = Path(dir, 'whatalgo-known-names.json')

    try:
        known_names_json = path.read_text(encoding='utf-8')
    except OSError:
        known_names = _fetch_known_names()
        content = json.dumps(known_names, indent=4) + '\n'
        path.write_text(content, encoding='utf-8')
    else:
        known_names = json.loads(known_names_json)

    return known_names


def _get_code_text(impl):
    """Get the source code text of an implementation."""
    if inspect.ismodule(impl):
        # The code could instead be gotten from the module itself, but we want
        # whatever is in the actual file (and to fail if there isn't one).
        return Path(impl.__file__).read_text('utf-8')

    if isinstance(impl, Path):
        return impl.read_text('utf-8')

    if not isinstance(impl, str):
        message = f'impl must be module, Path, or str, not {type(impl)!r}'
        raise TypeError(message)

    if '\n' not in impl:  # This is probably a bug in the caller.
        message = f"impl {impl!r} doesn't look like code (pass paths as Path)"
        raise ValueError(message)

    return impl


# TODO: Expand the docstring further, with details and/or usage guidance.
def compute_similarities(*, descriptions=None, implementations, dir=None):
    """
    Compute similarities between pieces of source code and descriptions.

    Returns a matrix with a row for each implementation. Each row's elements
    are semantic similarities of that implementation to each description.
    """
    if descriptions is None:
        descriptions = get_known_names(dir=dir)

    codes = [_get_code_text(impl) for impl in implementations]

    description_embeddings = cached.embed_many(descriptions, data_dir=dir)
    code_embeddings = cached.embed_many(codes, data_dir=dir)
    return code_embeddings @ description_embeddings.transpose()
