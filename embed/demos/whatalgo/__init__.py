"""Name algorithms from their code."""

import logging
import re

import attrs
import bs4
import openai
import wikipediaapi

_loggger = logging.getLogger(__name__)


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
"""Inormation on Wikipedia articles listing algorithms and data structures."""

_SKIP_SECTIONS = {'See also', 'External links'}
"""Titles of sections not listing specific algorithms and data structures."""

_NAME_PATTERN = re.compile(
    r'\S(?:[^:,\s]|,(?!\s)|\s(?![-\N{EN DASH}\N{EM DASH}]\s))*',
)
"""Regex for the "name" part of "name: summary" or several similar forms."""


def _parse_section(section):
    """Parse the HTML of a Wikipedia article section with Beautiful Soup."""
    return bs4.BeautifulSoup(section.full_text(), features='html.parser')


def get_known_names():
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
