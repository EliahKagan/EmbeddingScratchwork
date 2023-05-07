"""Name algorithms from their code."""

import logging

import bs4
import openai
import re
import wikipediaapi

_loggger = logging.getLogger(__name__)

_ARTICLE_TITLES = ['List of algorithms', 'List of data structures']
"""Titles of Wikipedia articles that list algorithms and data structures."""

_SKIP_SECTIONS = {'See also', 'External links'}
"""Titles of sections not listing specific algorithms and data structures."""

_NAME_PATTERN = re.compile(
    r'\S(?:[^:,\s]|,(?!\s)|\s(?![-\N{EN DASH}\N{EM DASH}]\s))*',
)
"""Regex for the "name" part of "name: summary" or "name, summary"."""


def get_known_names():
    """Retrieve some names of algorithms and data structures from Wikiedia."""
    wiki = wikipediaapi.Wikipedia('en', wikipediaapi.ExtractFormat.HTML)

    return sorted({
        _NAME_PATTERN.search(element.text).group()
        for article_title in _ARTICLE_TITLES
        for section in wiki.article(article_title).sections
        if section.title not in _SKIP_SECTIONS
        for element
        in bs4.BeautifulSoup(section.full_text(), features='html.parser')
              .find_all('li', recursive=True)
        if not element.find('ul')  # Filter out non-leaf list items.
    })
