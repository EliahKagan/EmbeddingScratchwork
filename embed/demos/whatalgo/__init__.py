"""Name algorithms from their code."""

import logging

import bs4
import html
import openai
import requests

_loggger = logging.getLogger(__name__)

_ARTICLE_TITLES = ['List of algorithms', 'List of data structures']
"""Titles of Wikipedia articles that list algorithms and data structures."""

_SKIP_SECTIONS = {'See also', 'External links'}
"""Titles of sections not listing specific algorithms and data structures."""


def _get_article(title):
    """Get a Wikipedia article as structured HTML with no table of contents."""
    url = f'https://en.wikipedia.org/wiki/{html.escape(title)}'
    response = requests.get(url)
    response.raise_for_status()
    doc = bs4.BeautifulSoup(response.text, features='html.parser')
    doc.find('div', attrs={'id': 'toc'}, recursive=True).decompose()
    return doc


def get_known_names():
    """Retrieve some names of algorithms and data structures from Wikipedia."""
    return [
        element.text
        for article_title in _ARTICLE_TITLES
        # if section.title not in _SKIP_SECTIONS  # FIXME: Do something here.
        for element in _get_article(article_title).find_all('li')
        if not element.find('ul')  # Filter out non-leaf list items.
    ]

    # for article_title in _ARTICLE_TITLES:
    #     for section in wiki.article(article_title).sections:
    #         if section.title in _SKIP_SECTIONS:
    #             continue
    #         yield bs4.BeautifulSoup(section.text)


    # return wiki.article('List of algorithms').sections

    #article = wiki.article('List of algorithms')
    #article = wiki.article('List of data structures')
    #[section for section in article.sections if section.title not in extra]
