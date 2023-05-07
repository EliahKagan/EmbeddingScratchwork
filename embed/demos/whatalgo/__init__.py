"""Name algorithms from their code."""

import logging

import bs4
import openai
import wikipediaapi

_loggger = logging.getLogger(__name__)

_ARTICLE_TITLES = ['List of algorithms', 'List of data structures']
"""Titles of Wikipedia articles that list algorithms and data structures."""

_SKIP_SECTIONS = {'See also', 'External links'}
"""Titles of sections not listing specific algorithms and data structures."""


def get_known_names():
    """Retrieve some names of algorithms and data structures from Wikiedia."""
    wiki = wikipediaapi.Wikipedia('en', wikipediaapi.ExtractFormat.HTML)

    return [
        element.text
        for article_title in _ARTICLE_TITLES
        for section in wiki.article(article_title).sections
        if section.title not in _SKIP_SECTIONS
        for element
        in bs4.BeautifulSoup(section.full_text(), features='html.parser')
              .find_all('li', recursive=True)
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
