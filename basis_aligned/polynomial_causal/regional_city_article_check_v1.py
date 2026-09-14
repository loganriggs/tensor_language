"""Experimental restriction preventing article/controlled-city interactions.

Reject *any* a/an immediately before the controlled city. This conservative
restriction avoids selecting an article that fits just one member of a pair;
it is not a general English grammaticality validator.
"""
import re


def validate_city_articles(rows):
    for row in rows:
        city=row.get('city')
        if city and re.search(r'\b(?:a|an)\s+'+re.escape(city)+r'\b',row['text'],re.I):
            raise ValueError('Indefinite article immediately before controlled city: '+row['text'])
    return dict(rows=len(rows),scope='No indefinite article immediately before controlled city')
