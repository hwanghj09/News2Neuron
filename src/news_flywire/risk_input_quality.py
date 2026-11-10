"""Flag unfilled broadcast numbers without inventing replacements."""
import re


PLACEHOLDER=re.compile(r'(?:\d[\d가-힣 .,%]{0,12}_{2,}|_{2,}\s*(?:%|퍼센트|(?:만|억)?\s*원|달러|백선|천선))')
PRODUCTION=re.compile(r'==\s*(?:클립인|\(\(ST\)\)|\(\(CG IN\)\))\s*==')


def incomplete_numeric_draft(body):
    placeholders=len(PLACEHOLDER.findall(body))
    markers=len(PRODUCTION.findall(body))
    return bool(placeholders>=2 or (placeholders>=1 and markers>=1))
