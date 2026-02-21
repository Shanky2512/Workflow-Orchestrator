from typing import Dict

from .parsers.base import BaseParser
from .parsers.json_parser import JsonParser
from .parsers.pdf_parser import PdfParser
from .parsers.xml_parser import XmlParser

PARSER_REGISTRY: Dict[str, BaseParser] = {
    "application/pdf": PdfParser(),
    "application/json": JsonParser(),
    "text/json": JsonParser(),
    "application/xml": XmlParser(),
    "text/xml": XmlParser(),
}
