from dataclasses import dataclass
from pathlib import Path
import os
import re


@dataclass
class Config:
    PROJECT_PATH = Path(os.path.dirname(__file__)).parent.parent
    ANON_REGEXES = {
        # Date regex
        "date": re.compile(r"\b(\d{1,2}\s*[/,\-]\s*\d{1,2}\s*[/,\-]\s*\d{4}|\d{1,2}\s+(Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre)\s+\d{4})\b", flags=re.IGNORECASE),
        # Email regex
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+(\s*\@\s*)[A-Za-z0-9.-]+(\s*\.\s*)[A-Z|a-z]{2,}\b", re.IGNORECASE),
        # Codice Fiscale regex
        "ssn": re.compile(r"\b([A-Za-z]{6}\d{2}[A-Za-z]\d{2}[A-Za-z]\d{3}[A-Za-z]|\d{11})\b", re.IGNORECASE),
        # Partitva Iva regex
        "company_vat": re.compile(r"\b(IT)?\d{11}\b", re.IGNORECASE)
    }
    NER_MODEL = "osiria/deberta-base-italian-uncased-ner"
    ASSETS_PATH = PROJECT_PATH / "assets"
    ANON_DOCS_PATH = PROJECT_PATH / "anonymized_docs"
    HARDCODED_TODROP = []
    DOCS_PATH = PROJECT_PATH / "data"
