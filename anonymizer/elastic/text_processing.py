from functools import reduce, partial
from typing import List, Union, Optional, Dict, Any

import nltk
nltk.download("stopwords")

import pandas as pd
import numpy as np
from nltk.corpus import stopwords


def to_lower(text: pd.Series) -> pd.Series:
    return text.str.lower()


def strip_punctuation(text: pd.Series) -> pd.Series:
    puncts = r"(?<!§)§(?!§)|[^\w\s§]"
    return text.str.replace(puncts, "", regex=True)


def remove_stopwords(text: pd.Series, language: str) -> pd.Series:
    stopwords_ = stopwords.words(language)
    stop_reg = r"\b(" + "|".join(stopwords_) + ")\\W"
    return text.str.replace(stop_reg, "", regex=True)


def splitter(text: pd.Series) -> pd.Series:
    return text.str.split()


def char_replacer(text: pd.Series) -> pd.Series:
    return text.apply(
        lambda x: [
            string.replace("§§", " ") for string in x
        ]
    )


def double_space_replace(text: pd.Series) -> pd.Series:
    return text.str.replace(r"\s{2,}", " ", regex=True)


def remove_stopwords(text: pd.Series, language: str) -> pd.Series:
    stopwords_ = stopwords.words(language)
    return text.apply(
        lambda x: [
            "" if string in stopwords_ else string for string in x
        ]
    )


def remove_line_splitted(text: pd.Series) -> pd.Series:
    return text.str.replace(r"##\w+", "", regex=True)


def postprocess(text: pd.Series) -> pd.Series:
    return text.apply(
        lambda x: list(
            filter(
                len, 
                ["" if len(string) == 1 else string for string in x]
            )
        )
    )


def process_pipeline(
    text: Union[pd.Series, List[str], np.ndarray], 
    stopwords_language: str,
    stop: Optional[int] = None
) -> List[str]:
    """
    Text processing pipeline. It just applies all the
    processing steps in series.

    args:
        text (pd.Series, list, np.ndarray): array-like 
        element with the texts to process.
        stop (int): the step at which the pipeline should
        stop.

    returns:
        list: the processed texts.
    """
    text = (
        pd.Series(text) 
        if not isinstance(text, pd.Series) else text
    )
    rem_stops = partial(remove_stopwords, language=stopwords_language)
    funcs = [
        to_lower,
        remove_line_splitted,
        double_space_replace,
        splitter,
        char_replacer,
        rem_stops,
        postprocess
    ]
    funcs = funcs if not stop else funcs[:stop]

    return reduce(
        lambda value, func: func(value), funcs, text
    )


# chatGPT helped a lot here
def replace_entities(texts: List[str], entities: List[Dict[str, Any]]) -> List[str]:
    """
    Replaces entites in the corpus as recognized by the
    named entity recognition model.

    args:
        texts (list): the list with the texts (chunks) that 
        define the corpus.
        entites (list): the spotted entities for each text (chunk).

    returns:
        text_ (list): list with processed texts (chunks).
    """
    texts_ = []
    sele = ["PER", "ORG", "LOC", "DEFAULT"]

    for text, entity in zip(texts, entities):
        # just a control for empty entities
        if len(entity) == 0:
            text = text

        else:
            for ent in reversed(entity):
                if ent['entity_group'] in sele:
                    # replace spaces in the entities with "§§"
                    entity_token = "§§".join(
                        text[ent['start']:ent['end']].split()
                    )
                    # substitute the text in the corpus with the
                    # entities
                    text = (
                        text[:ent['start']] 
                        + " " + entity_token 
                        + " " + text[ent['end']:]
                    )
        texts_.append(text)
    return texts_
