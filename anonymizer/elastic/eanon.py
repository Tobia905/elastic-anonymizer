import pickle
import re
import os
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_anonymizer.entities.engine import OperatorConfig
from langchain.schema.document import Document
from transformers import pipeline
from faker import Faker
from textdistance import jaro_winkler
from gensim.models import FastText

from anonymizer.config.config import Config
from anonymizer.utils import (
    render_presidio_results, 
    check_threads, 
    get_current_time, 
    chunk_list_to_df
)
from anonymizer.elastic.text_processing import process_pipeline, replace_entities


# TODO: build a method "update_anon_state"
class ElasticAnonymizer:
    """
    Elastic anonymization implementation. The algorithm works by 
    following 6 steps:

    1. Using a fine-tuned BERT to perform named entity recognition 
       on sensitive information found in the corpus.
    2. The entities are, then, insterted back in the original corpus, 
       as recognized by BERT.
    3. A Fast Text model is trained on the new corpus.
    4. For each new entity that needs to be anonymized, a similarity 
       space is built as follows: the semantic and syntactic similarity 
       between the entities and the other ones found in the corpus are 
       computed and an anonymization region is defined in that space 
       (for example, the square defined by $x=(.75, 1); y=(.75, 1)$).
    5. A DBSCAN algorithm is used to spot all the entities belonging to 
       the anonymization region.
    6. All the spotted entities are anonymized using the same faking 
       strategy.

    The entities and their correspective faker are then stored in a map, 
    the anonimization state, and will be re-used for the same entities 
    or for entities that live in the same anonymization space as an already 
    stored entity.

    args:
        ner_model (str): the named entity recognition model to use.
        ner_pipeline_name (str): the name of the ner model.
        faker_random_state (int): random seed for reproducibility in the 
        mapping entity -> faking.
        num_threads_anonimize (int): number of threads to use in the 
        computation.
        faking_locale (str): the locale of the fake names.
        w2v_model_path (str): the path to a pre-trained FastText model.
        use_pretrained_anon_state (bool): if True, a pre-built anonymization
        state is used.

    instance attributes:
        kind (str): the anonymization kind (only faker is available for now).
        anon_state_name (str): the name of the anonymization state.
        regex_anon_state_name (str): the name of the anonymization state for
        regex entities.
        anonymizer (AnonymizerEngine): the anonymizer instance from presidio.
        faker (Faker): the faker instance from faker.
        selectors (list): the possible entities' kinds.
        anon_operators (dict): the anonymization operations to be performed.
        w2v_model (FastText): the FastText model to use.
    """
    def __init__(
        self: Self, 
        ner_pipeline_name: str = "ner",
        ner_model: Optional[str] = None,
        faker_random_state: int = 42,
        num_threads_anonimize: int = -1,
        faking_locale: str = "it_IT",
        w2v_model_path: Optional[str] = None,
        use_pretrained_anon_state: bool = True
    ):
        if faking_locale not in ["it_IT", "en_US", "en_GB"]:
            raise ValueError(
                "The model pipelines are only available for italian and "
                "english. Please select from ('it_IT', 'en_US', 'en_GB')."
            )
        
        self.language = (
            "italian" if faking_locale not in ["en_US", "en_GB"] else "english"
        )
        # kind of anonymization (by now, only faking is available)
        self.kind = "faker" 
        self.anon_state_name = f"{self.kind}_anonimization_state.pickle"
        self.regex_anon_state_name = f"{self.kind}_regex_anonimization_state.pickle"
        # pre trained anonimization states need to be stored in the assets/ folder
        self.use_pretrained_anon_state = use_pretrained_anon_state
        self.initialize_anon_state()
        self.anonymizer = AnonymizerEngine()

        # set the faking locale (default Italy)
        self.faker = Faker(locale=[faking_locale])
        Faker.seed(faker_random_state)
        # anonymization steps are used as anonymization functions by the OperatorConfig
        anon_steps = [
            lambda x: 
                self.find_best_faker(x) if self.find_best_faker(x) else self.anonymize_one("name"),
            lambda x: 
                self.find_best_faker(x) if self.find_best_faker(x) else self.anonymize_one("company"),
            lambda x: 
                self.find_best_faker(x) if self.find_best_faker(x) else self.anonymize_one("street_address")
        ]

        anon_steps += [lambda x: x]
        self.selectors = ["PER", "ORG", "LOC", "DEFAULT"]

        self.anon_operators = {
            ent: OperatorConfig("custom", {"lambda": func})
            for ent, func in zip(self.selectors, anon_steps)
        }
        # initialize the NER pipeline
        default_model = (
            Config.ITA_NER_MODEL 
            if faking_locale == "it_IT" 
            else Config.ENG_NER_MODEL
        )
        self.ner_pipeline = pipeline(
            ner_pipeline_name, 
            model=ner_model or default_model, 
            aggregation_strategy="simple"
        )

        self.num_threads_anonimize = num_threads_anonimize
        # import the fast text model if a pre-trained version is available
        self.w2v_model = (
            self.import_trained_embedder(w2v_model_path) if w2v_model_path else None
        )

    def find_best_faker(self: Self, word: str) -> str:
        """
        Finds the best faking for a word based on
        semantic and syntactic similarity.

        args:
            word (str): the word to be faked.

        returns:
            str or None: the best faking for the word.
        """
        # compute the similarity space for word if not already done
        sim_df = (
            self.get_similarity_space(word) 
            if not isinstance(word, pd.DataFrame) 
            else word
        )
        try:
            # get all the fakers in the anonimization region
            best_fakers = (
                sim_df
                .query("clust==-1 and sint>0.80 and sem>0.8")
                .sort_values(by="sint", ascending=False)
            )
            # get the faking associated with the most similar one
            best_faker = (
                best_fakers
                .groupby("faked")
                .count()
                .sort_values(by=0, ascending=True)
                .index[0]
            )

        except (KeyError, IndexError):
            best_faker = None

        return best_faker
    
    def get_similarity_space(self: Self, word: str) -> pd.DataFrame:
        """
        Builds the similarity space for a word.
        
        args:
            word (str): the input word.

        returns:
            pd.DataFrame: the dataframe with the two
            similarities as columns.
        """
        # compute semantic similarity (cosine) using the trained FastText
        cosine_similarities = self.w2v_model.wv.most_similar(word, topn=len(self.w2v_model.wv))
        if word in self.w2v_model.wv:
            cosine_similarities = cosine_similarities + [(word, 1.0)]

        # compute the syntactic similarity (Jaro) using the closed form formula
        jaro_similarities = {
            string: jaro_winkler(word, string) 
            for string in self.anon_state.keys()
        }
        # build the space by merging the two
        sim_df = pd.merge(
            pd.DataFrame(cosine_similarities).rename({1: "sem"}, axis=1), 
            pd.DataFrame(jaro_similarities.items()).rename({1: "sint"}, axis=1),
            on=0
        )
        # add the faking
        sim_df["faked"] = pd.merge(
            sim_df, 
            pd.DataFrame(self.anon_state.items()).rename({1: "faked"}, axis=1),
            on=0
        )["faked"]
        
        # classify the word as belonging to the anonimization region or not
        return self.classify(sim_df)

    # NOTE: the show_ner parameter could be deleted, I don't think that 
    # texts will be displayed when running the anonymization in parallel
    def _anonymize(
        self: Self, 
        pipeline_results: List[Dict[str, Union[str, int]]], 
        sentence: Document, 
        show_ner: bool = False
    ) -> pd.DataFrame:
        """
        Anonymize the given document by using the find_best_faker 
        method when a best faker is present or the anonymize_one
        when is not.

        args:
            pipeline_results: the result for the ner pipeline.
            sentence (Document): Document (langchain) containing
            the texts to anonymize.
            show_ner (bool): if True, the text with highlighted
            entities will be displayed on the screen.

        returns:
            Document: anonymized langchain's Document.
        """
        # list of tuples with entities, their start and end
        entities = [
            # we need to control for entities that include the space before the entity itself
            (e["entity_group"], e["start"] + 1, e["end"])
            # + 1 to the start index if the space is included
            if (
                len(sentence.page_content[e["start"]:e["end"]]) 
                > 
                len(sentence.page_content[e["start"]:e["end"]].lstrip())
            )
            # else just take the regular start
            else (e["entity_group"], e["start"], e["end"])
            for e in pipeline_results
        ]
        # results must be passed to the anonymizer from presidio
        # in the following form
        results = [
            RecognizerResult(
                entity_type=e[0],
                start=e[1],
                end=e[2],
                score=1.0,
            )
            for e in entities
        ]

        # add an operator for the MISC entities
        operators = { 
            "MISC" : OperatorConfig(
                "custom", {
                    "lambda" : 
                        lambda x: self.find_best_faker(x) if self.find_best_faker(x) else x
                    }
                ),
            **self.anon_operators
        }

        # text are anonymized here; the routine is handled by results and operators
        anonymized_text = self.anonymizer.anonymize(
            text=sentence.page_content,
            analyzer_results=results,
            operators=operators
        ).text

        metadata = sentence.metadata
        # entities to be anonymized with regexes
        for name, regex in Config.ANON_REGEXES.items():
            matches = set(regex.findall(anonymized_text))

            for m in matches:
                m = m[0] if isinstance(m, tuple) else m
                if m not in self.anon_state_regex.keys():
                    self.anon_state_regex[m] = self.anonymize_one(name)

                anonymized_text = re.sub(m, self.anon_state_regex[m], anonymized_text)

        if show_ner:
            render_presidio_results(sentence.page_content, entities)

        return Document(page_content=anonymized_text, metadata=metadata)

    def anonymize(
        self: Self, 
        sentences: List[Document], 
        show_ner: bool = False,
        w2v_min_count: int = 2,
        w2v_window: int = 5,
        w2v_seed: int = 42,
        hardcoded_entities: List[str] = Config.HARDCODED_TODROP,
        **w2v_kwargs: Any
    ) -> pd.DataFrame:
        """
        Applies the _anonymize method on multiple documents.

        args:
            sentences (list): list of langchain Documents to 
            be anonymized.
            show_ner (bool): if True, the text with highlighted
            entities will be displayed on the screen.
            w2v_min_count (int): the min_count parameter for the
            gensim FastText class.
            w2v_window (int): the window parameter for the
            gensim FastText class.
            w2v_seed (int): the seed parameter for the gensim
            FastText class.
            w2v_kwargs (dict): generical keyword arguments for the
            gensim FastText class.

        returns:
            pd.DataFrame: dataframe storing anonymized documents.
        """
        # get the ner pipeline results
        pipeline_results = self.ner_pipeline(
            process_pipeline(
                pd.Series(
                    [sentence.page_content for sentence in sentences]
                ),
                stopwords_language=self.language,
                stop=1
            )
            .tolist()
        )
        pipeline_results = self.filter_recognition_noise(pipeline_results)
        # populate the anonymization state if this hasn't been done yet
        if not self.use_pretrained_anon_state and not self.w2v_model:
            self.populate_anon_state(
                sentences, 
                pipeline_results, 
                w2v_min_count=w2v_min_count,
                w2v_window=w2v_window,
                w2v_seed=w2v_seed,
                hardcoded_entities=hardcoded_entities,
                **w2v_kwargs
            )

        num_threads = check_threads(self.num_threads_anonimize)
        anon_func = partial(self._anonymize, show_ner=show_ner)
        # parallelizing the anonymization
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            anon_chunks = list(
                executor.map(anon_func, pipeline_results, sentences)
            )

        # dataframe with the chunks
        return chunk_list_to_df(anon_chunks)

    def deanonymize(self: Self, sentences: List[str]) -> List[str]:
        return [self.deanonymize_one(string) for string in sentences]

    def deanonymize_one(self: Self, sentence: str) -> str:
        """
        Deanonymize an entity using inverse mapping and
        regular expressions.

        args:
            sentence (str): the sentence to be de-anonymized.

        returns:
            str: the deanonymized sentence.
        """
        sentence = re.sub(r"\.$", "  ", sentence)
        result = " " + sentence + " "
        # TODO: control for entities in both anon_states
        for anon_state in [self.anon_state, self.anon_state_regex]:
            for k, v in anon_state.items():
                v = re.sub(r"^\s|\s$", "", v)
                result = re.sub(
                    v, 
                    " " + k + " ", 
                    result, 
                    flags=re.IGNORECASE
                )

        result = re.sub(r"^\s", "", result)
        return re.sub(r"\s{2,}", " ", result)

    def anonymize_one(
        self: Self, 
        func_name: Optional[Union[str, Callable]] = None
    ) -> str:
        """
        Anonymize one sentence.

        func_name (str, Callable): the anonymization function 
        name from the operator config.

        return:
            str: anonymization to use.
        """
        # here we get the faking using the names (name, company, date, email, ...)
        anon = getattr(self.faker, func_name)()

        return " " + anon + " "

    def initialize_anon_state(self: Self, save_path: Union[Path, str] = Config.ASSETS_PATH):
        if self.is_saved_anon_state(save_path) and self.use_pretrained_anon_state:
            self.import_anon_state(path=save_path)

        else:
            self.anon_state = {}
            self.anon_state_regex = {}

    def clear_anon_state(self):
        self.anon_state = {}
        self.anon_state_regex = {}

    def populate_anon_state(
        self: Self, 
        sentences: List[Document], 
        pipeline_results: List[Dict[str, Union[str, int]]],
        w2v_min_count: int = 2,
        w2v_window: int = 5,
        w2v_seed: int = 42,
        hardcoded_entities: List[str] = Config.HARDCODED_TODROP,
        **w2v_kwargs: Any
    ) -> None:
        """
        Populates the anonymization state that stores the 
        given word and it's correspective faking.

        args:
            sentences (list): The documents containing the
            sensitive entities.
            pipeline_results (list): list containing the
            ner pipeline results.
            w2v_min_count (int): the min_count parameter for the
            gensim FastText class.
            w2v_window (int): the window parameter for the
            gensim FastText class.
            w2v_seed (int): the seed parameter for the gensim
            FastText class.
            w2v_kwargs (dict): generical keyword arguments for the
            gensim FastText class.
        """
        # get a list of sensitive entities and their typing
        words, types = [], []
        for doc in pipeline_results:
            words.append([e["word"] for e in doc])
            types.append([e["entity_group"] for e in doc])

        words_, types_ = (
            [item for sublist in words for item in sublist],
            [item for sublist in types for item in sublist]
        )
        faker_func_name_mapping = {
            s: n
            for s, n in zip(self.selectors, ["name", "company", "street_address"])
        }
        # get a dataframe with words and their type
        word_types = pd.DataFrame({"words": words_, "types": types_})
        word_types = word_types.query("types != 'MISC'").copy()
        # map the types with the words defined in faker_func_name_mapping
        word_types["faker_func_name"] = word_types["types"].map(faker_func_name_mapping)

        # process data for the FastText model: replace the entities 
        # in the original documents as recognized from the ner pipeline
        sentences = replace_entities(
            [doc.page_content for doc in sentences], 
            pipeline_results
        )
        # lower, tokenize and process the corpus to pass to FastText
        sentences = process_pipeline(
            pd.Series(sentences), stopwords_language=self.language
        )
        self.w2v_model = FastText(
            sentences=sentences, 
            min_count=w2v_min_count, 
            window=w2v_window, 
            seed=w2v_seed,
            **w2v_kwargs
        )

        word_types = word_types[word_types.words.apply(len) > 1]
 
        # anon_state is empty, initialize it with all None
        if len(self.anon_state) == 0:
            self.anon_state = {key: None for key in word_types.words.to_list()}

        # populate the anon_state with fakings
        for _, row in word_types.iterrows():
            bf = self.find_best_faker(row.words)
            # if a best faking is not in bf, use anonymize_one, else get it
            if not bf:
                self.anon_state[row.words] = self.anonymize_one(row.faker_func_name)

            else:
                self.anon_state[row.words] = bf

        # drop the hardcoded entities
        self.clean_anon_state_from_hardcoded_entities(hardcoded_entities=hardcoded_entities)

    def export_anon_state(self: Self, path: str = Config.ASSETS_PATH):
        with open((path / self.anon_state_name), "wb") as file:
            pickle.dump(self.anon_state, file)

        with open((path / self.regex_anon_state_name), "wb") as file:
            pickle.dump(self.anon_state_regex, file)

    @staticmethod
    def export_anon_df(
        df: pd.DataFrame, 
        save_path: Union[Path, str] = Config.ANON_DOCS_PATH
    ):
        save_path = ElasticAnonymizer._check_is_path(save_path)
        filename = f"anonimized_docs_{get_current_time()}.csv"
        df.to_csv((Config.ANON_DOCS_PATH / filename).as_posix(), index=None)

    def import_anon_state(
        self: Self, 
        path: Union[Path, str] = Config.ASSETS_PATH,
        hardcoded_entities: List[str] = Config.HARDCODED_TODROP
    ):
        path = ElasticAnonymizer._check_is_path(path)
        with open((path / self.anon_state_name), "rb") as file:
            self.anon_state = pickle.load(file)

        with open((path / self.regex_anon_state_name), "rb") as file:
            self.anon_state_regex = pickle.load(file)

        self.clean_anon_state_from_hardcoded_entities(hardcoded_entities=hardcoded_entities)

    @staticmethod
    def classify(similarities: pd.DataFrame) -> pd.DataFrame:
        """
        Classify words to be inside or outside the anonymization
        region of the similarity space.

        args:
            similarities (pd.DataFrame): dataframe representing the
            similarity space for a word.

        returns:
            pd.DataFrame: similarity space with a column showing the
            classification.
        """
        # scale similarities to be between 0 and 1
        similarities["sem"] = MinMaxScaler().fit_transform(similarities["sem"].values[:, None])
        db = DBSCAN(eps=0.10, min_samples=10)
        # classification is unsupervised and done by selecting the outliers 
        # spotted by the DBSCAN algorithm.
        pred = db.fit_predict(similarities[["sem", "sint"]])
        similarities["clust"] = pred
        return similarities

    def export_trained_embedder(self, save_path: Union[Path, str] = Config.ASSETS_PATH):
        save_path = ElasticAnonymizer._check_is_path(save_path)
        filename = f"trained_ft_{get_current_time()}.model"
        self.w2v_model.save((save_path / filename).as_posix())

    @staticmethod
    def import_trained_embedder(path: str) -> FastText:
        return FastText.load(path)

    def is_saved_anon_state(self, save_path: Union[Path, str] = Config.ASSETS_PATH) -> bool:
        save_path = ElasticAnonymizer._check_is_path(save_path)
        return self.anon_state_name in os.listdir(save_path)
    
    def clean_anon_state_from_hardcoded_entities(
        self, 
        hardcoded_entities: List[str] = Config.HARDCODED_TODROP
    ):
        self.anon_state = {
            ent: replacer for ent, replacer in self.anon_state.items() 
            if (
                ent not in hardcoded_entities and "##" not in ent
            )
        }

    @staticmethod
    def filter_recognition_noise(
        pipeline_results: List[List[Dict[str, Union[int, str]]]]
    ) -> List[List[Dict[str, Union[int, str]]]]:
    
        return [
            [ent for ent in anon if len(ent["word"].replace("##", "")) > 1]
            for anon in pipeline_results
        ]
    
    @staticmethod
    def _check_is_path(save_path: Union[Path, str] = Config.ANON_DOCS_PATH) -> Path:
        return (
            Path(save_path) if not isinstance(save_path, Path) else save_path
        )

    def plot_anonimization_space(
        self: Self, 
        word: str, 
        add_labels: bool = False,
        add_anonymization_region: bool = False,
        anon_region_x: Optional[Tuple[float, float]] = None,
        anon_region_y: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plots the anonymization space.

        args:
            word (str): the input word.
            add_labels (bool): if True, labels are added in the
            scatterplot.
            add_anonymization_region (bool): if True, a squared 
            representing the anonymization region is displayed.
            anon_region_x (tuple): x coordinates of the anonymization
            region.
            anon_region_y (tuple): y coordinates of the anonymization
            region.
        """
        space = self.get_similarity_space(word)
        _, ax = plt.subplots(figsize=(10, 7))
        space.plot.scatter(
            x="sem", 
            y="sint", 
            c="clust", 
            cmap="coolwarm", 
            ax=ax,
            zorder=2
        )
        if add_labels:
            # add labels to the points (the entities)
            for _, v in space.query("clust==-1").iterrows():
                ax.text(v["sem"], v["sint"], v[0][:10], alpha=.5, ha="center")

        if add_anonymization_region:
            if not anon_region_x or not anon_region_y:
                raise ValueError(
                    "If You want to show the anonymization region, "
                    "you should pass x and y vertices coordinates."
                )
            
            # compute width and height of the square
            width = anon_region_x[1] - anon_region_x[0]
            height = anon_region_y[1] - anon_region_y[0]

            # add the rectangle
            rect = patches.Rectangle(
                (anon_region_x[0], anon_region_y[0]), 
                width, 
                height, 
                linewidth=1, 
                edgecolor='r', 
                facecolor="none"
            )
            _ = ax.add_patch(rect)

        _ = ax.grid(alpha=.3, zorder=-1)
        _ = ax.set(
            xlabel="Semantic Similarity", 
            ylabel="Syntactic Similarity",
            title=f"Similarity Space for {word}"
        )
    