from typing import List, Tuple, Union, Optional
from datetime import datetime
from multiprocessing import cpu_count

import pandas as pd
from numpy.typing import ArrayLike
from langchain.schema import Document
from IPython.display import display, HTML
from transformers import pipeline
from anonymizer.elastic.text_processing import process_pipeline
from transformers.pipelines.token_classification import TokenClassificationPipeline


def chunk_list_to_df(
    chunks: List[Document], 
    text_id: str = "text", 
    document_id: str = "document_id",
    chunk_id: str = "chunk_id"
) -> pd.DataFrame:
    """
    Turns a chunks list to a dataframe with the chunks 
    and their respective metadata. It also adds a chunk
    id and doc id column.

    args:
        chunks (list): list of langchain's Documents.
        text_id (str): the name of the text id column.
        document_id (str): the name of the document id column.
        chunk_id (str): the name of the chunk id column.

    returns:
        doc_df (pd.DataFrame): the output dataframe.
    """
    metadata_ = chunks[0].metadata.keys()
    # create a dictionary that will then be turned into a dataframe
    # with texts and their metadata
    doc_dicts = {
        text_id: [doc.page_content for doc in chunks],
        **{meta: [doc.metadata[meta] for doc in chunks] for meta in metadata_}
    }
    doc_df = pd.DataFrame(doc_dicts)
    did = doc_df[document_id]
    _ = doc_df.drop(document_id, axis=1, inplace=True)
    _ = doc_df.insert(0, document_id, did)
    # define a chunk id for each chunk belongin' to a specific 
    # document
    chunk_id_ = (
        doc_df
        .groupby(document_id)[text_id]
        .cumcount()
        .apply(lambda x: f"chunk-{x+1}")
    )
    _ = doc_df.insert(1, chunk_id, chunk_id_)
    return doc_df


def check_threads(num_threads: int = -1) -> Union[int, None]:
    num_threads = cpu_count() if num_threads == -1 else num_threads
    return num_threads


def render_presidio_results(text: str, entities: Tuple[str, int]):
    """
    Displays texts highlighting spotted entities.

    args:
        text (str): a chunk.
        entities (tuple): the entities spotted in that chunk.
    """
    marked_text = ""
    last_idx = 0
    for entity in entities:
        type_, start, end = entity
        marked_text += text[last_idx:start]
        marked_text += f"<mark style='background-color: green;'>{text[start:end]} ({type_})</mark> "
        last_idx = end
    marked_text += text[last_idx:]
    
    display(HTML(marked_text))


def get_current_time():
    current_time = datetime.now()
    return current_time.strftime("%Y-%m-%d_%H-%M-%S")


def create_document_ids(docs: ArrayLike) -> List[str]:
    return [f"doc-{n+1}" for n, _ in enumerate(docs)]


def create_documents_with_metadata(
    docs: pd.DataFrame, 
    metadata_cols: Optional[List[str]] = None,
    text_id: str = "text",
    document_id: str = "document_id"
) -> List[Document]:
    """
    Creates document with metadata from a given dataframe.

    args:
        docs (pd.DataFrame): the dataframe with the documents.
        metadata_cols (list): list of metadata.
        text_id (str): the name of the text id column.
        document_id (str): the name of the document id column.

    returns:
        dcs (list): list with langchain's Documents. 
    """
    if not isinstance(docs, pd.DataFrame):
        raise ValueError(
            "Data should be passed as a pandas dataframe, with "
            "each row representing a document, a column with "
            "documents content and optional metadata columns."
        )

    if text_id not in docs.columns:
        raise ValueError(
            "The column storing the documents content must be "
            f"named {text_id}"
        )

    # add document id
    if document_id not in docs.columns:
        docs[document_id] = create_document_ids(docs[text_id])

    # turn the dataframe into a list of Documents
    dcs = [
        Document(
            page_content=row[text_id], 
            # document id and other metadata
            metadata={
                document_id: row[document_id],
                **{meta: row[meta] for meta in metadata_cols}
            } 
            if metadata_cols else {
                document_id: row[document_id]
            }
        )
        # do this for each row of the dataframe
        for _, row in docs.iterrows()
    ]
    return dcs


def visualize_ner_on_chunk(
    chunk: Document, 
    pipeline: TokenClassificationPipeline,
    stopwords_language: str = "english"
) -> None:
    pipeline_results = pipeline(
        process_pipeline(
            pd.Series(
                [sentence.page_content for sentence in [chunk]]
            ),
            stopwords_language=stopwords_language,
            stop=1
        )
        .tolist()
    )
    entities = [(e["entity_group"], e["start"], e["end"]) for e in pipeline_results[0]]

    render_presidio_results(chunk.page_content, entities)