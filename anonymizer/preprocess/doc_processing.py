import os
from pathlib import Path
from typing import List, Dict, Tuple, Union

import pandas as pd
import numpy as np
from docx import Document


def read_word_file(path: str) -> List[str]:
    """
    Reads all `.docx` files from a given path.

    args:
        path (str): the path to the folder with word files.

    returns:
        docs (list): list of strings representing documents.
    """
    docs = []
    if os.path.splitext(path)[1] == ".docx":
        doc = Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text.lstrip())

        docs.append(
            " ".join(full_text).lstrip().replace("\xa0", " ")
        )
    return docs


def build_directory_tree(root_dir: str) -> Dict[str, Union[str, Dict]]:
    """
    Recursively builds a directory tree structure with content 
    from .docx files.

    Args:
        root_dir (str): The root directory to start building the 
        tree from.

    Returns:
        dict: A nested dictionary representing the directory tree. 
        Directories are represented as dictionaries, and .docx files 
        are represented by their content.
    """
    tree = {}
    root_dir = Path(root_dir).as_posix()

    # Iterate over each item in the root directory
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)

        # If the item is a directory, recursively build its tree
        if os.path.isdir(path):
            tree[item] = build_directory_tree(path)

        else:
            # If the item is a .docx file, read its content
            if item.endswith('.docx'):
                tree[item] = read_word_file(path)

    return tree


def process_doc_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame to extract document titles and 
    clean up the DataFrame.

    This function scans through all columns of the input DataFrame 
    to find values containing '.doc' or '.docx'. These values are 
    extracted and stored in a new column called 'title'. The original 
    cell values are replaced with NaN. Additionally, any columns that 
    become completely empty after this operation are removed from 
    the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing various data.

    Returns:
        pd.DataFrame: The processed DataFrame with a new 'title' column 
        and cleaned data.
    """
    doc_titles = []
    for column in df.columns:
        for i, value in enumerate(df[column]):
            # Check if the value contains '.doc' or '.docx'
            if '.doc' in value or '.docx' in value:
                # If it does, add the value to the doc_titles list
                doc_titles.append(value)
                # Replace the original value with NaN in the DataFrame
                df.at[i, column] = np.nan

    # Create a new column 'title' in the DataFrame with the extracted document titles
    df['title'] = pd.Series(doc_titles)
    # replace empty strings with nan
    df.replace('', np.nan, inplace=True)
    # drop any columns that are completely empty (all NaN)
    df.dropna(how='all', axis=1, inplace=True)

    return df


def extract_data(
    tree: Dict[str, Union[str, Dict]], 
    path: Tuple[None, None] = ()
) -> List[Tuple[str, str]]:
    """
    Extracts data from a nested dictionary structure, representing 
    a directory tree, and returns it as a list of tuples containing 
    the path and content.

    This function recursively traverses a dictionary (which can 
    contain other dictionaries or strings) and constructs a list of 
    tuples. Each tuple contains a path (represented as a tuple of 
    directory names) and the associated content (typically file content).

    Args:
        tree (dict): The nested dictionary representing the directory tree.
        path (tuple): The current path being traversed. Defaults to an 
        empty tuple.

    Returns:
        list: A list of tuples where each tuple contains the path to a file 
        and its content.
    """
    data = []
    # Iterate over each key-value pair in the dictionary
    for key, value in tree.items():
        # Update the current path by appending the current key
        new_path = path + (key,)

        # If the value is a dictionary, recursively extract data from it
        if isinstance(value, dict):
            data.extend(
                extract_data(value, new_path)
            )
        else:
            # If the value is not a dictionary, add the path and value to the data list
            data.append((new_path, value))

    return data


def multi_index_from_nested_dict(nested_dict: Dict[str, Union[str, Dict]]) -> pd.DataFrame:
    """
    Converts a nested dictionary into a multi-index DataFrame.

    This function takes a nested dictionary (representing a 
    directory tree), extracts the data using the `extract_data` 
    function, and constructs a DataFrame with multi-level indexing 
    based on the nested structure. Each level of the  dictionary 
    becomes a level in the DataFrame index, and the file content 
    is stored in a 'text' column. Additionally, the DataFrame is 
    processed to remove document titles and clean the data.

    Args:
        nested_dict (dict): The nested dictionary representing 
        the directory tree.

    Returns:
        pd.DataFrame: A DataFrame with multi-level indexing and 
        cleaned text data.
    """
    # Extract data from the nested dictionary into a list of tuples
    nested_list = extract_data(nested_dict)

    # Create a dictionary for multi-index data from the nested list
    multi_index_data = {
        f"level_{i}": [
            path[i] if i < len(path) else '' for path, _ in nested_list
        ]
        for i in range(max(len(path) for path, _ in nested_list))
    }
    # Add the 'text' column with file content to the dictionary
    multi_index_data["text"] = [content for _, content in nested_list]
    df = pd.DataFrame(multi_index_data)
    df["text"] = df["text"].apply(lambda x: x[0])
    
    # Clean the DataFrame: remove leading numbers and whitespace from directory/file names
    for col in df.columns.drop("text"):
        df[col] = df[col].str.replace(r"^\d{2}\.?\s?", "", regex=True)

    return process_doc_strings(df)


def process(path: str) -> pd.DataFrame:
    directory_tree = build_directory_tree(path)
    return multi_index_from_nested_dict(directory_tree)
