import numpy as np
from typing import List, Tuple
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from cynde.analysis_tools.ctfidf import ClassTfidfTransformer
import re

def preprocessing(documents: np.ndarray, language: str = "english") -> List[str]:
    """
    Preprocesses the documents by cleaning the text, removing special characters, and handling empty documents.
    """
    cleaned_documents = [doc.replace("\n", " ") for doc in documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    if language == "english":
        cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents

@staticmethod
def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    """ Return indices of top n values in each row of a sparse matrix

    Retrieved from:
        https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

    Arguments:
        matrix: The sparse matrix from which to get the top n indices per row
        n: The number of highest values to extract from each row

    Returns:
        indices: The top n indices per row
    """
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    return np.array(indices)

@staticmethod
def _top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
    """ Return the top n values for each row in a sparse matrix

    Arguments:
        matrix: The sparse matrix from which to get the top n indices per row
        indices: The top n indices per row

    Returns:
        top_values: The top n scores per row
    """
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    return np.array(top_values)

def tfidf_embed(documents_frame: pl.DataFrame,
                contents_column: str = "Document", 
                fit: bool = True, 
                partial_fit: bool = False, 
                n_gram_range: Tuple[int, int] = (1, 1),
                top_n_words: int = 30) -> Tuple[csr_matrix, List[str]]:
    """
    Embed documents using TF-IDF representation.

    This function takes a DataFrame of documents, applies TF-IDF transformation to represent the text numerically,
    and returns the transformed matrix along with the vocabulary. It supports both fitting the model to the data
    and transforming the data using an already fitted model. It also allows for partial fitting which is useful
    for large datasets.

    Parameters:
    - documents_frame (pl.DataFrame): DataFrame containing the documents to be embedded.
    - contents_column (str, optional): Name of the column in `documents_frame` containing the text to be transformed. Defaults to "Document".
    - fit (bool, optional): Whether to fit the TF-IDF model to the data. Defaults to True.
    - partial_fit (bool, optional): Whether to partially fit the TF-IDF model to the data. Useful for large datasets. Defaults to False.
    - n_gram_range (Tuple[int, int], optional): The range of n-values for different n-grams to be extracted. Defaults to (1, 1).
    - top_n_words (int, optional): Number of top words to be extracted from the TF-IDF matrix. Defaults to 30.

    Returns:
    - Tuple[csr_matrix, List[str]]: A tuple containing the TF-IDF matrix and the list of vocabulary words.
    """
    vectorizer_model = CountVectorizer(ngram_range=n_gram_range)
    ctfidf_model = ClassTfidfTransformer()
    #if both fit and partial fit are true return error
    if fit and partial_fit:
        raise ValueError("Both fit and partial_fit cannot be True at the same time")

    documents = preprocessing(documents_frame[contents_column].to_numpy())
    if partial_fit:
        X = vectorizer_model.partial_fit(documents).update_bow(documents)
    elif fit:
        vectorizer_model.fit(documents)
        X = vectorizer_model.transform(documents)
    else:
        X = vectorizer_model.transform(documents)

    words = vectorizer_model.get_feature_names_out()
    multiplier = None
    if fit:
        ctfidf_model = ctfidf_model.fit(X, multiplier=multiplier)

    c_tf_idf = ctfidf_model.transform(X)
    return ctfidf_model, c_tf_idf, words


def extract_words_per_topic(words:List[str],documents_frame: pl.DataFrame, c_tf_idf: csr_matrix = None,top_n_words:int = 30, index_column:str = "Index", mappings=None):
    """
    Extracts and assigns the top n words and their scores per topic from a TF-IDF matrix to a DataFrame.

    This function sorts words per topic based on their scores in descending order, extracts the top n words and their scores,
    and then updates the input DataFrame with two new columns: one for the words and one for the scores.

    Args:
        words (List[str]): A list of all words in the vocabulary.
        documents_frame (pl.DataFrame): The DataFrame containing the documents and their associated topics.
        c_tf_idf (csr_matrix, optional): The TF-IDF matrix where rows correspond to documents and columns to words. Defaults to None.
        top_n_words (int, optional): The number of top words to extract per topic. Defaults to 30.
        index_column (str, optional): The name of the column in `documents_frame` that contains the topic indices. Defaults to "Index".
        mappings (optional): Additional mappings that might be required for future extensions. Defaults to None.

    Returns:
        pl.DataFrame: The updated DataFrame with two new columns: 'tfidf_words' and 'word_scores', containing the top n words and their scores per topic, respectively.
    """
    top_n_words = max(top_n_words, 30)
    labels = sorted(list(documents_frame[index_column].unique()))
    labels = [int(label) for label in labels]
    indices = _top_n_idx_sparse(c_tf_idf, top_n_words)
    scores = _top_n_values_sparse(c_tf_idf, indices)
    sorted_indices = np.argsort(scores, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(scores, sorted_indices, axis=1)

    topics = {label: [(words[word_index], score)
                          if word_index is not None and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}
    topics = {label: values[:top_n_words] for label, values in topics.items()}


    topics_list = {i:[word_tuple[0] for word_tuple in topic] for i, topic in topics.items()}
    score_list = {i:[word_tuple[1] for word_tuple in topic] for i, topic in topics.items()}
    documents_frame = documents_frame.with_columns(pl.col(index_column).replace(topics_list).alias("tfidf_words"))
    documents_frame = documents_frame.with_columns(pl.col(index_column).replace(score_list).alias("word_scores"))
    return documents_frame
    

