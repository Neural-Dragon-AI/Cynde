from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable
import logging
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .ctfidf import ClassTfidfTransformer
import collections

from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import __version__ as sklearn_version

from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
import numpy as np
import re
import polars as pl
import pandas as pd
import scipy.sparse as sp
from packaging import version
from tqdm import tqdm

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from typing import Mapping, List, Tuple
from scipy.spatial.distance import squareform


class BaseRepresentation(BaseEstimator):
    """ The base representation model for fine-tuning topic representations """
    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Each representation model that inherits this class will have
        its arguments (topic_model, documents, c_tf_idf, topics)
        automatically passed. Therefore, the representation model
        will only have access to the information about topics related
        to those arguments.

        Arguments:
            topic_model: The BERTopic model that is fitted until topic
                         representations are calculated.
            documents: A dataframe with columns "Document" and "Topic"
                       that contains all documents with each corresponding
                       topic.
            c_tf_idf: A c-TF-IDF representation that is typically
                      identical to `topic_model.c_tf_idf_` except for
                      dynamic, class-based, and hierarchical topic modeling
                      where it is calculated on a subset of the documents.
            topics: A dictionary with topic (key) and tuple of word and
                    weight (value) as calculated by c-TF-IDF. This is the
                    default topics that are returned if no representation
                    model is used.
        """
        return topic_model.topic_representations_

class MyLogger:
    def __init__(self, level):
        self.logger = logging.getLogger('BERTopic')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]

logger = MyLogger("WARNING")

class TopicAnalysisPolars:
    def __init__(self,
                 language='english',
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 nr_topics: Union[int, str] = None,
                 embedding_model=None,
                 low_memory: bool = False,
                 calculate_probabilities: bool = False,
                 seed_topic_list: List[List[str]] = None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 ctfidf_model: TfidfTransformer = None,
                 representation_model: BaseRepresentation = None,
                 verbose: bool = False,
                 ):

        if top_n_words > 100:
            logger.warning("Note that extracting more than 100 words from a sparse "
                           "can slow down computation quite a bit.")
        self.language = language
        self.top_n_words = top_n_words
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.embedding_model = embedding_model
        self.low_memory = low_memory
        self.n_gram_range = n_gram_range
        self.calculate_probabilities = calculate_probabilities
        self.verbose = verbose
        self.seed_topic_list = seed_topic_list
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=self.n_gram_range)
        self.ctfidf_model = ctfidf_model or ClassTfidfTransformer()

        # Representation model
        self.representation_model = representation_model
        self.umap_model = umap_model or UMAP(n_neighbors=15,
                                             n_components=5,
                                             min_dist=0.0,
                                             metric='cosine',
                                             low_memory=self.low_memory)
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                                              metric='euclidean',
                                                              cluster_selection_method='eom',
                                                              prediction_data=True)
        
        # Public attributes
        self.topics_ = None
        self.probabilities_ = None
        self.topic_sizes_ = None
        self.topic_mapper_ = None
        self.topic_representations_ = None
        self.topic_embeddings_ = None
        self.topic_labels_ = None
        self.custom_labels_ = None
        self.c_tf_idf_ = None
        self.representative_images_ = None
        self.representative_docs_ = {}
        self.topic_aspects_ = {}

        # Private attributes for internal tracking purposes
        self._outliers = 1
        self._merged_topics = None

        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def get_topic(self, topic: int, full: bool = False) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores

        Arguments:
            topic: A specific topic for which you want its representation
            full: If True, returns all different forms of topic representations
                  for a topic, including aspects

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Examples:

        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if topic in self.topic_representations_:
            if full:
                representations = {"Main": self.topic_representations_[topic]}
                aspects = {aspect: representations[topic] for aspect, representations in self.topic_aspects_.items()}
                representations.update(aspects)
                return representations
            else:
                return self.topic_representations_[topic]
        else:
            return False

    def get_topic_info(self, topic: int = None) -> pd.DataFrame:
        """ Get information about each topic including its ID, frequency, and name.

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            info: The information relating to either a single topic or all topics

        Examples:

        ```python
        info_df = topic_model.get_topic_info()
        ```
        """
        check_is_fitted(self)

        info = pl.DataFrame(list(self.topic_sizes_.items()), schema=["Topic", "Count"]).sort("Topic")
        info = info.with_columns(pl.col("Topic").map_dict(self.topic_labels_).alias("Name"))

        # Custom label
        if self.custom_labels_ is not None:
            if len(self.custom_labels_) == info.height:
                labels = pl.Series([label for label in self.custom_labels_]).apply(lambda x, y: x - y, y=self._outliers)
                info = info.with_columns(pl.col("Topic").map(labels).alias("CustomName"))

        # Main Keywords
        values = {topic: list(list(zip(*values))[0]) for topic, values in self.topic_representations_.items()}
        info = info.with_columns(pl.col("Topic").map_dict(values).alias("Representation"))

        # Extract all topic aspects
        '''
        if self.topic_aspects_:
            for aspect, values in self.topic_aspects_.items():
                if isinstance(list(values.values())[-1], list):
                    if isinstance(list(values.values())[-1][0], tuple) or isinstance(list(values.values())[-1][0], list):
                        values = pl.Series([list(list(zip(*value))[0]) for topic, value in values.items()])
                    elif isinstance(list(values.values())[-1][0], str):
                        values = pl.Series([(" ".join(value).strip()) for topic, value in values.items()])
                info = info.with_columns(pl.col("Topic").map(values).alias(aspect))
        '''

        # Representative Docs 
        if self.representative_docs_ is not None:
            info = info.with_columns(pl.col("Topic").map_dict(self.representative_docs_).alias("Representative_Docs"))

        # Select specific topic to return
        if topic is not None:
            info = info.filter(pl.col("Topic") == topic)

        return info.with_row_count(name="index").drop("index")
    
    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None,
                      images: List[str] = None,
                      y: Union[List[int], np.ndarray] = None) -> Tuple[List[int],
                                                                       Union[np.ndarray, None]]:
        """ Fit the models on a collection of documents, generate topics,
        and return the probabilities and topic per document.

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            images: A list of paths to the images to fit on or the images themselves
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.
        """
        if documents is not None:
            check_documents_type(documents)
            check_embeddings_shape(embeddings, documents)

        doc_ids = range(len(documents)) if documents is not None else range(len(images))
        documents = pl.DataFrame({"Document": documents,
                                  "ID": doc_ids,
                                  "Topic": None,})


        # Dubiously Reduce dimensionality
        umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)

        # Sort and Map Topic IDs by their frequency
        #if not self.nr_topics:
        #    documents = self._sort_mappings_by_frequency(documents)


        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents, embeddings=embeddings, verbose=self.verbose)

        # Reduce topics
        #if self.nr_topics:
        #    documents = self._reduce_topics(documents)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents)

        # Resulting output
        self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents['Topic'].to_list()


        return predictions, self.probabilities_
    
    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the 3 most representative docs per topic

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Updates:
            self.representative_docs_: Populate each topic with 3 representative docs
        """
        repr_docs, _, _, _ = self._extract_representative_docs(
            self.c_tf_idf_,
            documents,
            self.topic_representations_,
            nr_samples=500,
            nr_repr_docs=3
        )
        self.representative_docs_ = repr_docs
    
    def _reduce_dimensionality(self,
                            embeddings: Union[np.ndarray, csr_matrix],
                            y: Union[List[int], np.ndarray] = None,
                            partial_fit: bool = False) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.
            y: The target class for (semi)-supervised dimensionality reduction
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            umap_embeddings: The reduced embeddings
        """
        logger.info("Dimensionality - Fitting the dimensionality reduction algorithm")
        # Partial fit
        if partial_fit:
            if hasattr(self.umap_model, "partial_fit"):
                self.umap_model = self.umap_model.partial_fit(embeddings)
            elif self.topic_representations_ is None:
                self.umap_model.fit(embeddings)

        # Regular fit
        else:
            try:
                # cuml umap needs y to be an numpy array
                y = np.array(y) if y is not None else None
                self.umap_model.fit(embeddings, y=y)
            except TypeError:

                self.umap_model.fit(embeddings)

        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Dimensionality - Completed \u2713")
        return np.nan_to_num(umap_embeddings)
    
    def _cluster_embeddings(self,
                            umap_embeddings: np.ndarray,
                            documents: pl.DataFrame,
                            partial_fit: bool = False,
                            y: np.ndarray = None) -> Tuple[pl.DataFrame,
                                                           np.ndarray]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: Dataframe with documents and their corresponding IDs
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics
            probabilities: The distribution of probabilities
        """
        logger.info("Cluster - Start clustering the reduced embeddings")
        if partial_fit:
            self.hdbscan_model = self.hdbscan_model.partial_fit(umap_embeddings)
            labels = self.hdbscan_model.labels_
            documents = documents.with_columns(pl.Series(name="Topic", values=labels))
            self.topics_ = labels
        else:
            try:
                self.hdbscan_model.fit(umap_embeddings, y=y)
            except TypeError:
                self.hdbscan_model.fit(umap_embeddings)

            try:
                labels = self.hdbscan_model.labels_
            except AttributeError:
                labels = y
            documents = documents.with_columns(pl.Series(name="Topic", values=labels))
            self._update_topic_size(documents)

        # Some algorithms have outlier labels (-1) that can be tricky to work
        # with if you are slicing data based on that labels. Therefore, we
        # track if there are outlier labels and act accordingly when slicing.
        self._outliers = 1 if -1 in set(labels) else 0

        # Extract probabilities
        probabilities = None
        if hasattr(self.hdbscan_model, "probabilities_"):
            probabilities = self.hdbscan_model.probabilities_

            if self.calculate_probabilities and is_supported_hdbscan(self.hdbscan_model):
                probabilities = hdbscan_delegator(self.hdbscan_model, "all_points_membership_vectors")

        if not partial_fit:
            self.topic_mapper_ = TopicMapper(self.topics_)
        logger.info("Cluster - Completed \u2713")
        return documents, probabilities
    
    def _update_topic_size(self, documents: pd.DataFrame):
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        self.topic_sizes_ = collections.Counter(dict(documents['Topic'].value_counts().iter_rows()))
        self.topics_ = documents["Topic"].to_list()

    def _extract_embeddings(self,
                            documents: Union[List[str], str],
                            images: List[str] = None,
                            method: str = "document",
                            verbose: bool = None) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            images: A list of paths to the images to fit on or the images themselves
            method: Whether to extract document or word-embeddings, options are "document" and "word"
            verbose: Whether to show a progressbar demonstrating the time to extract embeddings

        Returns:
            embeddings: The extracted embeddings.
        """
        if isinstance(documents, str):
            documents = [documents]

        if images is not None and hasattr(self.embedding_model, "embed_images"):
            embeddings = self.embedding_model.embed(documents=documents, images=images, verbose=verbose)
        elif method == "word":
            embeddings = self.embedding_model.embed_words(words=documents, verbose=verbose)
        elif method == "document":
            embeddings = self.embedding_model.embed_documents(documents, verbose=verbose)
        elif documents[0] is None and images is None:
            raise ValueError("Make sure to use an embedding model that can either embed documents"
                             "or images depending on which you want to embed.")
        else:
            raise ValueError("Wrong method for extracting document/word embeddings. "
                             "Either choose 'word' or 'document' as the method. ")
        return embeddings

    def _create_topic_vectors(self, documents: pl.DataFrame = None, embeddings: np.ndarray = None, mappings=None):
        """ Creates embeddings per topics based on their topic representation

        As a default, topic vectors (topic embeddings) are created by taking
        the average of all document embeddings within a topic. If topics are
        merged, then a weighted average of topic embeddings is taken based on
        the initial topic sizes.

        For the `.partial_fit` and `.update_topics` method, the average
        of all document embeddings is not taken since those are not known.
        Instead, the weighted average of the embeddings of the top n words
        is taken for each topic. The weighting is done based on the c-TF-IDF
        score. This will put more emphasis to words that represent a topic best.
        """
        # Topic embeddings based on input embeddings
        if embeddings is not None and documents is not None:
            topic_embeddings = []
            topics = documents.sort("Topic")['Topic'].unique()
            for topic in topics:
                indices = documents.filter(pl.col("Topic") == topic)["ID"].to_numpy()
                indices = [int(index) for index in indices]
                topic_embedding = np.mean(embeddings[indices], axis=0)
                topic_embeddings.append(topic_embedding)
            self.topic_embeddings_ = np.array(topic_embeddings)

        # Topic embeddings when merging topics
        elif self.topic_embeddings_ is not None and mappings is not None:
            topic_embeddings_dict = {}
            for topic_from, topics_to in mappings.items():
                topic_ids = topics_to["topics_to"]
                topic_sizes = topics_to["topic_sizes"]
                if topic_ids:
                    embds = np.array(self.topic_embeddings_)[np.array(topic_ids) + self._outliers]
                    topic_embedding = np.average(embds, axis=0, weights=topic_sizes)
                    topic_embeddings_dict[topic_from] = topic_embedding

            # Re-order topic embeddings
            topics_to_map = {topic_mapping[0]: topic_mapping[1] for topic_mapping in np.array(self.topic_mapper_.mappings_)[:, -2:]}
            topic_embeddings = {}
            for topic, embds in topic_embeddings_dict.items():
                topic_embeddings[topics_to_map[topic]] = embds
            unique_topics = sorted(list(topic_embeddings.keys()))
            self.topic_embeddings_ = np.array([topic_embeddings[topic] for topic in unique_topics])
        '''
        # Topic embeddings based on keyword representations
        elif self.embedding_model is not None and type(self.embedding_model) is not BaseEmbedder:
            topic_list = list(self.topic_representations_.keys())
            topic_list.sort()

            # Only extract top n words
            n = len(self.topic_representations_[topic_list[0]])
            if self.top_n_words < n:
                n = self.top_n_words

            # Extract embeddings for all words in all topics
            topic_words = [self.get_topic(topic) for topic in topic_list]
            topic_words = [word[0] for topic in topic_words for word in topic]
            word_embeddings = self._extract_embeddings(
                topic_words,
                method="word",
                verbose=False
            )
            

            # Take the weighted average of word embeddings in a topic based on their c-TF-IDF value
            # The embeddings var is a single numpy matrix and therefore slicing is necessary to
            # access the words per topic
            topic_embeddings = []
            for i, topic in enumerate(topic_list):
                word_importance = [val[1] for val in self.get_topic(topic)]
                if sum(word_importance) == 0:
                    word_importance = [1 for _ in range(len(self.get_topic(topic)))]
                topic_embedding = np.average(word_embeddings[i * n: n + (i * n)], weights=word_importance, axis=0)
                topic_embeddings.append(topic_embedding)

            self.topic_embeddings_ = np.array(topic_embeddings)
        '''

    def _extract_topics(self, documents: pl.DataFrame, embeddings: np.ndarray = None, mappings=None, verbose: bool = False):
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            embeddings: The document embeddings
            mappings: The mappings from topic to word
            verbose: Whether to log the process of extracting topics

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        if verbose:
            logger.info("Representation - Extracting topics from clusters using representation models.")
        documents_per_topic = documents.group_by(['Topic']).agg(pl.col('Document').str.concat(" "))
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        #self.ctfidf_model, self.c_tf_idf_, words = tfidf_embed(documents_per_topic, fit=True, partial_fit=False)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)
        self._create_topic_vectors(documents=documents, embeddings=embeddings, mappings=mappings)
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in
                              self.topic_representations_.items()}
        if verbose:
            logger.info("Representation - Completed \u2713")
    
    def _extract_words_per_topic(self,
                                 words: List[str],
                                 documents: pl.DataFrame,
                                 c_tf_idf: csr_matrix = None,
                                 calculate_aspects: bool = True) -> Mapping[str,
                                                                            List[Tuple[str, float]]]:
        """ Based on tf_idf scores per topic, extract the top n words per topic

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents['Topic'].unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = _top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = _top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {label: [(words[word_index], score)
                          if word_index is not None and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}

        topics = {label: values[:self.top_n_words] for label, values in topics.items()}
        '''
        # Extract additional topic aspects
        if calculate_aspects and isinstance(self.representation_model, dict):
            for aspect, aspect_model in self.representation_model.items():
                aspects = topics.copy()
                if aspect != "Main":
                    if isinstance(aspect_model, list):
                        for tuner in aspect_model:
                            aspects = tuner.extract_topics(self, documents, c_tf_idf, aspects)
                        self.topic_aspects_[aspect] = aspects
                    elif isinstance(aspect_model, BaseRepresentation):
                        self.topic_aspects_[aspect] = aspect_model.extract_topics(self, documents, c_tf_idf, aspects)
        '''
        return topics

    def _extract_representative_docs(self,
                                     c_tf_idf: csr_matrix,
                                     documents: pl.DataFrame,
                                     topics: Mapping[str, List[Tuple[str, float]]],
                                     nr_samples: int = 500,
                                     nr_repr_docs: int = 5,
                                     diversity: float = None
                                     ) -> Union[List[str], List[List[int]]]:
        """ Approximate most representative documents per topic by sampling
        a subset of the documents in each topic and calculating which are
        most represenative to their topic based on the cosine similarity between
        c-TF-IDF representations.

        Arguments:
            c_tf_idf: The topic c-TF-IDF representation
            documents: All input documents
            topics: The candidate topics as calculated with c-TF-IDF
            nr_samples: The number of candidate documents to extract per topic
            nr_repr_docs: The number of representative documents to extract per topic
            diversity: The diversity between the most representative documents.
                       If None, no MMR is used. Otherwise, accepts values between 0 and 1.

        Returns:
            repr_docs_mappings: A dictionary from topic to representative documents
            representative_docs: A flat list of representative documents
            repr_doc_indices: Ordered indices of representative documents
                              that belong to each topic
            repr_doc_ids: The indices of representative documents
                          that belong to each topic
        """
        # Sample documents per topic
        documents = documents.with_row_index("row_idx")
        documents_per_topic_ = (
                    documents
                     .groupby('Topic')
                     .agg(
                         [pl.col("row_idx").sample(fraction=0.6, with_replacement=False, seed=1)]
                     )
                     .explode("row_idx")
        )
        documents_per_topic = (
            pl.concat(
                [documents_per_topic_, 
                 documents
                 .select(pl.all().exclude(["Topic", "row_idx"])
                 .take(documents_per_topic_["row_idx"]))], how="horizontal")
            
            )
        
        

        # Find and extract documents that are most similar to the topic
        repr_docs = []
        repr_docs_indices = []
        repr_docs_mappings = {}
        repr_docs_ids = []
        labels = sorted(list(topics.keys()))
        for index, topic in enumerate(labels):

            # Slice data using polars
            selection = documents_per_topic.filter(pl.col("Topic") == topic)
            selected_docs = selection["Document"].to_list()
            selected_docs_ids = selection['row_idx'].to_list()

            # Calculate similarity
            nr_docs = nr_repr_docs if len(selected_docs) > nr_repr_docs else len(selected_docs)
            bow = self.vectorizer_model.transform(selected_docs)
            ctfidf = self.ctfidf_model.transform(bow)
            sim_matrix = cosine_similarity(ctfidf, c_tf_idf[index])

            # Use MMR to find representative but diverse documents
            if diversity:
                docs = mmr(c_tf_idf[index], ctfidf, selected_docs, top_n=nr_docs, diversity=diversity)

            # Extract top n most representative documents
            else:
                indices = np.argpartition(sim_matrix.reshape(1, -1)[0], -nr_docs)[-nr_docs:]
                docs = [selected_docs[index] for index in indices]

            doc_ids = [selected_docs_ids[index] for index, doc in enumerate(selected_docs) if doc in docs]
            repr_docs_ids.append(doc_ids)
            repr_docs.extend(docs)
            repr_docs_indices.append([repr_docs_indices[-1][-1] + i + 1 if index != 0 else i for i in range(nr_docs)])
        repr_docs_mappings = {topic: repr_docs[i[0]:i[-1]+1] for topic, i in zip(topics.keys(), repr_docs_indices)}

        return repr_docs_mappings, repr_docs, repr_docs_indices, repr_docs_ids

    def _c_tf_idf(self,
                  documents_per_topic: pl.DataFrame,
                  fit: bool = True,
                  partial_fit: bool = False) -> Tuple[csr_matrix, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic['Document'].to_numpy())

        if partial_fit:
            X = self.vectorizer_model.partial_fit(documents).update_bow(documents)
        elif fit:
            self.vectorizer_model.fit(documents)
            X = self.vectorizer_model.transform(documents)
        else:
            X = self.vectorizer_model.transform(documents)
        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = self.vectorizer_model.get_feature_names_out()
        else:
            words = self.vectorizer_model.get_feature_names()
        
        multiplier = None
        '''
        if self.ctfidf_model.seed_words and self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([self.ctfidf_model.seed_multiplier if word in self.ctfidf_model.seed_words else 1 for word in words])
            multiplier = np.array([1.2 if word in seed_topic_list else value for value, word in zip(multiplier, words)])
        elif self.ctfidf_model.seed_words:
            multiplier = np.array([self.ctfidf_model.seed_multiplier if word in self.ctfidf_model.seed_words else 1 for word in words])
        elif self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        '''
        if fit:
            self.ctfidf_model = self.ctfidf_model.fit(X, multiplier=multiplier)

        c_tf_idf = self.ctfidf_model.transform(X)
        return c_tf_idf, words
    
    def _preprocess_text(self, documents: np.ndarray) -> List[str]:
        """ Basic preprocessing of text

        Steps:
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.replace("\n", " ") for doc in documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
        if self.language == "english":
            cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
        cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
        return cleaned_documents

    def hierarchical_topics(self,
                            docs: List[str],
                            linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                            distance_function: Callable[[csr_matrix], csr_matrix] = None) -> pd.DataFrame:
        """ Create a hierarchy of topics

        To create this hierarchy, BERTopic needs to be already fitted once.
        Then, a hierarchy is calculated on the distance matrix of the c-TF-IDF
        representation using `scipy.cluster.hierarchy.linkage`.

        Based on that hierarchy, we calculate the topic representation at each
        merged step. This is a local representation, as we only assume that the
        chosen step is merged and not all others which typically improves the
        topic representation.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            linkage_function: The linkage function to use. Default is:
                              `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
            distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                               `lambda x: 1 - cosine_similarity(x)`.
                               You can pass any function that returns either a square matrix of 
                               shape (n_samples, n_samples) with zeros on the diagonal and 
                               non-negative values or condensed distance matrix of shape
                               (n_samples * (n_samples - 1) / 2,) containing the upper
                               triangular of the distance matrix.

        Returns:
            hierarchical_topics: A dataframe that contains a hierarchy of topics
                                 represented by their parents and their children

        Examples:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        ```

        A custom linkage function can be used as follows:

        ```python
        from scipy.cluster import hierarchy as sch
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)

        # Hierarchical topics
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
        hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
        ```
        """
        check_documents_type(docs)
        if distance_function is None:
            distance_function = lambda x: 1 - cosine_similarity(x)

        if linkage_function is None:
            linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

        # Calculate distance
        embeddings = self.c_tf_idf_[self._outliers:]
        X = distance_function(embeddings)
        X = validate_distance_matrix(X, embeddings.shape[0])

        # Use the 1-D condensed distance matrix as an input instead of the raw distance matrix
        Z = linkage_function(X)

        # Calculate basic bag-of-words to be iteratively merged later
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": self.topics_})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]
        clean_documents = self._preprocess_text(documents_per_topic.Document.values)

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = self.vectorizer_model.get_feature_names_out()
        else:
            words = self.vectorizer_model.get_feature_names()

        bow = self.vectorizer_model.transform(clean_documents)

        # Extract clusters
        hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name", "Topics",
                                            "Child_Left_ID", "Child_Left_Name",
                                            "Child_Right_ID", "Child_Right_Name"])
        for index in tqdm(range(len(Z))):

            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion='distance') - self._outliers
            nr_clusters = len(clusters)

            # Extract first topic we find to get the set of topics in a merged topic
            topic = None
            val = Z[index][0]
            while topic is None:
                if val - len(clusters) < 0:
                    topic = int(val)
                else:
                    val = Z[int(val - len(clusters))][0]
            clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

            # Group bow per cluster, calculate c-TF-IDF and extract words
            grouped = csr_matrix(bow[clustered_topics].sum(axis=0))
            c_tf_idf = self.ctfidf_model.transform(grouped)
            selection = documents.loc[documents.Topic.isin(clustered_topics), :]
            selection.Topic = 0
            words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf, calculate_aspects=False)

            # Extract parent's name and ID
            parent_id = index + len(clusters)
            parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

            # Extract child's name and ID
            Z_id = Z[index][0]
            child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
            else:
                child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

            # Extract child's name and ID
            Z_id = Z[index][1]
            child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_right_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
            else:
                child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

            # Save results
            hier_topics.loc[len(hier_topics), :] = [parent_id, parent_name,
                                                    clustered_topics,
                                                    int(Z[index][0]), child_left_name,
                                                    int(Z[index][1]), child_right_name]

        hier_topics["Distance"] = Z[:, 2]
        hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
        hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]].astype(str)

        return hier_topics
    
    @staticmethod
    def get_topic_tree(hier_topics: pl.DataFrame,
                       max_distance: float = None,
                       tight_layout: bool = False) -> str:
        """ Extract the topic tree such that it can be printed

        Arguments:
            hier_topics: A dataframe containing the structure of the topic tree.
                         This is the output of `topic_model.hierachical_topics()`
            max_distance: The maximum distance between two topics. This value is
                          based on the Distance column in `hier_topics`.
            tight_layout: Whether to use a tight layout (narrow width) for
                          easier readability if you have hundreds of topics.

        Returns:
            A tree that has the following structure when printed:
                .
                .
                └─health_medical_disease_patients_hiv
                    ├─patients_medical_disease_candida_health
                    │    ├─■──candida_yeast_infection_gonorrhea_infections ── Topic: 48
                    │    └─patients_disease_cancer_medical_doctor
                    │         ├─■──hiv_medical_cancer_patients_doctor ── Topic: 34
                    │         └─■──pain_drug_patients_disease_diet ── Topic: 26
                    └─■──health_newsgroup_tobacco_vote_votes ── Topic: 9

            The blocks (■) indicate that the topic is one you can directly access
            from `topic_model.get_topic`. In other words, they are the original un-grouped topics.

        Examples:

        ```python
        # Train model
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        hierarchical_topics = topic_model.hierarchical_topics(docs)

        # Print topic tree
        tree = topic_model.get_topic_tree(hierarchical_topics)
        print(tree)
        ```
        """
        width = 1 if tight_layout else 4
        if max_distance is None:
            max_distance = hier_topics.Distance.max() + 1

        max_original_topic = hier_topics.Parent_ID.astype(int).min() - 1

        # Extract mapping from ID to name
        topic_to_name = dict(zip(hier_topics.Child_Left_ID, hier_topics.Child_Left_Name))
        topic_to_name.update(dict(zip(hier_topics.Child_Right_ID, hier_topics.Child_Right_Name)))
        topic_to_name = {topic: name[:100] for topic, name in topic_to_name.items()}

        # Create tree
        tree = {str(row[1].Parent_ID): [str(row[1].Child_Left_ID), str(row[1].Child_Right_ID)]
                for row in hier_topics.iterrows()}

        def get_tree(start, tree):
            """ Based on: https://stackoverflow.com/a/51920869/10532563 """

            def _tree(to_print, start, parent, tree, grandpa=None, indent=""):

                # Get distance between merged topics
                distance = hier_topics.loc[(hier_topics.Child_Left_ID == parent) |
                                           (hier_topics.Child_Right_ID == parent), "Distance"]
                distance = distance.values[0] if len(distance) > 0 else 10

                if parent != start:
                    if grandpa is None:
                        to_print += topic_to_name[parent]
                    else:
                        if int(parent) <= max_original_topic:

                            # Do not append topic ID if they are not merged
                            if distance < max_distance:
                                to_print += "■──" + topic_to_name[parent] + f" ── Topic: {parent}" + "\n"
                            else:
                                to_print += "O \n"
                        else:
                            to_print += topic_to_name[parent] + "\n"

                if parent not in tree:
                    return to_print

                for child in tree[parent][:-1]:
                    to_print += indent + "├" + "─"
                    to_print = _tree(to_print, start, child, tree, parent, indent + "│" + " " * width)

                child = tree[parent][-1]
                to_print += indent + "└" + "─"
                to_print = _tree(to_print, start, child, tree, parent, indent + " " * (width+1))

                return to_print

            to_print = "." + "\n"
            to_print = _tree(to_print, start, start, tree)
            return to_print

        start = str(hier_topics.Parent_ID.astype(int).max())
        return get_tree(start, tree)

    
    
    def _map_probabilities(self,
                           probabilities: Union[np.ndarray, None],
                           original_topics: bool = False) -> Union[np.ndarray, None]:
        """ Map the probabilities to the reduced topics.
        This is achieved by adding together the probabilities
        of all topics that are mapped to the same topic. Then,
        the topics that were mapped from are set to 0 as they
        were reduced.

        Arguments:
            probabilities: An array containing probabilities
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mapped_probabilities: Updated probabilities
        """
        mappings = self.topic_mapper_.get_mappings(original_topics)

        # Map array of probabilities (probability for assigned topic per document)
        if probabilities is not None:
            if len(probabilities.shape) == 2:
                mapped_probabilities = np.zeros((probabilities.shape[0],
                                                 len(set(mappings.values())) - self._outliers))
                for from_topic, to_topic in mappings.items():
                    if to_topic != -1 and from_topic != -1:
                        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]

                return mapped_probabilities

        return probabilities




def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        diversity: float = 0.1,
        top_n: int = 10) -> List[str]:
    """ Maximal Marginal Relevance

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        diversity: The diversity of the selected embeddings. 
                   Values between 0 and 1.
        top_n: The top n items to return

    Returns:
            List[str]: The selected keywords/keyphrases
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]



def check_documents_type(documents):
    """ Check whether the input documents are indeed a list of strings """
    if isinstance(documents, pd.DataFrame):
        raise TypeError("Make sure to supply a list of strings, not a dataframe.")
    elif isinstance(documents, Iterable) and not isinstance(documents, str):
        if not any([isinstance(doc, str) for doc in documents]):
            raise TypeError("Make sure that the iterable only contains strings.")
    else:
        raise TypeError("Make sure that the documents variable is an iterable containing strings only.")


def check_embeddings_shape(embeddings, docs):
    """ Check if the embeddings have the correct shape """
    if embeddings is not None:
        if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
            raise ValueError("Make sure to input embeddings as a numpy array or scipy.sparse.csr.csr_matrix. ")
        else:
            if embeddings.shape[0] != len(docs):
                raise ValueError("Make sure that the embeddings are a numpy array with shape: "
                                 "(len(docs), vector_dim) where vector_dim is the dimensionality "
                                 "of the vector embeddings. ")


def check_is_fitted(topic_model):
    """ Checks if the model was fitted by verifying the presence of self.matches

    Arguments:
        model: BERTopic instance for which the check is performed.

    Returns:
        None

    Raises:
        ValueError: If the matches were not found.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this estimator.")

    if topic_model.topics_ is None:
        raise ValueError(msg % {'name': type(topic_model).__name__})
        
def hdbscan_delegator(model, func: str, embeddings: np.ndarray = None):
    """ Function used to select the HDBSCAN-like model for generating 
    predictions and probabilities.

    Arguments:
        model: The cluster model.
        func: The function to use. Options:
                - "approximate_predict"
                - "all_points_membership_vectors"
                - "membership_vector"
        embeddings: Input embeddings for "approximate_predict"
                    and "membership_vector"
    """

    # Approximate predict
    if func == "approximate_predict":
        if isinstance(model, hdbscan.HDBSCAN):
            predictions, probabilities = hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan
            predictions, probabilities = cuml_hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        predictions = model.predict(embeddings)
        return predictions, None

    # All points membership
    if func == "all_points_membership_vectors":
        if isinstance(model, hdbscan.HDBSCAN):
            return hdbscan.all_points_membership_vectors(model)

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan
            return cuml_hdbscan.all_points_membership_vectors(model)

        return None
    
    # membership_vector
    if func == "membership_vector":
        if isinstance(model, hdbscan.HDBSCAN):
            probabilities = hdbscan.membership_vector(model, embeddings)
            return probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster.hdbscan.prediction import approximate_predict
            probabilities = approximate_predict(model, embeddings)
            return probabilities

        return None


def is_supported_hdbscan(model):
    """ Check whether the input model is a supported HDBSCAN-like model """
    if isinstance(model, hdbscan.HDBSCAN):
        return True

    str_type_model = str(type(model)).lower()
    if "cuml" in str_type_model and "hdbscan" in str_type_model:
        return True

    return False

class TopicMapper:


    """ Keep track of Topic Mappings

    The number of topics can be reduced
    by merging them together. This mapping
    needs to be tracked in BERTopic as new
    predictions need to be mapped to the new
    topics.

    These mappings are tracked in the `self.mappings_`
    attribute where each set of topic is stacked horizontally.
    For example, the most recent topics can be found in the
    last column. To get a mapping, simply take two columns
    of topics.

    In other words, it is represented as graph:
    Topic 1 --> Topic 11 --> Topic 4 --> etc.

    Attributes:
        self.mappings_ (np.ndarray) : A  matrix indicating the mappings from one topic
                                      to another. The columns represent a collection of topics
                                      at any time. The last column represents the current state
                                      of topics and the first column represents the initial state
                                      of topics.
    """
    def __init__(self, topics: List[int]):
        """ Initialization of Topic Mapper

        Arguments:
            topics: A list of topics per document
        """
        base_topics = np.array(sorted(set(topics)))
        topics = base_topics.copy().reshape(-1, 1)
        self.mappings_ = np.hstack([topics.copy(), topics.copy()]).tolist()

    def get_mappings(self, original_topics: bool = True) -> Mapping[int, int]:
        """ Get mappings from either the original topics or
        the second-most recent topics to the current topics

        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mappings: The mappings from old topics to new topics

        Examples:

        To get mappings, simply call:
        ```python
        mapper = TopicMapper(topics)
        mappings = mapper.get_mappings(original_topics=False)
        ```
        """
        if original_topics:
            mappings = np.array(self.mappings_)[:, [0, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        else:
            mappings = np.array(self.mappings_)[:, [-3, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        return mappings

    def add_mappings(self, mappings: Mapping[int, int]):
        """ Add new column(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        for topics in self.mappings_:
            topic = topics[-1]
            if topic in mappings:
                topics.append(mappings[topic])
            else:
                topics.append(-1)

    def add_new_topics(self, mappings: Mapping[int, int]):
        """ Add new row(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        length = len(self.mappings_[0])
        for key, value in mappings.items():
            to_append = [key] + ([None] * (length-2)) + [value]
            self.mappings_.append(to_append)



def validate_distance_matrix(X, n_samples):
    """ Validate the distance matrix and convert it to a condensed distance matrix
    if necessary.

    A valid distance matrix is either a square matrix of shape (n_samples, n_samples)
    with zeros on the diagonal and non-negative values or condensed distance matrix
    of shape (n_samples * (n_samples - 1) / 2,) containing the upper triangular of the
    distance matrix.

    Arguments:
        X: Distance matrix to validate.
        n_samples: Number of samples in the dataset.

    Returns:
        X: Validated distance matrix.

    Raises:
        ValueError: If the distance matrix is not valid.
    """
    # Make sure it is the 1-D condensed distance matrix with zeros on the diagonal
    s = X.shape
    if len(s) == 1:
        # check it has correct size
        n = s[0]
        if n != (n_samples * (n_samples - 1) / 2):
            raise ValueError("The condensed distance matrix must have "
                             "shape (n*(n-1)/2,).")
    elif len(s) == 2:
        # check it has correct size
        if (s[0] != n_samples) or (s[1] != n_samples):
            raise ValueError("The distance matrix must be of shape "
                             "(n, n) where n is the number of samples.")
        # force zero diagonal and convert to condensed
        np.fill_diagonal(X, 0)
        X = squareform(X)
    else:
        raise ValueError("The distance matrix must be either a 1-D condensed "
                         "distance matrix of shape (n*(n-1)/2,) or a "
                         "2-D square distance matrix of shape (n, n)."
                         "where n is the number of documents."
                         "Got a distance matrix of shape %s" % str(s))

    # Make sure its entries are non-negative
    if np.any(X < 0):
        raise ValueError("Distance matrix cannot contain negative values.")

    return X


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