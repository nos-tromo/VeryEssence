import io
import logging
from pathlib import Path
import re

import arabic_reshaper
from bertopic import BERTopic
from bidi.algorithm import get_display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords
import pandas as pd
from PIL import Image
import plotly.io as pio
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch
from umap import UMAP

from modules.file_processing import FileProcessor


class TopicModeling(FileProcessor):
    """
    This class predicts the underlying topics in the data and creates visual outputs:
    - barchart for topic overview
    - heatmap for topic correlation
    - node graph for keyword interrelations

    Inherits from:
        FileProcessor

    Attributes:
        df (pd.DataFrame): The processed dataframe from transcription/translation modules.
        column (str): The text column within the dataframe.
        language (str): The language of the text (either the source or translation language).
        device (str): The device used for model inference (CPU, CUDA, or MPS).
        model_id (str): The path to the pretrained sentence transformer model.
        topic_model (BERTopic): The BERTopic model used for topic modeling.
        topic_list (pd.DataFrame): The dataframe containing the predicted topics and number of related documents.

    Methods:
        _convert_to_language_name() -> str | None: Converts the language code to its language name for use with CountVectorizer.
        _write_file_output(data: str | list | pd.DataFrame | plt.Figure, label: str): Writes a string, DataFrame, or Figure to an output file.
        model_inference(min_topic_size: int = 5, n_gram_range: tuple[int, int] = (1, 1), label: str = "topic_modeling") -> tuple[pd.DataFrame, pd.DataFrame]:
            Runs inference on the data and predicts the underlying topics.
        topic_visual_outputs(label: str) -> Image: Creates visual outputs of the topic modeling results as a barchart or heatmap.
        process_for_graph() -> tuple[list[str], list[str]]: Preprocesses for node graph visualization by splitting topics into single keywords.
        topic_node_graph(topics: list, label: str, k: float = 0.45, figsize: tuple[float, float] = (16, 12)) -> plt.Figure:
            Visualizes the split topic labels and representations in a graph. Topic keywords are nodes, and connections between keywords are shown as edges.
    """
    def __init__(
            self,
            file_path: str,
            output_dir: str,
            data: pd.DataFrame,
            column: str,
            language: str,
            message: str
    ) -> None:
        """
        Initialize the TopicModeling object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param data: The processed dataframe from transcription/translation modules.
        :param column: The text column within the dataframe.
        :param language: The language of the text (either the source or translation language).
        :param message: A message to print when initializing the topic modeling.
        """
        super().__init__(file_path=file_path, output_dir=output_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.df = data
        self.column = column
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.topic_model = None
        self.topic_list = None

        self.logger.info(f"{message}: {self.file_path}.")
        self.logger.info(f"Device selected for topic modeling: {self.device}")

    def _convert_to_language_name(self) -> str | None:
        """
        Converts the language code to its language name to be handled by the CountVectorizer.

        :return: The language name as a string, or None if the conversion fails.
        """
        try:
            return pycountry.languages.get(alpha_2=self.language.lower()).name.lower()
        except Exception as e:
            self.logger.error(f"Error converting language: {e}.", exc_info=True)
            return None

    def _write_file_output(self, data: str | list | pd.DataFrame | plt.Figure, label: str) -> None:
        """
        Write a string, DataFrame, or Figure to an output file.

        :param data: The data to write.
        :param label: Label for the output file.
        """
        try:
            if isinstance(data, str | list | pd.DataFrame):
                super().write_file_output(data, label)
            elif isinstance(data, plt.Figure):
                plt.savefig(Path(self.output_dir) / f"{self.file_prefix}_{label}.png", bbox_inches="tight")
            else:
                self.logger.error("Error - Invalid data format.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error writing file output: {e}.", exc_info=True)
            raise

    def model_inference(
            self,
            min_topic_size: int = 5,
            n_gram_range: tuple[int, int] = (1, 1),
            label: str = "topic_modeling"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run inference on the data and predict the underlying topics.

        :param min_topic_size: Minimum number of keywords to form a topic (default: 5).
        :param n_gram_range: Range of word combination length to form a keyword (default: (1, 1)).
        :param label: Label for the output file.
        :return: A tuple containing:
            - self.topic_list: Dataframe containing the predicted topics and number of related documents.
            - topic_docs: Dataframe containing the text's documents and the topics identified with them.
        """
        try:
            docs = self.df[self.column].dropna().tolist()
            language_name = self._convert_to_language_name()
            print(f"Selected embedding model: '{self.model_id}'.")

            stop_words = stopwords.words(language_name)
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words)
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
            embedding_model = SentenceTransformer(self.model_id, device=self.device)
            embeddings = embedding_model.encode(docs, show_progress_bar=False)

            self.topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                language="multilingual",  # do not change this again ffs
                min_topic_size=min_topic_size,
                calculate_probabilities=True,
                n_gram_range=n_gram_range,
                verbose=True
            )
            topics, _ = self.topic_model.fit_transform(docs, embeddings)
            self.topic_list = self.topic_model.get_topic_info().drop(index=[0])
            topic_docs = pd.DataFrame({"topic": topics, "document": docs})
            self._write_file_output(self.topic_list, label)
            return self.topic_list, topic_docs[topic_docs.topic != -1]
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}.", exc_info=True)
            raise

    def topic_visual_outputs(self, label: str) -> Image:
        """
        Create visual outputs of the topic modeling results as a barchart or heatmap.

        :param label: Keyword to toggle barchart or heatmap creation.
        :return: An image containing either a barchart or a heatmap, or None if an error occurs.
        """
        try:
            # Check if the topic model and topic list are initialized
            if self.topic_model is None or self.topic_list is None or self.topic_list.empty:
                raise ValueError("Topic model is not initialized or there are no topics to visualize.")

            # Generate the appropriate visualization
            fig = self.topic_model.visualize_barchart() if label == "barchart" \
                else self.topic_model.visualize_heatmap() if label == "heatmap" else None

            if fig is None:
                raise ValueError(f"Invalid label '{label}' for topic visualization. Please use 'barchart' or 'heatmap'.")

            # Convert the Plotly figure to an image
            img_bytes = pio.to_image(fig, format="png")
            return Image.open(io.BytesIO(img_bytes))

        except ValueError as ve:
            self.logger.warning(f"ValueError: {ve}")
            return None  # Return None if there's an issue with visualization parameters

        except Exception as e:
            self.logger.error(f"Error creating topic visualization: {e}", exc_info=True)
            return None  # Return None to allow the program to continue even if visualization fails

    def process_for_graph(self) -> tuple[list[str], list[str]]:
        """
        Preprocess for node graph visualization by splitting topics into single keywords.

        :return: A tuple containing:
            - topic_labels: List with the split topic labels.
            - topic_representations: List with the split topic representations.
        """
        try:
            if self.topic_list is None or self.topic_list.empty:
                raise ValueError("Topic list is empty or not initialized.")

            topic_labels = self.topic_list["Name"][:40].to_list()
            topic_labels = [re.sub(r"^\d+_", "", topic) for topic in topic_labels]
            topic_representations = ["_".join(map(str, rep_list)) for rep_list in self.topic_list["Representation"]]

            return topic_labels, topic_representations

        except KeyError as ke:
            self.logger.error(f"KeyError: {ke} - Ensure the DataFrame has the correct columns.", exc_info=True)
            raise  # Reraise if it's critical to the process

        except ValueError as ve:
            self.logger.warning(f"ValueError: {ve}")
            return [], []  # Return empty lists if there's no data to process

        except Exception as e:
            self.logger.error(f"Error processing data for graph: {e}", exc_info=True)
            raise  # Reraise the exception for critical errors

    def topic_node_graph(
            self,
            topics: list,
            label: str,
            k: float = 0.45,
            figsize: tuple[float, float] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize the split topic labels and representations in a graph. Topic keywords are nodes and connections
        between keywords shown as edges.

        :param topics: Input list of topics, where each topic is a string of words joined by underscores.
        :param label: Label used in the filename for saving the plot.
        :param k: Optimal distance between nodes in the layout algorithm.
        :param figsize: Size of the figure (width, height) in inches.
        :return: Matplotlib figure of the visualized topic keywords.
        """
        try:
            G = nx.Graph()

            for topic in topics:
                words = topic.split("_")
                reshaped_words = [get_display(arabic_reshaper.reshape(word)) if self.language == "ar" else word for word in
                                  words]
                G.add_nodes_from(reshaped_words)
                for i in range(len(reshaped_words)):
                    for j in range(i + 1, len(reshaped_words)):
                        G.add_edge(reshaped_words[i], reshaped_words[j])

            pos = nx.spring_layout(G, k=k)

            fig = plt.figure(figsize=figsize)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                edge_color="gray",
                node_size=400,
                width=2,
                font_size=10
            )

            plt.title(f"Topic map for file: '{self.file_prefix}'", fontsize=18)
            self._write_file_output(fig, label)

            return fig
        except Exception as e:
            self.logger.error(f"Error creating topic node graph: {e}.", exc_info=True)
            raise
