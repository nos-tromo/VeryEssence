import logging
import unicodedata

import arabic_reshaper
from bidi.algorithm import get_display
from keybert import KeyBERT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pycountry
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from config.ar_tokenizer_cfg import ar_custom_sentence_splitter
from modules.file_processing import FileProcessor


class WordCount(FileProcessor):
    """
    This class extracts word frequencies and keywords from the input text and generates visual outputs.

    Inherits from:
        FileProcessor

    Attributes:
        df (pd.DataFrame): The processed dataframe from transcription/translation modules.
        column (str): The text column within the dataframe.
        language (str): The language of the text (either the source or translation language).
        font_path (str): Path to the font used to write the word cloud.
        all_words (list): A list to store all tokenized words.
        words_df (pd.DataFrame): A dataframe to store word frequencies.
        words_column (str): The name of the column for words in the dataframe.
        counts_column (str): The name of the column for counts in the dataframe.

    Methods:
        _arabic_sentence_splitter(text: str) -> list[str]: Splits the given Arabic text into sentences.
        _tokenize_words(text: str, language_name: str) -> list: Tokenizes the given text into a list of words.
        _convert_to_language_name() -> str | None: Converts the language code to its language name for use with CountVectorizer.
        _write_file_output(data: pd.DataFrame | list | str | plt.Figure, label: str): Writes a DataFrame, list, or string to an output file.
        split_text_into_sentences(text: str) -> list[str]: Splits the given text into sentences.
        word_frequencies(label: str = "top_words") -> None: Calculates word frequencies and keywords from the input text and stores them in a DataFrame.
        extract_keywords(label: str = "keywords") -> str: Calculates the keywords that are most descriptive for a given text.
        create_histogram(label: str = "histogram") -> plt.Figure: Creates a histogram of the word frequencies and keywords.
        create_wordcloud(label: str = "wordcloud") -> plt.Figure: Creates a wordcloud of the most frequent words.
    """
    def __init__(
            self,
            file_path: str,
            output_dir: str,
            data: pd.DataFrame,
            column: str,
            language: str,
            font_path: str,
            message: str
    ) -> None:
        """
        Initialize the WordCount object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param data: The processed dataframe from transcription/translation modules.
        :param column: The text column within the dataframe.
        :param language: The language of the text.
        :param font_path: Path to the font used to write the word cloud.
        :param message: A message to print when initializing the word count.
        """
        super().__init__(file_path=file_path, output_dir=output_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.df = data
        self.column = column
        self.language = language
        self.font_path = font_path
        self.all_words = []
        self.words_df = None
        self.words_column = "word"
        self.counts_column = "count"

        self.logger.info(f"{message}: {self.file_path}")

    def _arabic_sentence_splitter(self, text: str) -> list[str]:
        """
        Split Arabic text into sentences based on specific Arabic conjunctions and sentence boundaries.

        :param text: The Arabic text to split.
        :return: A list of sentences.
        """
        try:
            for splitter in ar_custom_sentence_splitter:
                text = text.replace(splitter, splitter + '\n')
            sentences = text.split('\n')
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception as e:
            self.logger.error(f"Error splitting Arabic text: {e}", exc_info=True)
            raise

    @staticmethod
    def _tokenize_words(text: str, language_name: str) -> list:
        """
        Tokenize the given text into a list of words.

        :param text: The text to be tokenized.
        :param language_name: The language of the text.
        :return: A list of tokenized words.
        """
        try:
            words = nltk.word_tokenize(text)
            words = [word.lower() for word in words if word.isalpha()]
            return [word for word in words if word not in stopwords.words(language_name)]
        except Exception as e:
            logging.getLogger(WordCount.__name__).error(f"Error tokenizing words: {e}", exc_info=True)
            raise

    def _convert_to_language_name(self) -> str | None:
        """
        Convert the language code to its language name to be handled by the CountVectorizer.

        :return: The language name as a string, or None if the conversion fails.
        """
        try:
            return pycountry.languages.get(alpha_2=self.language.lower()).name.lower()
        except Exception as e:
            self.logger.error(f"Error converting language: {e}", exc_info=True)
            return None

    def _write_file_output(self, data: pd.DataFrame | list | str | plt.Figure, label: str) -> None:
        """
        Write a DataFrame, list or string to an output file.

        :param data: The data to write.
        :param label: Label for the output file.
        """
        try:
            if isinstance(data, str | list | pd.DataFrame | plt.Figure):
                super().write_file_output(data, label)
            else:
                self.logger.error("Error - Invalid data format.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error writing file output: {e}", exc_info=True)
            raise

    def split_text_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences based on the language.

        :param text: The text to split.
        :return: A list of sentences.
        """
        try:
            if self.language == 'ar':
                return self._arabic_sentence_splitter(text)
            else:
                return nltk.tokenize.sent_tokenize(text)
        except Exception as e:
            self.logger.error(f"Error splitting text into sentences: {e}", exc_info=True)
            raise

    def word_frequencies(self, label: str = "top_words") -> None:
        """
        Calculate word frequencies and keywords from the input text and store them in a DataFrame.

        :param label: Label for the output file.
        """
        if self.column not in self.df.columns:
            raise ValueError(f"Column '{self.column}' does not exist in DataFrame.")

        texts = self.df[self.column].dropna().astype(str)
        vectorizer = CountVectorizer(
            tokenizer=lambda txt: self._tokenize_words(
                txt, self._convert_to_language_name()
            )
        )

        try:
            word_matrix = vectorizer.fit_transform(texts)
            word_counts = word_matrix.sum(axis=0)
            words = vectorizer.get_feature_names_out()
            counts = word_counts.tolist()[0]
            word_freq = list(zip(words, counts))
            word_freq.sort(key=lambda x: x[1], reverse=True)  # sort by frequency
            self.words_df = pd.DataFrame(word_freq[:100], columns=[self.words_column, self.counts_column])
        except Exception as e:
            self.logger.error(f"Error during vectorization: {e}", exc_info=True)
            self.words_df = pd.DataFrame(columns=[self.words_column, self.counts_column])  # returns an empty df

        self._write_file_output(self.words_df, label)

    def extract_keywords(self, label: str = "keywords") -> str:
        """
        Calculate the keywords that are most descriptive for a given text.

        :param label: Label for the output file.
        :return: A string containing the five top keywords.
        """
        try:
            docs = "".join(self.df[self.column])
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 1), stop_words=[])
            keywords_str = "\n".join(str(item[0]) for item in keywords)
            self._write_file_output(keywords_str, label)
            return keywords_str
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}", exc_info=True)
            raise

    def create_histogram(self, label: str = "histogram") -> plt.Figure:
        """
        Create a histogram of the word frequencies and keywords.

        :param label: Label for the output file.
        :return: A matplotlib figure containing a word-frequency histogram.
        """
        try:
            df = self.words_df[:20]

            fig = plt.figure(figsize=(14, 7))
            plt.bar(df[self.words_column], df[self.counts_column], color="skyblue", edgecolor="black")

            fontsize_ticks = 16
            fontsize_labels = 16

            # Handling right-to-left languages
            words = []
            for word in df[self.words_column]:
                if any("ARABIC" in unicodedata.name(char) for char in word):
                    reshaped_word = arabic_reshaper.reshape(word)  # Reshape for Arabic words
                    bidi_word = get_display(reshaped_word)  # Get the display order
                    words.append(bidi_word)
                else:
                    words.append(word)

            plt.xticks(range(len(words)), words, rotation=45, ha="right", fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)
            plt.title("Most used words (n=20)", fontsize=18)
            plt.xlabel("Words", fontsize=fontsize_labels)
            plt.ylabel("Count", fontsize=fontsize_labels)
            plt.tight_layout()
            self._write_file_output(fig, label)
            return fig
        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}", exc_info=True)
            raise

    def create_wordcloud(self, label: str = "wordcloud") -> plt.Figure:
        """
        Create a wordcloud of the most frequent words.

        :param label: Label for the output file.
        :return: A matplotlib figure containing a wordcloud of the most frequent words.
        """
        try:
            df = self.words_df.drop_duplicates()  # Ensure there are no duplicate rows in the DataFrame
            # Create a string of words for the wordcloud
            text = " ".join(df.apply(lambda row: " ".join([row[self.words_column]] * row[self.counts_column]), axis=1))
            if self.language == "ar":
                text = arabic_reshaper.reshape(text)
                text = get_display(text)
            # Generate the word cloud
            wc = WordCloud(
                font_path=self.font_path, background_color="white", width=2000, height=1000, collocations=False
            ).generate(text)

            fig = plt.figure(figsize=(20, 10), facecolor="k")
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout(pad=0)
            self._write_file_output(fig, label)
            return fig
        except Exception as e:
            self.logger.error(f"Error creating wordcloud: {e}", exc_info=True)
            raise
