import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from datetime import datetime
from pathlib import Path

logfile_name = f"veryessence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logs_dir = Path(".logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()
    ]
)

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from reportlab.platypus import KeepTogether, Spacer

from modules.file_processing import FileProcessor
from modules.llm_analysis import LLMAnalysis
from modules.report_builder import PDFReportBuilder
from modules.speaker_diarization import SpeakerDiarization
from modules.topic_modeling import TopicModeling
from modules.whisper_transcription import WhisperTranscriber
from modules.word_counting import WordCount


def parse_arguments(args_list: list = None) -> argparse.Namespace:
    """
    VeryEssence pipelines ML models to transcribe, translate, and analyze natural language from audio/video.

    :param args_list: A list of arguments from the GUI to simulate command line input.
    :return: Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files. Specify the path to the audio file or a YouTube URL and configure "
                    "transcription tasks.",
        epilog="""
        Examples:
        python cli.py -f audio.mp3 -s 00:00:30 -e 00:05:00 -sl en -m large -t 1 -a 
        python cli.py -f lecture.wav --start 00:10:00 --end 00:45:00 --source-lang es --model base --task 3
        python cli.py -f - video.mp4 -F
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-f", "--file", dest="file_path", type=str, required=True,
        help="Specify the file path and name of the audio file to be transcribed."
    )
    parser.add_argument(
        "-s", "--start", dest="start_time", type=str, default="0",
        help="Set the starting timecode for transcription in hh:mm:ss format (default: 0)."
    )
    parser.add_argument(
        "-e", "--end", dest="end_time", type=str, default="0",
        help="Set the ending timecode for transcription in hh:mm:ss format (default: 0)."
    )
    parser.add_argument(
        "-sl", "--src-lang", dest="source_language", type=str, default="de",
        help="Specify the language code (ISO 639-1) of the source audio (default: 'de')."
    )
    parser.add_argument(
        "-tl", "--tgt-lang", dest="target_language", type=str, default="de",
        help="Specify the language code (ISO 639-1) of the target language (default: 'de')."
    )
    parser.add_argument(
        "-m", "--model", dest="model_size", type=str, default="default",
        help="Choose the model checkpoint for processing (sizes: tiny, base, small, medium, large-v2, large-v3, default; default: default)."
    )
    parser.add_argument(
        "-t", "--task", dest="task", type=int, default=1,
        help="Select from 0 (language detection), 1 (transcription), 2 (translation), or 3 (transcription & translation) (default: 1)."
    )
    parser.add_argument(
        "-w", "--words", dest="words", action="store_true",
        help="Show most frequently used words (default: False)."
    )
    parser.add_argument(
        "-ta", "--topics", dest="topics", action="store_true",
        help="Enable topic modeling analysis (default: False)."
    )
    parser.add_argument(
        "-ner", "--entities", dest="entities", action="store_true",
        help="Enable named entity recognition (default: False)."
    )
    parser.add_argument(
        "-sent", "--sentiment", dest="sentiment", action="store_true",
        help="Enable sentiment analysis (default: False)."
    )
    parser.add_argument(
        "-d", "--diarization", dest="diarization", action="store_true",
        help="Enable speaker diarization (default: False)."
    )
    parser.add_argument(
        "-sum", "--summarization", dest="summarization", action="store_true",
        help="Additional text and topic summarization (default: False)."
    )
    parser.add_argument(
        "-hdr", "--header", dest="header", type=str, default=None,
        help="Add a header to the report."
    )
    parser.add_argument(
        "-u", "--user", dest="user", type=str, default=None,
        help="Add a username to the report."
    )
    parser.add_argument(
        "-o", "--output", dest="output_directory", type=str, default="output",
        help="Specify the output directory (default: output)."
    )
    parser.add_argument(
        "-F", "--full-analysis", dest="full_analysis", action="store_true",
        help="Enable full analysis, equivalent to using -w -ta -ner -sent -d -sum (default: False)."
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if args.full_analysis:
        args.words = True
        args.topics = True
        args.entities = True
        args.sentiment = True
        args.diarization = True
        args.summarization = True

    return args


def _file_handling(
        args: argparse.Namespace,
        message: str = "processing".upper()
) -> tuple[FileProcessor, int, str, int, str]:
    """
    Handle file processing and preparation.

    :param args: Parsed command-line arguments.
    :param message: Message to print during file processing.
    :return: A tuple containing the FileProcessor object, file name, file prefix, start time in
             seconds, temporary file path, and file length in seconds.
    """
    logging.info(f"{message}: {args.file_path}")
    file_processor = FileProcessor(args.file_path, args.output_directory, args.start_time, args.end_time)
    file_start_timecode_seconds, tempfile, file_length = file_processor.audio_processing()
    file_length_hms = file_processor.hms(file_length)
    return file_processor, file_start_timecode_seconds, tempfile, file_length, file_length_hms


def _detect_language(
        args: argparse.Namespace,
        file: str,
        file_start_timecode_seconds: int,
        column: str,
        label: str = "detect",
        message: str = "language detection".upper()
) -> tuple[str, str, str]:
    """
    Detect the language of the input audio file.

    :param args: Parsed command-line arguments.
    :param file: Path to the temporary audio file.
    :param file_start_timecode_seconds: Start time in seconds.
    :param column: Column name for the transcription result.
    :param label: Label for the transcription task.
    :param message: Message to print during language detection.
    :return language: A text example, the detected language of the audio file, and the model path.
    """
    args.source_language = "unknown"
    args.model_size = "tiny"
    whisper_detector = WhisperTranscriber(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.source_language,
        model_size=args.model_size,
        file=file,
        task=label,
        start_time_seconds=file_start_timecode_seconds,
        text_column=column,
        message=message
    )
    example_text, detected_language = whisper_detector.data_pipeline()
    return example_text, detected_language, whisper_detector.model_id


def _transcribe_audio(
        args: argparse.Namespace,
        file: str,
        file_start_timecode_seconds: int,
        column: str,
        label: str = "transcribe",
        message: str = "speech-to-text transcription".upper()
) -> tuple[pd.DataFrame, str]:
    """
    Transcribe/translate the audio file with Whisper.

    :param args: Parsed command-line arguments.
    :param file: Path to the temporary audio file.
    :param file_start_timecode_seconds: Start time in seconds.
    :param label: Label for the transcription task.
    :param message: Message to print during transcription.
    :param column: Column name for the transcription result.
    :return: A tuple containing the transcribed DataFrame and the transcription model id.
    """
    whisper_transcriber = WhisperTranscriber(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.source_language,
        model_size=args.model_size,
        file=file,
        task=label,
        start_time_seconds=file_start_timecode_seconds,
        text_column=column,
        message=message
    )
    df = whisper_transcriber.data_pipeline()
    return df, whisper_transcriber.model_id


def _translate_text(
        args: argparse.Namespace,
        data: pd.DataFrame,
        input_column: str,
        output_column: str,
        message: str = "text-to-text translation".upper()
) -> tuple[pd.DataFrame, str]:
    """
    Translate the DataFrame using the LLM model.

    :param args: Parsed command-line arguments.
    :param message: Message to print during translation.
    :param data: Dataframe containing the transcription.
    :param input_column: Column name for the transcribed input.
    :param output_column: Column name for the translation result.
    :return: A tuple containing the translated DataFrame and the machine translation model path.
    """
    text_translation = LLMAnalysis(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.target_language,
        task="translation",
        message=message
    )
    translated_df = text_translation.data_pipeline(
        data=data,
        text_column=input_column,
        results_column=output_column
    )
    return translated_df, text_translation.model_id


def _analysis_data_pipeline(
        translated_df: pd.DataFrame,
        transcribed_df: pd.DataFrame,
        mt_column: str,
        whisper_column: str,
        target_language: str,
        source_language: str
) -> tuple[pd.DataFrame, str, str]:
    """
    Set up the data pipeline for analysis.

    :param translated_df: DataFrame with translated data.
    :param transcribed_df: DataFrame with transcribed data.
    :param mt_column: Column name for the translated data.
    :param whisper_column: Column name for the transcribed data.
    :param target_language: Target language for translation.
    :param source_language: Source language for transcription.
    :return: A tuple containing the analysis DataFrame, analysis column, and analysis language.
    """
    analysis_df = translated_df if translated_df is not None else transcribed_df
    analysis_column = mt_column if translated_df is not None and target_language != "en" else whisper_column
    analysis_language = target_language if translated_df is not None else source_language
    return analysis_df, analysis_column, analysis_language


def _extract_from_text(
        args: argparse.Namespace,
        data: pd.DataFrame,
        column: str,
        message: str,
        task: str
) -> tuple[str | pd.DataFrame | None, str, int | None]:
    """
    Process the transcribed/translated text using LLMs.

    :param args: Parsed command-line arguments.
    :param message: Additional message or information to be used during the processing.
    :param task: Task label for the LLM processing.
    :param data: DataFrame containing the text data to be summarized or extracted.
    :param column: Name of the column in data that contains the text to be summarized or extracted.
    :return: A tuple containing the results, result and sentence lengths, and the model used for the inference.
    """
    text_analysis = LLMAnalysis(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.target_language,
        task=task,
        message=message
    )
    result = text_analysis.data_pipeline(
        data=data,
        text_column=column
    )

    if task == "entities" and isinstance(result, pd.DataFrame) and result is not None:
        result = result.iloc[:, :2]
        result.columns = result.columns.str.capitalize()
        n_results = len(result)
    else:
        n_results = None

    return result, text_analysis.model_id, n_results


def _word_statistics(
        args: argparse.Namespace,
        data: pd.DataFrame,
        column: str,
        language: str,
        font_path: str = "static/fonts/Amiri-Regular.ttf",
        message: str = "word statistics".upper()
) -> tuple[int | None, str, plt.Figure, plt.Figure, str]:
    """
    Perform word counting and keyword extraction.

    :param args: Parsed command-line arguments.
    :param data: DataFrame with the data to analyse.
    :param column: Column name for the text data.
    :param language: Language of the text data.
    :param font_path: Path to the font used to write the word cloud.
    :param message: Message to print during word counting.
    :return: A tuple containing the number of sentences in the text, extracted keywords, histogram figure, wordcloud
    figure, and the keyword model used.
    """
    word_count = WordCount(
        file_path=args.file_path,
        output_dir=args.output_directory,
        data=data,
        column=column,
        language=language,
        font_path=font_path,
        message=message
    )

    sentences = word_count.split_text_into_sentences("".join(data[column]))
    if len(sentences) > 0:
        n_sentences = len(sentences)
    else:
        n_sentences = None

    word_count.word_frequencies()
    keywords = word_count.extract_keywords()
    words_histogram = word_count.create_histogram()
    wordcloud_fig = word_count.create_wordcloud()
    keyword_model = "KeyBERT"
    return n_sentences, keywords, words_histogram, wordcloud_fig, keyword_model


def _topic_analysis(
        args: argparse.Namespace,
        data: pd.DataFrame,
        column: str,
        source_language: str,
        file_length: float,
        topic_modeling_message: str = "topic modeling".upper(),
        topic_summarization_message: str = "generating titles for topics".upper()
) -> tuple[pd.DataFrame, plt.Figure, str, str]:
    """
    Perform topic modeling analysis.

    :param args: Parsed command-line arguments.
    :param data: DataFrame with the data to analyze.
    :param column: Column name for the text data.
    :param source_language: Language of the text data.
    :param file_length: Length of the audio file in seconds.
    :param topic_modeling_message: Message to print during topic analysis.
    :param topic_summarization_message: Message to print during summarization.
    :return: A tuple containing the top topics DataFrame, topic node graph, and the models (topic modeling and summarization) used.
    """
    topic_modeling = TopicModeling(
        file_path=args.file_path,
        output_dir=args.output_directory,
        data=data,
        column=column,
        language=source_language,
        message=topic_modeling_message
    )
    topic_df, _ = topic_modeling.model_inference()
    topic_labels, topic_representations = topic_modeling.process_for_graph()
    topic_map_labels = topic_modeling.topic_node_graph(topic_labels, "topic_map_labels")
    topic_map_representations = topic_modeling.topic_node_graph(topic_representations, "topic_map_representations")
    top_topics_df = topic_df[:10][["Name", "Representative_Docs"]].rename(columns={"Name": "Topic", "Representative_Docs": "Description"})
    topic_graph = topic_map_representations if file_length < 3600 and topic_map_representations is not None \
        else topic_map_labels

    # generate named labels and summaries for each topic
    keywords_column = top_topics_df.columns[0]  # extract the column name where the keyword names are stored
    documents_column = top_topics_df.columns[1]  # extract the column name where the example docs are stored
    summarized_topics_df = None
    llm_inference_model = None
    if args.summarization and len(top_topics_df) > 0:
        label_summarization = LLMAnalysis(
            file_path=args.file_path,
            output_dir=args.output_directory,
            language=args.target_language,
            task="topic summary",
            message=topic_summarization_message
        )
        summarized_topics_df = label_summarization.data_pipeline(
            data=top_topics_df,
            keywords_column=keywords_column,
            documents_column=documents_column
        )
        llm_inference_model = label_summarization.model_id

    updated_topics_df = (summarized_topics_df if summarized_topics_df is not None else top_topics_df).iloc[:, :3]
    return updated_topics_df, topic_graph, topic_modeling.model_id, llm_inference_model


def _sentiment_analysis(
        args: argparse.Namespace,
        message: str,
        task: str,
        data: pd.DataFrame,
        column: str,
) -> tuple[pd.DataFrame, str, int]:
    """
    Performs sentiment analysis on the provided data using a language model and returns negative sentiment results.

    :param args: Command-line arguments containing file path, output directory, and target language information.
    :param message: A message or prompt for the sentiment analysis model.
    :param task: The task type, typically indicating the column for storing sentiment scores.
    :param data: The input DataFrame containing the data to be analyzed.
    :param column: The column in "data" that contains the text for sentiment analysis.
    :return: A tuple containing:
             - A DataFrame with rows of negative sentiment, including only start, sentiment, and text columns.
             - The model ID used for the analysis.
             - The total number of results in the original analysis output.
    """
    sentiment_analysis = LLMAnalysis(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.target_language,
        task=task,
        message=message
    )
    result = sentiment_analysis.data_pipeline(
        data=data,
        text_column=column,
        results_column=task
    )
    result.columns = result.columns.str.capitalize()
    neg_sent_result = result[result[task.capitalize()] <= 0].sort_values(by=task.capitalize()).iloc[:, [0, 2, 3]][:20]  # drop values above 0, sort, reorganize columns, use only top 20 values
    n_results = len(result)
    return neg_sent_result, sentiment_analysis.model_id, n_results


def _speaker_detection(
        args: argparse.Namespace,
        file: str,
        message: str = "speaker diarization".upper()
) -> tuple[int, plt.Figure, str]:
    """
    Perform speaker diarization on the audio file.

    :param args: Parsed command-line arguments.
    :param file: Path to the audio file.
    :param message: Message to print during speaker diarization.
    :return: A tuple containing the speaker count, speaker diarization figure, and the model path.
    """
    speaker_diarization = SpeakerDiarization(
        file_path=args.file_path,
        output_dir=args.output_directory,
        file=file,
        message=message
    )
    speaker_count, speaker_fig = speaker_diarization.apply_diarization()
    return speaker_count, speaker_fig, speaker_diarization.model_id


def _list_utilized_models(
        language_detection_model: str = None,
        whisper_transcription_model: str = None,
        whisper_translation_model: str = None,
        machine_translation_model: str = None,
        text_summarization_model: str = None,
        keyword_model: str = None,
        topic_modeling_model: str = None,
        topic_summarization_model: str = None,
        entities_model: str = None,
        sentiment_model: str = None,
        speaker_diarization_model: str = None
) -> pd.DataFrame:
    """
    List the models utilized during the various tasks.

    :param language_detection_model: Model used for language detection.
    :param whisper_transcription_model: Model used for transcription.
    :param whisper_translation_model: Model used for translation.
    :param machine_translation_model: Model used for text-to-text translation.
    :param text_summarization_model: Model used for text and topic label summarization.
    :param keyword_model: Model used for keyword extraction.
    :param topic_modeling_model: Model used for topic modeling.
    :param topic_summarization_model: Model used for creating titles from topic keywords.
    :param entities_model: Model used for named entity recognition.
    :param sentiment_model: Model used for sentiment analysis.
    :param speaker_diarization_model: Model used for speaker diarization.
    :return: A DataFrame listing the tasks and the corresponding models used.
    """
    models_dict = {}

    if language_detection_model:
        models_dict["Language Detection"] = language_detection_model
    if whisper_transcription_model:
        models_dict["Transcription (Speech-to-Text)"] = whisper_transcription_model
    if whisper_translation_model:
        models_dict["Translation (Speech-to-Text)"] = whisper_translation_model
    if machine_translation_model:
        models_dict["Translation (Text-to-Text)"] = machine_translation_model
    if text_summarization_model:
        models_dict["Text Summarization"] = text_summarization_model
    if keyword_model:
        models_dict["Keyword Extraction"] = keyword_model
    if topic_modeling_model:
        models_dict["Topic Modeling"] = topic_modeling_model
    if topic_summarization_model:
        models_dict["Topic Summarization"] = topic_summarization_model
    if entities_model:
        models_dict["Named Entity Recognition"] = entities_model
    if sentiment_model:
        models_dict["Toxicity Analysis"] = sentiment_model
    if speaker_diarization_model:
        models_dict["Speaker Diarization"] = speaker_diarization_model

    return pd.DataFrame(list(models_dict.items()), columns=["Task", "Model Name"])


def _generate_report(
        args: argparse.Namespace,
        file_name: str,
        file_length: str,
        summary: str = None,
        langdetect_text: str = None,
        keywords: str = None,
        wordcloud_fig: plt.Figure = None,
        words_histogram: Image = None,
        n_sentences: int = None,
        entities_df: pd.DataFrame = None,
        n_entities: int = None,
        top_topics: pd.DataFrame = None,
        topic_graph: plt.Figure = None,
        neg_sentiment_df: pd.DataFrame = None,
        n_neg_results: int = None,
        speaker_count: int = None,
        speaker_fig: plt.Figure = None,
        models_df: pd.DataFrame = None,
        regular_font_path: str = "static/fonts/Cairo-Regular.ttf",
        bold_font_path: str = "static/fonts/Cairo-Bold.ttf",
        message: str = "creating report".upper()
) -> None:
    """
    Generate a PDF report summarizing the analysis results.

    :param args: Parsed command-line arguments.
    :param regular_font_path: Path to the regular font used to write the report.
    :param bold_font_path: Path to the bold font used to write the report.
    :param message: Message to print during report generation.
    :param file_name: Prefix for the output files.
    :param file_length: Length of the file.
    :param summary: Summary of the transcribed/translated text.
    :param langdetect_text: Example text used for language detection.
    :param keywords: Extracted keywords.
    :param wordcloud_fig: Wordcloud figure.
    :param words_histogram: Histogram of most frequently used words.
    :param entities_df: DataFrame of extracted named entities.
    :param n_entities: Number of extracted named entities.
    :param top_topics: DataFrame of top topics.
    :param topic_graph: Figure of topic interconnections.
    :param neg_sentiment_df: DataFrame of negative expressions.
    :param n_neg_results: Number of sentences labeled as negative sentiment.
    :param n_sentences: Number of sentences in the data.
    :param speaker_count: Count of speakers.
    :param speaker_fig: Speaker diarization figure.
    :param models_df: DataFrame of utilized models.
    """
    report_generator = PDFReportBuilder(
        file_path=args.file_path,
        output_dir=args.output_directory,
        language=args.source_language,
        regular_font_path=regular_font_path,
        bold_font_path=bold_font_path,
        message=message
    )

    # header section
    if args.header is not None:
        report_generator.set_header_center(args.header)
    if args.user is not None:
        report_generator.set_header_left("VeryEssence")
        report_generator.set_header_right(args.user)

    # first section
    title = "ANALYSIS REPORT"
    report_generator.elements.append(report_generator.add_title(title))
    report_generator.elements.append(report_generator.add_text(f"File: {file_name} (language: {args.source_language})"))
    report_generator.elements.append(report_generator.add_text(f"File length: {file_length}"))

    if langdetect_text is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_text(f"Example text: {langdetect_text}"),
            Spacer(1, 12)
        ]))
        report_generator.add_page_break()

    if wordcloud_fig is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_image(wordcloud_fig, 450, 225),
            Spacer(1, 12)
        ]))

    if summary is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_text(f"Summary: {summary}"),
            Spacer(1, 12)
        ]))

    if keywords is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_text(f"Keywords: {keywords}"),
            Spacer(1, 20)
        ]))

    report_generator.add_page_break()

    # second section
    if top_topics is not None and len(top_topics) > 0:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Topic Analysis"),
            report_generator.add_dataframe(top_topics),
            Spacer(1, 20)
        ]))

    if topic_graph is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Topic Interconnections"),
            report_generator.add_image(topic_graph, 600, 450),
        ]))
        report_generator.add_page_break()

    if words_histogram is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Word Analysis"),
            report_generator.add_image(words_histogram, 500, 250),
            Spacer(1, 12)
        ]))

    if entities_df is not None and len(entities_df) > 0:
        report_generator.elements.append(KeepTogether([
            report_generator.add_text(f"Found {n_entities} named entities among {n_sentences} sentences:"),
            report_generator.add_dataframe(entities_df)
        ]))

    if words_histogram is not None or entities_df is not None and len(entities_df) > 0:
        report_generator.add_page_break()

    if neg_sentiment_df is not None and len(neg_sentiment_df) > 0:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Sentiment Analysis"),
            report_generator.add_text(f"Found {n_neg_results} negative expressions among {n_sentences} sentences:"),
            report_generator.add_dataframe(neg_sentiment_df)
        ]))
        report_generator.add_page_break()

    # fourth section
    if speaker_fig is not None and speaker_count > 1:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Speaker Diarization"),
            report_generator.add_text(f"Number of speakers: {speaker_count}."),
            report_generator.add_image(speaker_fig, 500, 300),
        ]))
        report_generator.add_page_break()

    # fifth section
    report_generator.add_explanation_page()
    report_generator.add_page_break()

    # sixth section
    if models_df is not None:
        report_generator.elements.append(KeepTogether([
            report_generator.add_title("Labelling Note"),
            report_generator.add_text(
                "The information provided was generated using artificial intelligence. It may contain errors or "
                "inaccuracies, and should not be relied upon as a substitute for professional advice."),
            report_generator.add_text("For the present case, these models were applied:"),
            report_generator.add_dataframe(models_df)
        ]))

    report_generator.save()


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Main function to run the transcription, translation, analysis, and report generation tasks.

    :param args: The command line arguments.
    """
    whisper_column = "transcription"
    mt_column = "translation"
    tempfile = None
    langdetect_example = None
    transcribed_df = None
    translated_df = None
    summary = None
    keywords = None
    speaker_count = None
    speaker_fig = None
    wordcloud_fig = None
    words_histogram = None
    topics_df = None
    topic_graph = None
    entities_df = None
    n_entities = None
    n_neg_results = None
    n_sentences = None
    neg_sentiment_df = None
    language_detection_model = None
    whisper_transcription_model = None
    whisper_translation_model = None
    llm_translation_model = None
    text_summarization_model = None
    keyword_model = None
    topic_modeling_model = None
    topic_summarization_model = None
    entities_model = None
    sentiment_model = None
    speaker_diarization_model = None

    try:
        logging.info("\n\nInitiating VeryEssence...\n")

        file_processor, file_start_timecode_seconds, tempfile, file_length, file_length_hms = _file_handling(
            args=args
        )

        if args.task == 0:
            langdetect_example, args.source_language, language_detection_model = _detect_language(
                args=args,
                file=tempfile,
                file_start_timecode_seconds=file_start_timecode_seconds,
                column=whisper_column
            )

        if args.source_language == "de" and args.task not in [0, 1]:  # no translation required if language is "de"
            args.task = 1

        if args.task in [1, 3]:
            transcribed_df, whisper_transcription_model = _transcribe_audio(
                args=args,
                file=tempfile,
                file_start_timecode_seconds=file_start_timecode_seconds,
                column=whisper_column
            )

        if args.task in [2, 3]:
            translated_df, whisper_translation_model = _transcribe_audio(
                args=args,
                file=tempfile,
                file_start_timecode_seconds=file_start_timecode_seconds,
                column=whisper_column,
                label="translate",
                message="speech-to-text translation".upper()
            )
            if args.target_language != "en":
                translated_df, llm_translation_model = _translate_text(
                    args=args,
                    data=translated_df,
                    input_column=whisper_column,
                    output_column=mt_column
                )

        analysis_df, analysis_column, analysis_language = _analysis_data_pipeline(
            translated_df=translated_df,
            transcribed_df=transcribed_df,
            mt_column=mt_column,
            whisper_column=whisper_column,
            target_language=args.target_language,
            source_language=args.source_language
        )

        if args.summarization:
            summary, text_summarization_model, _ = _extract_from_text(
                args=args,
                data=analysis_df,
                column=analysis_column,
                message="text summarization".upper(),
                task="summary"
            )

        if args.words:
            n_sentences, keywords, words_histogram, wordcloud_fig, keyword_model = _word_statistics(
                args=args,
                data=analysis_df,
                column=analysis_column,
                language=analysis_language,
            )

        if args.topics:
            topics_df, topic_graph, topic_modeling_model, topic_summarization_model = (
                _topic_analysis(
                    args=args,
                    data=analysis_df,
                    column=analysis_column,
                    source_language=analysis_language,
                    file_length=file_length,
                )
            )

        if args.entities:
            entities_df, entities_model, n_entities = _extract_from_text(
                args=args,
                data=analysis_df,
                column=analysis_column,
                message="named entity recognition".upper(),
                task="entities"
            )

        if args.sentiment:
            neg_sentiment_df, sentiment_model, n_neg_results = _sentiment_analysis(
                args=args,
                data=analysis_df,
                column=analysis_column,
                message="sentiment analysis".upper(),
                task="sentiment"
            )

        if args.diarization:
            speaker_count, speaker_fig, speaker_diarization_model = _speaker_detection(
                args=args,
                file=tempfile,
            )

        models_df = _list_utilized_models(
            language_detection_model=language_detection_model,
            whisper_transcription_model=whisper_transcription_model,
            whisper_translation_model=whisper_translation_model,
            machine_translation_model=llm_translation_model,
            text_summarization_model=text_summarization_model,
            keyword_model=keyword_model,
            topic_modeling_model=topic_modeling_model,
            topic_summarization_model=topic_summarization_model,
            entities_model=entities_model,
            sentiment_model=sentiment_model,
            speaker_diarization_model=speaker_diarization_model
        )

        _generate_report(
            args=args,
            file_name=file_processor.file_name,
            file_length=file_length_hms,
            summary=summary,
            langdetect_text=langdetect_example,
            keywords=keywords,
            wordcloud_fig=wordcloud_fig,
            words_histogram=words_histogram,
            n_sentences=n_sentences,
            entities_df=entities_df,
            n_entities=n_entities,
            top_topics=topics_df,
            topic_graph=topic_graph,
            neg_sentiment_df=neg_sentiment_df,
            n_neg_results=n_neg_results,
            speaker_count=speaker_count,
            speaker_fig=speaker_fig,
            models_df=models_df,
            regular_font_path="static/fonts/Cairo-Regular.ttf",
            bold_font_path="static/fonts/Cairo-Bold.ttf",
            message="creating report".upper()
        )

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        raise

    finally:
        try:
            logging.info(f"Deleting tempfile: {tempfile}")
            tempfile_path = Path(tempfile)
            tempfile_path.unlink()
        except FileNotFoundError as e:
            logging.error(f"Error deleting file '{tempfile}': {e}")
        logging.info("The end of our elaborate plans, the end of everything that stands.")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
