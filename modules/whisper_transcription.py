from datetime import timedelta
import gc
import logging

import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig, pipeline
import whisper

from modules.file_processing import FileProcessor
from config.whisper_cfg import whisper_model_config


class WhisperTranscriber(FileProcessor):
    """
    This class transcribes/translates the preprocessed input file, stores the result in a DataFrame, and writes it to
    tabular output.

    Inherits from:
        FileProcessor

    Attributes:
        language (str): The source language of the text.
        model_size (str): The size of the Whisper model (tiny, base, small, medium, large).
        file (str): The name of the preprocessed input file.
        task (str): Indicates whether the task is transcription or translation.
        start_time_seconds (int): Starting point of the Whisper transcription to be aligned with timestamps.
        start_column (str): The text column with the starting timestamp.
        end_column(str): The text column with the ending timestamp.
        text_column (str): The text column where the result is stored.
        device (torch.device): The device used for model inference (CPU, CUDA, or MPS).
        model_id (str): The path of the Whisper model to be used for inference.

    Methods:
        _select_model(): Select the appropriate model for transcription/translation based on the specified model size and source language.
        _detect_language() -> tuple[str, str]: Detect the language of the input file and return a short language example.
        _model_inference() -> dict: Run inference on the preprocessed file using the selected model and store the result in a dictionary.
        _store_result_in_df(data: dict) -> pd.DataFrame: Process the inference result by extracting relevant data, formatting it into a DataFrame.
        _ends_with_punctuation(text: str) -> bool: Check if the given text ends with a sentence-ending punctuation mark.
        _merge_transcriptions_by_sentence(data: pd.DataFrame) -> pd.DataFrame: Merge rows of the transcription DataFrame until a sentence-ending punctuation mark is found.
        _write_file_output(data: pd.DataFrame) -> None: Write the processed DataFrame to the appropriate file format(s) based on the task and language settings.
        data_pipeline() -> tuple[str, str] | pd.DataFrame: Execute the full data pipeline for the specified task, including language detection, model inference, processing the results, and writing the output.
    """
    def __init__(
            self,
            file_path: str,
            output_dir: str,
            language: str,
            model_size: str,
            file: str,
            task: str,
            start_time_seconds: int,
            text_column: str,
            message: str
    ) -> None:
        """
        Initialize the WhisperTranscriber object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param language: The source language of the file.
        :param model_size: The size of the Whisper model (tiny, base, small, medium, large).
        :param file: The name of the preprocessed input file.
        :param task: Indicates whether the task is detection, transcription, or translation.
        :param start_time_seconds: Starting point of the Whisper transcription to be aligned with timestamps.
        :param text_column: The text column where the result is stored.
        :param message: A message to print when initializing the transcription.
        """
        super().__init__(file_path=file_path, output_dir=output_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.language = language
        self.model_size = model_size
        self.file = file
        self.task = task
        self.start_time_seconds = start_time_seconds
        self.start_column = "start"
        self.end_column = "end"
        self.text_column = text_column
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.logger.info(f"{message}: {self.file_path} (language: {self.language})")
        self.logger.info(f"Device selected: {self.device}")

        self.model_id = "openai/whisper-tiny" if self.task == "detect" else self._select_model()

    def _select_model(self) -> str:
        """
        Select the appropriate model for transcription/translation based on the specified model size and source
        language.
        """
        try:
            config_key = f"{self.model_size}_{self.task}" if self.model_size not in whisper_model_config else self.model_size
            self.model_config = whisper_model_config.get(config_key)
            if self.model_config is None:
                raise ValueError(f"Configuration for key '{config_key}' not found.")

            model_id = self.model_config.get("model_id")
            self.generation_config_id = self.model_config.get("config")
            dtype = self.model_config.get("dtype")
            if "dtype" not in self.model_config:
                self.logger.warning(f"'dtype' not specified for model '{model_id}'. Defaulting to 'float16'.")
            torch_dtype = torch.float32 if dtype == "float32" else torch.float16
            self.logger.info(f"Using torch dtype: {torch_dtype} for model: {model_id}")

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model.generation_config = GenerationConfig.from_pretrained(self.generation_config_id)

            chunk_length_s = 30 if torch.cuda.is_available() else 20 if torch.backends.mps.is_available() else 15
            batch_size = 16 if torch.cuda.is_available() else 4 if torch.backends.mps.is_available() else 2
            self.logger.info(f"Chunk length (seconds): {chunk_length_s}")
            self.logger.info(f"Batch size: {batch_size}")

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                feature_extractor=self.processor.feature_extractor,
                tokenizer=self.processor.tokenizer,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                stride_length_s=int(chunk_length_s / 3),
                return_timestamps=True,
                device=self.device,
                torch_dtype=torch_dtype
            )
            self.logger.info("Model, config and pipeline set up successfully.")
            return model_id

        except Exception as e:
            self.logger.error(
                f"Error setting up model '{self.model_id}' for language '{self.language}': {e}", exc_info=True
            )
            raise

    def _detect_language(self) -> tuple[str, str]:
        """
        Detect the language of the input file and return a short language example.

        :return: A tuple containing a short text example and the language code of the input file.
        """
        try:
            model = whisper.load_model(self.model_size)
            audio = whisper.load_audio(self.file)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)

            options = whisper.DecodingOptions(fp16=False)
            inference = whisper.decode(model, mel, options)
            example_text = inference.text
            self.language = max(probs, key=probs.get)

            self.logger.info(f"Language detected: {self.language}")
            return example_text, self.language
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}", exc_info=True)
            raise

    def _model_inference(self) -> dict:
        """
        Run inference on the preprocessed file using the selected model and store the result in a dictionary.

        :return: A dictionary containing the model inference results.
        """
        try:
            start = self.get_raw_time()
            self.logger.info(f"{self.task.capitalize()} with model '{self.model_id}' - Initialized.")
            result = self.pipe(self.file, generate_kwargs={"language": self.language, "task": self.task})
            end = self.get_raw_time()
            duration_seconds = end - start
            duration = str(timedelta(seconds=round(duration_seconds.total_seconds())))
            self.logger.info(
                f"{self.task.capitalize()} with model '{self.model_id}' - Finished with duration: {duration}."
            )
            return result

        except Exception as e:
            self.logger.error(f"Error during task '{self.task}': {e}", exc_info=True)
            raise

        finally:
            # Clean up resources
            del self.model
            del self.processor
            del self.pipe

            # Clear cache and run garbage collection
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Flushed GPU memory.")
            gc.collect()

            self.logger.info("Resources cleaned up.")

    def _store_result_in_df(self, data: dict) -> pd.DataFrame:
        """
        Process the inference result by extracting relevant data, formatting it into a DataFrame,
        and preparing it for output.

        :param data: The inference result data as a dictionary.
        :return: A DataFrame containing the processed transcription data with timestamps.
        """
        try:
            data_list = []
            for chunk in data.get("chunks"):
                if "text" in chunk and chunk.get("text"):
                    timestamp = chunk.get("timestamp")
                    start = timestamp[0] + self.start_time_seconds if timestamp[0] is not None else None
                    end = timestamp[1] + self.start_time_seconds if timestamp[1] is not None else None
                    text = chunk.get("text")
                    data_list.append([start, end, text])
            column_names = [self.start_column, self.end_column, self.text_column]
            df = pd.DataFrame(data_list, columns=column_names)
            df[self.start_column] = df.get(self.start_column).apply(super().hms)
            df[self.end_column] = df.get(self.end_column).apply(super().hms)
            self.logger.info("DataFrame created successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}", exc_info=True)
            raise

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """
        Check if the given text ends with a sentence-ending punctuation mark.

        :param text: The text string to check.
        :return: True if the text ends with ".", "!", or "?", otherwise False.
        """
        try:
            return text.strip().endswith((".", "!", "?"))
        except Exception as e:
            logging.error(f"Error in _ends_with_punctuation: {e}", exc_info=True)
            raise

    def _merge_transcriptions_by_sentence(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge rows of the transcription DataFrame until a sentence-ending punctuation mark is found,
        adjusting the start and end timestamps accordingly.

        :param data: The original DataFrame containing transcription data.
        :return: A new DataFrame with merged sentences and adjusted timestamps.
        """
        try:
            new_rows = []
            current_row = {self.start_column: None, self.end_column: None, self.text_column: ""}

            for index, row in data.iterrows():
                if current_row.get(self.start_column) is None:
                    current_row[self.start_column] = row.get(self.start_column)

                if row[self.text_column]:
                    current_row[self.text_column] += row[self.text_column].strip() + " "

                if self._ends_with_punctuation(row[self.text_column]):
                    current_row[self.end_column] = row.get(self.end_column)
                    new_rows.append(current_row)
                    current_row = {self.start_column: None, self.end_column: None, self.text_column: ""}

            merged_df = pd.DataFrame(new_rows)
            merged_df[self.text_column] = merged_df[self.text_column].str.strip()
            self.logger.info("Transcriptions successfully merged by sentence.")
            return merged_df
        except Exception as e:
            self.logger.error(f"Error in _merge_transcriptions_by_sentence: {e}", exc_info=True)
            raise

    def _write_file_output(self, data: pd.DataFrame) -> None:
        """
        Write the processed DataFrame to the appropriate file format(s) based on the task and language settings.

        :param data: The DataFrame containing the processed transcription data.
        """
        try:
            super().write_file_output(data, self.task, self.language)
        except Exception as e:
            self.logger.error(f"Error writing file output: {e}", exc_info=True)
            raise

    def data_pipeline(self) -> tuple[str, str] | pd.DataFrame:
        """
        Execute the full data pipeline for the specified task, including language detection, model inference, processing
        the results, and writing the output to files. Returns the DataFrame containing the final output, if applicable.

        :return: The processed data, either as a tuple of language and text or a DataFrame depending on the task.
        """
        try:
            if self.task == "detect":
                data = self._detect_language()
            else:
                result = self._model_inference()
                data = self._store_result_in_df(result)
                if (self.task == "transcribe" and self.language != "ar") or self.task == "translate":
                    data = self._merge_transcriptions_by_sentence(data)
            if self.task != "translate":
                self._write_file_output(data)
            return data
        except Exception as e:
            self.logger.error(f"Error in data pipeline for task '{self.task}': {e}", exc_info=True)
            raise
