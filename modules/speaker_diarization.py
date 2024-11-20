import datetime
from datetime import timedelta
import logging
import os

from dotenv import find_dotenv, load_dotenv, set_key
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyannote.audio import Pipeline
import torch

from modules.file_processing import FileProcessor


class SpeakerDiarization(FileProcessor):
    """
    A class for performing speaker diarization on audio files.

    Inherits from:
        FileProcessor

    Attributes:
        file (str): The processed audio file for diarization.
        model_id (str): The path to the pre-trained diarization model.
        device (torch.device): The device used for model inference (CPU, CUDA, or MPS).
        diarization: The diarization result object.

    Methods:
        _get_api_key(): Retrieves the API key from the environment or prompts the user to enter it.
        _model_inference(): Performs model inference for speaker diarization.
        _write_file_output(label: str, data: str | list | plt.Figure | object = None): Writes data to an output file.
        _count_speakers(label: str = "speakers_count") -> str: Counts the number of unique speakers in the diarization result.
        _plot_speakers(label: str = "speakers_plot") -> plt.Figure: Plots the speaker diarization result and saves it as a PNG file.
        apply_diarization() -> tuple[str, plt.Figure]: Applies speaker diarization to the processed audio file and saves the results.
    """
    def __init__(self, file_path: str, output_dir: str, file: str, message: str) -> None:
        """
        Initialize the SpeakerDiarization object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param file: The processed audio file for diarization.
        :param message: A message to print when initializing the diarization.
        """
        super().__init__(file_path=file_path, output_dir=output_dir)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.file = file
        self.model_id = "pyannote/speaker-diarization-3.1"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.diarization = None
        self.logger.info(f"{message}: {self.file_path}.")
        self.logger.info(f"Device selected for speaker diarization: {self.device}")

    def _get_api_key(self, token: str = "HF_HUB_TOKEN"):
        """
        Retrieve the API key from the environment or prompt the user to enter it.
        """
        try:
            dotenv_path = find_dotenv()
            load_dotenv(dotenv_path)
            self.api_key = os.getenv(token)
            if not self.api_key:
                self.api_key = input("Hugging Face token not found. Please enter your API key: ")
                if dotenv_path:
                    set_key(dotenv_path, token, self.api_key)
                else:
                    with open(".env", "w") as env_file:
                        env_file.write(f"{token}={self.api_key}\n")
        except Exception as e:
            self.logger.error(f"Error retrieving Hugging Face token: {e}", exc_info=True)
            raise

    def _model_inference(self) -> None:
        """
        Perform model inference for speaker diarization.
        """
        try:
            pipeline = Pipeline.from_pretrained(
                self.model_id,
                use_auth_token=self._get_api_key()
            )
            pipeline.to(self.device)
            self.diarization = pipeline(self.file)
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}", exc_info=True)
            raise

    def _write_file_output(self, label: str, data: str | list | plt.Figure | object = None) -> None:
        """
        Write data to an output file.

        :param data: The data to write.
        :param label: Label for the output file.
        """
        try:
            if isinstance(data, str | plt.Figure):
                super().write_file_output(data, label)
            elif label == "rttm":
                with open(self.output_dir / f"{self.file_prefix}_speakers.rttm", "w", encoding="utf-8") as rttm:
                    self.diarization.write_rttm(rttm)
            else:
                self.logger.error("Error - Invalid data format.")
        except Exception as e:
            self.logger.error(f"Error writing file output: {e}", exc_info=True)
            raise

    def _count_speakers(self, label: str = "speakers_count") -> int:
        """
        Count the number of unique speakers in the diarization result and save to a text file.

        :return: The number of speakers as a string.
        """
        try:
            speaker_count = len(self.diarization.labels())
            self._write_file_output(label, str(speaker_count))
            return speaker_count
        except Exception as e:
            self.logger.error(f"Error counting speakers: {e}", exc_info=True)
            raise

    def _plot_speakers(self, label: str = "speakers_plot") -> plt.Figure:
        """
        Plot the speaker diarization result and save as a PNG file.

        :return: A matplotlib Figure object representing the speaker plot.
        """
        try:
            speakers = sorted(
                self.diarization.labels(),
                key=lambda s: min(
                    segment.start for segment, _, speaker_label in self.diarization.itertracks(yield_label=True)
                    if speaker_label == s
                )
            )

            colors = plt.cm.rainbow(np.linspace(0, 1, len(speakers)))
            color_map = dict(zip(speakers, colors))

            fig = plt.figure(figsize=(10, 6))

            total_length = max(segment.end for segment, _, _ in self.diarization.itertracks(yield_label=True))

            x_ticks = np.arange(0, total_length, max(30, total_length // 10))
            x_tick_labels = [str(datetime.timedelta(seconds=tick)).split(":") for tick in x_ticks]
            x_tick_labels = [":".join([i for i in label if i != "00"]) for label in x_tick_labels]

            for segment, _, current_speaker_label in self.diarization.itertracks(yield_label=True):
                speaker = current_speaker_label
                start_time = segment.start
                end_time = segment.end
                plt.plot([start_time, end_time], [speakers.index(speaker)]*2, color=color_map[speaker], linewidth=10)

            fontsize_ticks = 16
            fontsize_labels = 16

            plt.yticks(range(len(speakers)), speakers, fontsize=fontsize_ticks)
            plt.xticks(x_ticks, x_tick_labels, fontsize=fontsize_ticks)
            plt.xlabel("Time (hh:mm:ss)", fontsize=fontsize_labels)
            plt.ylabel("Speaker", fontsize=fontsize_labels)
            plt.tight_layout()

            # Pass the desired label to _write_file_output
            self._write_file_output(label, fig)
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting speakers: {e}", exc_info=True)
            raise

    def apply_diarization(self) -> tuple[int, plt.Figure]:
        """
        Apply speaker diarization to the processed audio file and save the results.

        :return: A tuple containing the number of speakers and the plot Figure.
        """
        try:
            start = self.get_raw_time()
            self.logger.info(f"Speaker diarization with model '{self.model_id}' - Initialized.")
            self._model_inference()
            end = self.get_raw_time()
            duration_seconds = end - start
            duration = str(timedelta(seconds=round(duration_seconds.total_seconds())))
            self.logger.info(f"Speaker diarization for {self.file_prefix} - Finished with duration: {duration}.")
            self._write_file_output("rttm")
            speaker_count = self._count_speakers()
            fig = self._plot_speakers()
            return speaker_count, fig
        except Exception as e:
            self.logger.error(f"Error during speaker diarization: {e}", exc_info=True)
            raise
