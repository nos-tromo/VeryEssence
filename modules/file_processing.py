from datetime import datetime, timedelta
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment


class FileProcessor:
    """
    FileProcessor is the central class for file processing within VeryEssence and is connected to all other classes through
    inheritance.

    Attributes:
        file_start_timecode (str): The start timecode for slicing input files.
        file_end_timecode (str): The end timecode for slicing input files.

    Methods:
        _setup_directories(file_path: str, output_dir: str): Sets up necessary directories for file processing.
        _get_seconds(time_str: str): Converts a time string "hh:mm:ss" to total seconds.
        _trim_audio(file_start_timecode_seconds: int, audio: AudioSegment): Trims the audio segment based on the start and end timecodes.
        _add_silence(spacermilli: int = 2000): Adds a padding of silence to the beginning and end of the processed audio file.
        get_raw_time() -> datetime: Retrieves the current raw datetime.
        audio_processing() -> tuple[int, str, int]: Processes the audio file, converts it to .wav, and adds silence padding.
        hms(seconds: int) -> None | str: Converts seconds to a "HH:MM:SS" formatted string.
        write_file_output(data: str | list | tuple | pd.DataFrame | plt.Figure, label: str, target_language: str = False) -> str | list | pd.DataFrame | plt.Figure:
            Writes the provided data to an appropriate output file based on its type (text, list, DataFrame, or plot).
    """
    def __init__(
            self,
            file_path: str = None,
            output_dir: str | Path = None,
            file_start_timecode: str = None,
            file_end_timecode: str = None
    ) -> None:
        """
        Initialize the FileProcessor object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param file_start_timecode: The start timecode for slicing input files.
        :param file_end_timecode: The end timecode for slicing input files.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self._setup_directories(file_path, output_dir)
        self.file_start_timecode = file_start_timecode
        self.file_end_timecode = file_end_timecode

    def _setup_directories(self, file_path: str, output_dir: str) -> None:
        """
        Set up the necessary directories for file processing. Create the output subdirectory if it doesn't exist.
        """
        try:
            self.file_path = file_path
            self.file_name = Path(self.file_path).name
            self.file_prefix = Path(self.file_name).stem
            output_subdir = Path(output_dir) / self.file_prefix
            if not output_subdir.exists():
                output_subdir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Directory {output_subdir} does not exist. Creating a new one.")
            else:
                self.logger.info(f"Directory {output_subdir} already exists.")
            self.output_dir = output_subdir
        except Exception as e:
            self.logger.error(f"Error processing input file: {e}", exc_info=True)
            raise

    @staticmethod
    def _get_seconds(time_str: str) -> int:
        """
        Convert a time string "hh:mm:ss" to total seconds.

        :param time_str: The time string to convert.
        :return: The total number of seconds.
        """
        try:
            return sum(int(part) * factor for part, factor in zip(map(int, time_str.split(":")), (3600, 60, 1)))
        except ValueError as e:
            logging.getLogger(FileProcessor.__name__).error(f"Error converting time to seconds: {e}", exc_info=True)
            raise

    def _trim_audio(self, file_start_timecode_seconds: int, audio: AudioSegment) -> AudioSegment:
        """
        Trim the provided audio segment based on the start and end timecodes.

        This method trims the input `audio` from the specified start time (`file_start_timecode_seconds`)
        to either the specified end time (`self.file_end_timecode`) or the full length of the audio segment if no end time is provided.

        :param file_start_timecode_seconds: The starting point of the audio in seconds from which trimming should begin.
        :param audio: The `AudioSegment` object representing the audio to be trimmed.
        :return: A trimmed `AudioSegment` object from the specified start time to the end time or end of the audio.
        """
        try:
            start_ms = file_start_timecode_seconds * 1000
            end_ms = self._get_seconds(
                self.file_end_timecode
            ) * 1000 if self.file_end_timecode != "0" else len(audio)
            return audio[start_ms:end_ms]
        except Exception as e:
            self.logger.error(f"Error trimming audio: {e}", exc_info=True)
            raise

    def _add_silence(self, spacermilli: int = 2000) -> AudioSegment:
        """
        Add a padding of silence to the beginning and end of the processed audio file to minimize timestamp loss.

        :param spacermilli: The duration of the silence in milliseconds (default: 2000).
        :return: An AudioSegment of silence.
        """
        try:
            return AudioSegment.silent(duration=spacermilli)
        except Exception as e:
            self.logger.error(f"Error adding silence to audio: {e}", exc_info=True)
            raise

    def get_raw_time(self) -> datetime:
        """
        Get the current raw datetime.

        :return: The current datetime.
        """
        try:
            return datetime.now()
        except Exception as e:
            self.logger.error(f"Error getting raw time: {e}", exc_info=True)
            raise

    def audio_processing(self) -> tuple[int, str, int]:
        """
        Calculate the start time in seconds and convert the file to .wav format. Convert to mono if necessary, then
        resample to 16kHz if necessary. Append silence padding to the end of the audio segment to prevent timestamp loss.

        :return: A tuple containing the start timecode in seconds, the temporary file name, and the duration in seconds.
        """
        try:
            file_start_timecode_seconds = self._get_seconds(self.file_start_timecode)
            file_extension = "wav"
            tempfile = f"tempfile_{self.file_prefix.replace(' ', '_')}.{file_extension}"
            audio = AudioSegment.from_file(self.file_path)

            if audio.channels > 1:
                audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            # Trim the audio if needed
            if self.file_start_timecode != "0" or self.file_end_timecode != "0":
                audio = self._trim_audio(file_start_timecode_seconds, audio)

            # Add silence padding
            silence = self._add_silence()
            padded_audio = silence + audio + silence

            # Export the processed audio
            padded_audio.export(tempfile, format=file_extension)
            duration_seconds = math.ceil(len(padded_audio) / 1000)

            return file_start_timecode_seconds, tempfile, duration_seconds
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}", exc_info=True)
            raise

    def hms(self, seconds: int) -> None | str:
        """
        Convert seconds to a "HH:MM:SS" formatted string.

        :param seconds: The total number of seconds.
        :return: The formatted time string or None if input is invalid.
        """
        try:
            if seconds is None or pd.isna(seconds):
                return None
            td = timedelta(seconds=seconds)
            return str(td).split(".")[0]
        except Exception as e:
            self.logger.error(f"Error processing 'HH:MM:SS' time to formatted string: {e}", exc_info=True)
            raise

    def write_file_output(
            self,
            data: str | list | tuple | pd.DataFrame | plt.Figure,
            label: str,
            target_language: str = False
    ) -> str | list | pd.DataFrame | plt.Figure:
        """
        Write the provided data to an appropriate output file based on its type (text, list, DataFrame, or plot).
        Handles creating file paths, appending data if necessary, and saving the output in various formats (TXT, XLSX, PNG).

        :param data: The data to be written, which can be a string, list, DataFrame, or matplotlib Figure.
        :param label: The label used to create the file name.
        :param target_language: Optional language code to be appended to the file name.
        :return: The data that was written, which can be a string, list, DataFrame, or Figure.
        """
        try:
            if isinstance(data, str | list | tuple | pd.DataFrame | plt.Figure):
                # Creating paths for file output
                language_suffix = f"_{target_language}" if target_language else ""
                output_file_path_txt = self.output_dir / f"{self.file_prefix}_{label}{language_suffix}.txt"
                output_file_path_excel = self.output_dir / f"{self.file_prefix}_{label}{language_suffix}.xlsx"
                output_file_path_png = self.output_dir / f"{self.file_prefix}_{label}{language_suffix}.png"

                # String file operations
                if isinstance(data, str):
                    with open(f"{output_file_path_txt}", "w", encoding="utf-8") as f:
                        f.write(data)
                # List/tuple file operations
                elif isinstance(data, list | tuple):
                    with open(f"{output_file_path_txt}", "w", encoding="utf-8") as f:
                        for item in data:
                            f.write(f"{item}\n")
                # DataFrame file operations
                elif isinstance(data, pd.DataFrame):
                    # Handle text file append/write
                    if output_file_path_txt.exists():
                        with open(output_file_path_txt, "r", encoding="utf-8") as file:
                            existing_content = file.read()
                        new_content = data.to_csv(sep="\t", index=False, header=False, encoding="utf-8")
                        combined_content = existing_content + "\n" + new_content
                    else:
                        combined_content = data.to_csv(sep="\t", index=False, encoding="utf-8")
                    with open(output_file_path_txt, "w", encoding="utf-8") as file:
                        file.write(combined_content)

                    # Handle Excel file
                    if output_file_path_excel.exists():
                        # Load existing workbook and append data
                        with pd.ExcelWriter(output_file_path_excel, engine="openpyxl", mode="a") as writer:
                            # Create a new unique sheet name if necessary
                            sheet_name = "Sheet1"
                            existing_sheets = writer.book.sheetnames
                            if sheet_name in existing_sheets:
                                sheet_name = f"{sheet_name}_{len(existing_sheets) + 1}"

                            # Write to the Excel file
                            data.to_excel(writer, index=False, sheet_name=sheet_name)
                    else:
                        with pd.ExcelWriter(output_file_path_excel, engine="openpyxl") as writer:
                            data.to_excel(writer, index=False, sheet_name="Sheet1")
                elif isinstance(data, plt.Figure):
                    data.savefig(output_file_path_png)

                # File saving notifications
                if output_file_path_txt.exists():
                    self.logger.info(f"Saved output: {output_file_path_txt}")
                if output_file_path_excel.exists():
                    self.logger.info(f"Saved output: {output_file_path_excel}")
                if output_file_path_png.exists():
                    self.logger.info(f"Saved output: {output_file_path_png}")

            else:
                self.logger.error(f"Data type not supported for writing output: {type(data)}", exc_info=True)

            return data
        except Exception as e:
            self.logger.error(f"Error creating output: {e}", exc_info=True)
            raise
