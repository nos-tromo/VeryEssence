from datetime import datetime
import io
import logging
from pathlib import Path
import time

import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, \
    TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from modules.file_processing import FileProcessor
from config.pdfreport_cfg import explanations


class PDFReportBuilder(FileProcessor):
    """
    A class to generate PDF reports with customized styles and content including titles, headings, text, tables,
    img, and page numbers.

    Inherits from:
        FileProcessor

    Attributes:
        report_filepath (str): The path to the PDF report file.
        doc (SimpleDocTemplate): The document template for the PDF report.
        language (str): The language of the report content.
        regular_font_path (str): Path to the regular font used to write the report.
        bold_font_path (str): Path to the bold font used to write the report.
        elements (list): A list to store elements to be added to the PDF.
        styles (StyleSheet1): A collection of styles for formatting the PDF.
        temp_files (list): A list to keep track of temporary files.
        timestamp (str): A timestamp of the report creation.
        header_center (str): Text for the center header.
        header_right (str): Text for the right header.
        header_left (str): Text for the left header.

    Methods:
        _customize_styles(): Customizes the styles used in the PDF report, including fonts and default styles.
        add_title(title: str) -> Paragraph: Adds a title to the PDF report.
        _add_heading(heading: str): Adds a heading to the PDF report.
        add_text(text: str) -> Paragraph: Adds a paragraph of text to the PDF report.
        add_dataframe(df: pd.DataFrame) -> Table: Adds a DataFrame as a table to the PDF report.
        add_image(img_obj, width: int = 400, height: int = 300) -> Image: Adds an image to the PDF report.
        add_page_break(): Adds a page break to the PDF report.
        set_header_center(text: str): Sets the center header text for the PDF report.
        set_header_right(text: str): Sets the right header text for the PDF report.
        set_header_left(text: str): Sets the left header text for the PDF report.
        _draw_header_center(canvas: Canvas, doc: BaseDocTemplate): Draws the center header text on the canvas.
        _draw_header_right(canvas: Canvas, doc: BaseDocTemplate): Draws the right header text on the canvas.
        _draw_header_left(canvas: Canvas, doc: BaseDocTemplate): Draws the left header text on the canvas.
        _draw_timestamp(canvas: Canvas): Draws the timestamp at the bottom of the page.
        _add_page_number(canvas: Canvas, doc: BaseDocTemplate): Adds headers, page numbers, and footers to the canvas.
        _draw_page_number(canvas: Canvas): Draws the page number at the bottom of the page.
        _draw_bottomline(canvas: Canvas, doc: BaseDocTemplate): Draws the bottom line text centered at the bottom of the page.
        add_explanation_page(): Adds an explanatory page to the PDF report with descriptions of the features.
        save(): Saves the PDF report to a file.
    """
    def __init__(
            self,
            file_path: str,
            output_dir: str,
            language: str,
            regular_font_path: str,
            bold_font_path: str,
            message: str
    ) -> None:
        """
        Initialize the PDFReport object.

        :param file_path: The path to the input file.
        :param output_dir: The directory to save the output files.
        :param language: The language of the report content.
        :param regular_font_path: Path to the regular font used to write the report.
        :param bold_font_path: Path to the bold font used to write the report.
        :param message: A message to print when initializing the report.
        """
        super().__init__(file_path=file_path, output_dir=output_dir)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{message}: {self.file_path}")

        file_name = Path(self.file_path).name
        file_prefix = Path(file_name).stem
        unix_time = str(int(time.time()))
        self.report_filepath = Path(self.output_dir) / f"REPORT_{unix_time}_{file_prefix}.pdf"
        self.doc = SimpleDocTemplate(str(self.report_filepath), pagesize=letter)
        self.language = language
        self.regular_font_path = regular_font_path
        self.bold_font_path = bold_font_path
        self.elements = []
        self.styles = getSampleStyleSheet()
        self._customize_styles()
        self.temp_files = []  # To keep track of temporary files
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialize header texts
        self.header_center = None
        self.header_right = None
        self.header_left = None

    def _customize_styles(self) -> None:
        """
        Customize the styles used in the PDF report, including registering custom fonts and modifying default styles.
        Fonts must be downloaded and stored under fonts/ before usage.
        """
        try:
            pdfmetrics.registerFont(TTFont("Cairo", self.regular_font_path))
            pdfmetrics.registerFont(TTFont("Cairo-Bold", self.bold_font_path))

            # Modify the existing Title style
            title_style = self.styles["Title"]
            title_style.fontName = "Cairo-Bold"
            title_style.fontSize = 14
            title_style.leading = 18
            title_style.spaceAfter = 14
            title_style.textColor = colors.HexColor("#2E4053")
            title_style.alignment = 1  # Center alignment

            # Modify the existing Heading2 style or add it if it doesn't exist
            if "Heading2" in self.styles:
                heading2_style = self.styles["Heading2"]
            else:
                heading2_style = ParagraphStyle(name="Heading2")
                self.styles.add(heading2_style)
            heading2_style.fontName = "Cairo-Bold"
            heading2_style.fontSize = 12
            heading2_style.leading = 16
            heading2_style.spaceAfter = 12
            heading2_style.textColor = colors.HexColor("#1F618D")
            heading2_style.backColor = colors.HexColor("#D5D8DC")
            heading2_style.leftIndent = 5

            # Modify the existing BodyText style
            body_text_style = self.styles["BodyText"]
            body_text_style.fontName = "Cairo"
            body_text_style.fontSize = 10
            body_text_style.leading = 12
            body_text_style.spaceAfter = 12
            body_text_style.textColor = colors.black  # HexColor("#212F3D")

            # Add or modify the TableHeader style
            if "TableHeader" in self.styles:
                table_header_style = self.styles["TableHeader"]
            else:
                table_header_style = ParagraphStyle(name="TableHeader")
                self.styles.add(table_header_style)
            table_header_style.fontName = "Cairo-Bold"
            table_header_style.fontSize = 10
            table_header_style.leading = 12
            table_header_style.textColor = colors.whitesmoke
            table_header_style.backColor = colors.HexColor("#1F618D")
            table_header_style.alignment = 1

            # Add or modify the TableBody style
            if "TableBody" in self.styles:
                table_body_style = self.styles["TableBody"]
            else:
                table_body_style = ParagraphStyle(name="TableBody")
                self.styles.add(table_body_style)
            table_body_style.fontName = "Cairo"
            table_body_style.fontSize = 10
            table_body_style.leading = 12
            table_body_style.textColor = colors.HexColor("#212F3D")
            table_body_style.alignment = 1

            self.logger.info("Styles customized successfully.")
        except Exception as e:
            self.logger.error(f"Error customizing styles: {e}", exc_info=True)
            raise

    def add_title(self, title: str) -> Paragraph:
        """
        Add a title to the PDF report.

        :param title: The title text.
        :return: A Paragraph object containing the title.
        """
        try:
            if self.language == "ar":
                title = get_display(arabic_reshaper.reshape(title))
            return Paragraph(title, self.styles["Title"])
        except Exception as e:
            self.logger.error(f"Error adding title: {e}", exc_info=True)
            raise

    def _add_heading(self, heading: str) -> None:
        """
        Add a heading to the PDF report.

        :param heading: The heading text.
        """
        try:
            if self.language == "ar":
                heading = get_display(arabic_reshaper.reshape(heading))
            self.elements.append(Paragraph(heading, self.styles["Heading2"]))
            self.elements.append(Spacer(1, 12))
        except Exception as e:
            self.logger.error(f"Error adding heading: {e}", exc_info=True)
            raise

    def add_text(self, text: str) -> Paragraph:
        """
        Add a paragraph of text to the PDF report.

        :param text: The text to add.
        :return: A Paragraph object containing the text.
        """
        try:
            if self.language == "ar":
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                text = bidi_text
            return Paragraph(text, self.styles["BodyText"])
        except Exception as e:
            self.logger.error(f"Error adding text: {e}", exc_info=True)
            raise

    def add_dataframe(self, df: pd.DataFrame) -> Table:
        """
        Add a DataFrame as a table to the PDF report.

        :param df: The DataFrame to add.
        :return: A Table object containing the DataFrame data.
        """
        # Convert DataFrame to a list of lists and wrap text elements in Paragraph
        try:
            # Convert DataFrame to a list of lists and wrap text elements in Paragraph
            if self.language == "ar":
                table_data = [[Paragraph(get_display(arabic_reshaper.reshape(str(col))),
                                         self.styles["TableHeader"]) for col in df.columns]] + \
                             [[Paragraph(get_display(arabic_reshaper.reshape(str(cell))),
                                         self.styles["TableBody"]) for cell in row] for row in df.itertuples(index=False)]
            else:
                table_data = [[Paragraph(str(col), self.styles["TableHeader"]) for col in df.columns]] + \
                             [[Paragraph(str(cell),
                                         self.styles["TableBody"]) for cell in row] for row in df.itertuples(index=False)]

            table = Table(table_data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F618D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Cairo-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F9F9")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ROWBACKGROUNDS", (1, 1), (-1, -1), [colors.white, colors.HexColor("#EBF5FB")])
            ]))
            return table
        except Exception as e:
            self.logger.error(f"Error adding DataFrame to PDF: {e}", exc_info=True)
            raise

    @staticmethod
    def add_image(img_obj, width: int = 400, height: int = 300) -> Image:
        """
        Add an image to the PDF report.

        :param img_obj: The image object, which can be a matplotlib figure, PIL Image, BytesIO object, or file path.
        :param width: The width of the image in the PDF (default: 400).
        :param height: The height of the image in the PDF (default: 300).
        :return: An Image object.
        """
        try:
            buf = io.BytesIO()
            if isinstance(img_obj, plt.Figure):
                img_obj.savefig(buf, format="png")
                buf.seek(0)
            elif isinstance(img_obj, PILImage.Image):
                img_obj.save(buf, format="PNG")
                buf.seek(0)
            elif isinstance(img_obj, io.BytesIO):
                buf = img_obj
            else:
                return Image(img_obj, width, height)

            return Image(buf, width, height)
        except Exception as e:
            logging.getLogger(PDFReportBuilder.__name__).error(f"Error adding image: {e}", exc_info=True)
            raise

    def add_page_break(self) -> None:
        """
        Add a page break to the PDF report.
        """
        try:
            self.elements.append(PageBreak())
        except Exception as e:
            self.logger.error(f"Error adding page break: {e}", exc_info=True)
            raise

    def set_header_center(self, text: str) -> None:
        """
        Set the center header text for the PDF report.

        :param text: The text to set as the center header.
        """
        try:
            if self.language == "ar":
                text = get_display(arabic_reshaper.reshape(text))
            self.header_center = text
        except Exception as e:
            self.logger.error(f"Error setting center header: {e}", exc_info=True)
            raise

    def set_header_right(self, text: str) -> None:
        """
        Set the right header text for the PDF report.

        :param text: The text to set as the right header.
        """
        try:
            if self.language == "ar":
                text = get_display(arabic_reshaper.reshape(text))
            self.header_right = text
        except Exception as e:
            self.logger.error(f"Error setting right header: {e}", exc_info=True)
            raise

    def set_header_left(self, text: str) -> None:
        """
        Set the left header text for the PDF report.

        :param text: The text to set as the left header.
        """
        try:
            if self.language == "ar":
                text = get_display(arabic_reshaper.reshape(text))
            self.header_left = text
        except Exception as e:
            self.logger.error(f"Error setting left header: {e}", exc_info=True)
            raise

    def _draw_header_center(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        """
        Draw the center header text on the canvas.
        """
        try:
            if self.header_center:
                header_center_fontsize = 11
                canvas.setFont("Helvetica-Bold", header_center_fontsize)
                text_width = canvas.stringWidth(self.header_center, "Helvetica-Bold", header_center_fontsize)
                center_position = (doc.pagesize[0] - text_width) / 2
                canvas.drawString(center_position, doc.pagesize[1] - 20 * mm, self.header_center)
        except Exception as e:
            self.logger.error(f"Error drawing center header: {e}", exc_info=True)
            raise

    def _draw_header_right(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        """
        Draw the right header text on the canvas.
        """
        try:
            if self.header_right:
                header_right_fontsize = 10
                canvas.setFont("Helvetica-Bold", header_right_fontsize)
                text_width = canvas.stringWidth(self.header_right, "Helvetica", header_right_fontsize)
                canvas.drawString(doc.pagesize[0] - text_width - 20 * mm, doc.pagesize[1] - 25 * mm, self.header_right)
        except Exception as e:
            self.logger.error(f"Error drawing right header: {e}", exc_info=True)
            raise

    def _draw_header_left(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        """
        Draw the left header text on the canvas.
        """
        try:
            if self.header_left:
                header_left_fontsize = 10
                canvas.setFont("Helvetica", header_left_fontsize)
                canvas.drawString(20 * mm, doc.pagesize[1] - 25 * mm, self.header_left)
        except Exception as e:
            self.logger.error(f"Error drawing left header: {e}", exc_info=True)
            raise

    def _draw_timestamp(self, canvas: Canvas) -> None:
        """
        Draw the timestamp at the bottom of the page.
        """
        try:
            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y - %H:%M:%S")
            canvas.setFont("Helvetica", 9)
            canvas.drawString(15 * mm, 10 * mm, timestamp)
        except Exception as e:
            self.logger.error(f"Error drawing timestamp: {e}", exc_info=True)
            raise

    def _add_page_number(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        """
        Add headers, page numbers, and footers to the canvas.
        """
        try:
            canvas.saveState()
            self._draw_header_center(canvas, doc)
            self._draw_header_right(canvas, doc)
            self._draw_header_left(canvas, doc)
            self._draw_timestamp(canvas)
            self._draw_page_number(canvas)
            self._draw_bottomline(canvas, doc)
            canvas.restoreState()
        except Exception as e:
            self.logger.error(f"Error adding page number: {e}", exc_info=True)
            raise

    @staticmethod
    def _draw_page_number(canvas: Canvas) -> None:
        """
        Draw the page number at the bottom of the page.
        """
        try:
            page_num = canvas.getPageNumber()
            text = str(page_num)
            canvas.setFont("Helvetica", 9)
            canvas.drawRightString(200 * mm, 10 * mm, text)
        except Exception as e:
            logging.getLogger(PDFReportBuilder.__name__).error(f"Error drawing page number: {e}", exc_info=True)
            raise

    @staticmethod
    def _draw_bottomline(canvas: Canvas, doc: BaseDocTemplate) -> None:
        """
        Draw the bottom line text centered at the bottom of the page.
        """
        try:
            bottomline = "ai-generated report".upper()
            text_width = canvas.stringWidth(bottomline, "Helvetica", 9)
            center_position = (doc.pagesize[0] - text_width) / 2
            canvas.drawString(center_position, 10 * mm, bottomline)
        except Exception as e:
            logging.getLogger(PDFReportBuilder.__name__).error(f"Error drawing bottom line: {e}", exc_info=True)
            raise

    def add_explanation_page(self) -> None:
        """
        Add an explanatory page to the PDF report with descriptions of the features.
        """
        try:
            self.elements.append(self.add_title("Explanation of Features"))

            for explanation in explanations:
                self.elements.append(self.add_text(explanation))
                self.elements.append(Spacer(1, 12))

            self.elements.append(self.add_text(
                "This information is intended to help you understand the technical details provided in this report. "
                "If you have any further questions, please feel free to ask for clarification."
            ))
        except Exception as e:
            self.logger.error(f"Error adding explanation page: {e}", exc_info=True)
            raise

    def save(self) -> None:
        """
        Save the PDF report to a file.

        Builds the PDF document with the added elements and saves it to the specified output directory.
        """
        try:
            self.doc.build(self.elements, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
            self.logger.info(f"Saved report to {self.report_filepath}")
            for tmpfile in self.temp_files:
                try:
                    tmpfile_path = Path(tmpfile)
                    tmpfile_path.unlink()
                except OSError as e:
                    self.logger.error(f"Error removing temporary file {tmpfile}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error saving PDF report: {e}", exc_info=True)
            raise
