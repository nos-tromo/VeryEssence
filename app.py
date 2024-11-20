import argparse
import logging
import os
import threading

from flask import Flask, redirect, render_template, request, Response, url_for

from config.pdfreport_cfg import explanations
from cli import parse_arguments, run_pipeline


app = Flask(__name__)

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists('uploads'):
    logger.info(f"Upload folder {UPLOAD_FOLDER} does not exist. Creating one...")
    os.makedirs('uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home() -> str | tuple[str, int]:
    """
    Render the home page with a list of files available for processing.

    :return: Rendered HTML template for the home page or an error message.
    """
    try:
        logger.info("Serving the home page.")
        # Get the list of files in the uploads folder, excluding hidden files
        files = sorted([f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if not f.startswith(".")])
        return render_template("index.html", files=files)
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return str(e), 500


def run_task_in_background(args: argparse.Namespace) -> None:
    """
    Run the processing pipeline in a separate thread.

    :param args: Parsed arguments to pass to the pipeline.
    """
    logger.info("Starting task in the background...")
    try:
        run_pipeline(args)
        logger.info("Pipeline task finished.")
    except Exception as e:
        logger.error(f"Error in running pipeline: {e}")


@app.route("/upload", methods=["POST"])
def upload_file() -> Response | tuple[str, int]:
    """
    Handle file processing based on the form input, start the pipeline task in the background,
    and redirect the user to a results page or home page.

    :return: Redirection to another page with a message or an error.
    """
    try:
        # Get the filename from the form
        file_name = request.form.get("file")
        if not file_name:
            logger.warning("No file provided.")
            return "No file provided", 400

        # Construct the file path and check if it exists
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "File not found", 404

        # Get other form parameters for start, end time, language, model size, and task type
        start_time = request.form.get("start_time", "00:00:00")
        end_time = request.form.get("end_time", "00:00:00")
        source_language = request.form.get("source_language", "de")
        target_language = request.form.get("target_language", "de")
        model_size = request.form.get("model_size", "default")
        task = request.form.get("task", "1")

        # Construct the argument list to simulate a CLI call
        args_list = [
            "--file", file_path,
            "--start", start_time,
            "--end", end_time,
            "--src-lang", source_language,
            "--tgt-lang", target_language,
            "--model", model_size,
            "--task", task
        ]

        # Check and add optional checkboxes for analysis features
        if request.form.get("words"):
            args_list.append("--words")
        if request.form.get("topics"):
            args_list.append("--topics")
        if request.form.get("entities"):
            args_list.append("--entities")
        if request.form.get("sentiment"):
            args_list.append("--sentiment")
        if request.form.get("diarization"):
            args_list.append("--diarization")
        if request.form.get("summarization"):
            args_list.append("--summarization")
        if request.form.get("full_analysis"):
            args_list.append("--full-analysis")

        # Add optional header and username
        header = request.form.get("header")
        if header:
            args_list.extend(["--header", header])

        user = request.form.get("user")
        if user:
            args_list.extend(["--user", user])

        # Parse the arguments and log the start of the pipeline
        args = parse_arguments(args_list)
        logger.info("Arguments parsed. Running the VeryEssence pipeline now...")

        # Start the pipeline in a separate thread
        task_thread = threading.Thread(target=run_task_in_background, args=(args,))
        task_thread.start()

        # Option 1: Redirect to a results page (results.html should be created)
        return redirect(url_for("results", message="Processing started!"))

    except Exception as e:
        logger.error(f"Error in upload route: {e}")
        return str(e), 500


@app.route("/results")
def results() -> str | tuple[str, int]:
    """
    Display the results page with a message.

    :return: Rendered HTML template for the results page or an error message.
    """
    try:
        message = request.args.get("message", "No message provided.")
        return render_template("result.html", message=message)
    except Exception as e:
        logger.error(f"Error in results route: {e}")
        return "An error occurred while rendering the results page.", 500


@app.route("/view-license")
def view_license() -> str | tuple[str, int]:
    """
    Display the content of the license file.

    :return: Rendered HTML template for the license or an error message.
    """
    try:
        with open("LICENSE", "r") as license_file:
            license_content = license_file.read()
        return render_template("license.html", content=license_content)
    except FileNotFoundError:
        return "License file not found.", 404


@app.route("/view-disclaimer")
def view_disclaimer() -> str | tuple[str, int]:
    """
    Display the disclaimer page.

    :return: Rendered HTML template for the disclaimer or an error message.
    """
    try:
        return render_template("disclaimer.html")
    except FileNotFoundError:
        return "Disclaimer file not found.", 404


@app.route("/view-features")
def view_features() -> str | tuple[str, int]:
    """
    Display the features page with explanations.

    :return: Rendered HTML template for the features or an error message.
    """
    try:
        features = "\n".join(explanations)
        return render_template("features.html", content=features)
    except FileNotFoundError:
        return "Disclaimer file not found.", 404


if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(debug=True)
