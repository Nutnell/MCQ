import os
import pypdf
import json
import traceback
from pypdf import PdfReader


def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            return text
        except Exception as e:
            raise Exception("error reading the PDF file") from e
    elif file.name.endswith(".txt"):
        # For Streamlit uploaded file, .read() might return bytes,
        # so decode it. This was already correct.
        return file.read().decode("utf-8")
    else:
        raise Exception("unsupported file format only pdf and text file suppoted")


# The input `quiz_data` to this function is ALREADY expected to be a dictionary
# because MCQGenerator.py now handles the JSON parsing using Pydantic.
def get_table_data(quiz_data):
    try:
        # `quiz_data` is already a dictionary, directly use it.
        # No need for json.loads(quiz_str) anymore as the LLM output is parsed before reaching here.
        quiz_table_data = []

        # Iterate over the quiz dictionary and extract the required information
        for key, value in quiz_data.items(): # Use quiz_data directly
            mcq = value["mcq"]
            options = " || ".join(
                [
                    f"{option}-> {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )

            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        # Return None or an empty list if there's an issue processing the dict
        # This will prevent the pd.DataFrame error.
        return None