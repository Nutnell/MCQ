import os
import json
import traceback
import pandas as pd
import pathlib
from dotenv import load_dotenv
from src.MCQGenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI
from src.MCQGenerator.MCQGenerator import generate_evaluate_chain
from src.MCQGenerator.logger import logging
from openai import OpenAI

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

current_dir = pathlib.Path(__file__).parent.resolve()
response_file_path = current_dir / "Response.json"
with open(response_file_path, "r") as file:
    RESPONSE_JSON = json.load(file)


# creating a title for the app
st.title("MCQs Creator Application with Langchain 🐦⛓️")

# create a form using st.form
with st.form("user_inputs"):
    # File input
    uploaded_file = st.file_uploader("Upload a PDF or Txt File")

    # Input Fields
    mcq_count = st.number_input("No. of MCQ's", min_value=3, max_value=50)

    # subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone = st.text_input(
        "Complexity level of Questions", max_chars=20, placeholder="simple"
    )

    # Add Button
    button = st.form_submit_button("Create MCQ's")

    # check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                # count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON),
                        }
                    )
                    st.success("MCQ's Created Successfully")
                    # st.write(response)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("ERROR!!")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # Display the review in a text box as well
                        else:
                            st.error("Error in the table data")
                    else:
                        st.write(response)
