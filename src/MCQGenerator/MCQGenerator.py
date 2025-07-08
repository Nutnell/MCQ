import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.MCQGenerator.utils import read_file, get_table_data
from src.MCQGenerator.logger import logging

# Updated LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# Old: from langchain_core.pydantic_v1 import BaseModel, Field # Removed this line
# New: Import BaseModel and Field directly from pydantic (for regular models)
# And import RootModel for models that were using __root__
from pydantic import BaseModel, Field, RootModel # <--- ADD ROOTMODEL HERE
import time # For retries

# Load environment variables
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
    temperature=0.7, # Adjusted temperature
)

# --- Define Pydantic Model for Quiz Output ---
# These remain BaseModel as they have named fields
class MCQOption(BaseModel):
    a: str = Field(description="Choice A")
    b: str = Field(description="Choice B")
    c: str = Field(description="Choice C")
    d: str = Field(description="Choice D")

class MCQItem(BaseModel):
    no: str = Field(description="Question number")
    mcq: str = Field(description="The multiple choice question")
    options: MCQOption = Field(description="Available options for the MCQ")
    correct: str = Field(description="The correct answer option (e.g., 'a', 'b', 'c', 'd')")

# ===>>> CORRECTED QuizOutput MODEL DEFINITION <<<===
class QuizOutput(RootModel[dict[str, MCQItem]]): # <--- CHANGE THIS LINE
    # Root models directly take the type they wrap as a generic argument.
    # No need for __root__ = Field(...) anymore.
    """Dictionary of MCQs, where keys are question numbers."""
# ===>>> END OF CORRECTION <<<===


# Prompt for quiz generation
parser_quiz = JsonOutputParser(pydantic_object=QuizOutput)


template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response as a JSON object with the following structure:
{format_instructions}

Ensure to make {number} MCQs.
"""

quiz_generator_prompt = PromptTemplate(
    template=template,
    input_variables=["text", "number", "subject", "tone"],
    partial_variables={"format_instructions": parser_quiz.get_format_instructions()},
)

# Quiz generation chain - now outputs a raw AIMessage object
quiz_chain_raw_message = quiz_generator_prompt | llm

# Prompt for quiz evaluation
template2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
If the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and adjust the tone accordingly.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate.from_template(template2)

# Quiz review chain
review_chain = quiz_evaluation_prompt | llm | StrOutputParser()


# Combine chains with a bridging function
def to_review_input(quiz_dict: dict, subject: str) -> dict:
    # Convert the dictionary quiz back to a JSON string for the review chain's input
    return {"quiz": json.dumps(quiz_dict), "subject": subject}


# Final chain composition with retry logic
def generate_evaluate_chain(inputs: dict, max_retries=3) -> dict:
    quiz_dict = None
    quiz_raw_str = ""  # To store the raw LLM output for debugging
    for attempt in range(max_retries):
        try:
            # Get the raw AIMessage object from the LLM
            logging.info(f"Attempt {attempt + 1}: Generating quiz...")
            llm_response_message = quiz_chain_raw_message.invoke(inputs)

            # Extract the content string from the AIMessage
            if hasattr(llm_response_message, "content"):
                quiz_raw_str = llm_response_message.content
            else:
                raise ValueError("LLM response did not have a 'content' attribute.")

            logging.info(f"Raw LLM quiz output content:\n{quiz_raw_str}")

            # Try to parse the raw string using the Pydantic parser
            quiz_dict_pydantic = parser_quiz.parse(quiz_raw_str)
            quiz_dict = (
                quiz_dict_pydantic  # quiz_dict_pydantic is already the dictionary
            )

            logging.info("Quiz parsed successfully.")
            break  # Exit loop if successful
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error on attempt {attempt + 1}: {e}")
            logging.error(f"Problematic string: {quiz_raw_str}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in 2 seconds...")
                time.sleep(2)  # Wait before retrying
            else:
                logging.error(
                    "Max retries reached for quiz generation (JSON parsing failed)."
                )
                raise e  # Re-raise if all retries fail
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during quiz generation on attempt {attempt + 1}: {e}"
            )
            if attempt < max_retries - 1:
                logging.info(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logging.error(
                    "Max retries reached for quiz generation due to unexpected error."
                )
                raise e

    if quiz_dict is None:
        raise Exception("Failed to generate and parse quiz after multiple retries.")

    logging.info("Starting quiz review...")
    review = review_chain.invoke(to_review_input(quiz_dict, inputs["subject"]))
    logging.info("Quiz review completed.")

    return {"quiz": quiz_dict, "review": review}
