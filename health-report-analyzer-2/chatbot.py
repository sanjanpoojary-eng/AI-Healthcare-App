import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure the API key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

def generate_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

def validate_response(findings, user_query, initial_response):
    validation_prompt = f"""
You are a healthcare assistant. Given the following key findings from a healthcare report,
a user's question, and an initial response, please evaluate whether the response is accurate,
relevant to the report, and provides genuine information.

Key findings:
{findings}

User's question: {user_query}

Initial response: {initial_response}

Evaluation:
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(validation_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error validating response: {e}"