import os
from ..compoundai.llm import initialize_services

def config(
    db_path="db_path.sqlite",
    openai_api_key="OpenAI_API_KEY",
    anthropic_api_key="ANTHROPIC_API_KEY",
    together_ai_api_key="TOGETHER_API_KEY",
    gemini_api_key="GEMINI_API_KEY",
):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key 
    os.environ['TOGETHER_API_KEY'] = together_ai_api_key
    os.environ['GEMINI_API_KEY'] = gemini_api_key
    os.environ['DB_PATH'] = db_path

    task_name = "test"  # Read environment variable
    initialize_services(task_name)

    pass