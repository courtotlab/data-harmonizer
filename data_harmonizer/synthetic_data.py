import os
from openai import OpenAI

# TODO: load environment variables

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)