import os
import ast
from openai import OpenAI

# TODO: load environment variables

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def field_name_gen_openai(
        field_name: str, client: OpenAI = client, model_name: str = 'gpt-4o-mini', num_syn: int = 7
    ) -> list[str] | None:
    prompt = f"""
    You are a helpful medical research assistant.
    
    Create {num_syn} wholly unique field names, formatted in either snake case and camel case, to be used in a clinical data set that are synonymous with the given field name. Return the result as a list of strings without comments. Exclude the given field name in the list.

    <<<
    Field name: {field_name}
    >>>
    """

    completion = client.chat.completions.create(
        model=model_name,
        store=False,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    attempt=1
    while attempt<=5:
        try:
            completion.choices[0].message.content
            result_list = ast.literal_eval(
                completion.choices[0].message.content
            )
            break
        except:
            attempt=attempt+1

    return result_list