import os
import ast
import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

def get_schema_features():
    pass

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def field_name_gen_openai(
        field_name: str, client: OpenAI = client, model_name: str = 'gpt-4o-mini', num_syn: int = 7
    ) -> list[str] | None:
    """Generates a list of synonyms from a given field name using OpenAI

    Parameters
    ----------
    field_name : str
        The field name to generate synonyms for
    client : OpenAI, optional
        Client to connect to OpenAI, by default client
    model_name : str, optional
        OpenAI model to use to generate synonyms, by default 'gpt-4o-mini'
    num_syn : int, optional
        The number of synonyms to return, by default 7

    Returns
    -------
    list[str] | None
        Python list of synonyms
    """

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
    
    return completion.choices[0].message.content

def field_desc_gen_openai(
        field_desc: str, client: OpenAI = client, model_name: str = 'gpt-4o-mini', num_syn: int = 7
    ) -> list[str] | None:
    """Generates a list of synonyms from a given field descriptions using OpenAI

    Parameters
    ----------
    field_desc : str
        The field description to generate synonyms for
    client : OpenAI, optional
        Client to connect to OpenAI, by default client
    model_name : str, optional
        OpenAI model to use to generate synonyms, by default 'gpt-4o-mini'
    num_syn : int, optional
        The number of synonyms to return, by default 7

    Returns
    -------
    list[str] | None
        Python list of synonyms
    """

    prompt = f"""
    You are a helpful medical research assistant.

    Create {num_syn} unique field descriptions, of 2-3 sentences each, to be used in a clinical data set that are synonymous with the given field description. Return the result as a list of strings without comments. Exclude the given field description in the list.

    <<<
    Field description: {field_desc}
    >>>
    """

    completion = client.chat.completions.create(
        model=model_name,
        store=False,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    return completion.choices[0].message.content

def retry_gen_data(llm_call, attribute, num_syn=7, num_retries=5):
    attempt=1
    result_list = None
    while attempt<=num_retries:
        try:
            result_list_attempt = ast.literal_eval(
                llm_call(attribute, num_syn)
            )
            
            if len(result_list_attempt) == num_syn:
                result_list = result_list_attempt
                break
            else:
                attempt=attempt+1
        
        except:
            attempt=attempt+1

    return result_list

def main():
    gen_func = {
        'field_name': field_name_gen_openai,
        'field_description': field_desc_gen_openai
    }

    # used to store synthetic data; may need to be modified if additional features are used in modelling
    gen_data_dict = {
        'field_name': [],
        'field_description': [],
        'reference_field_name': []
    }

    # TODO: load data into a pandas dataframe (schema_df)
    schema_df = get_schema_features()

    for row in schema_df.itertuples(index=False):
        for gen_type in ['field_name', 'field_description']:
            # represents the value we want to get synonyms for
            attribute = getattr(row, gen_type)           

            while True:
                # generate synthetic data
                gen_data = retry_gen_data(gen_func[gen_type], attribute)
            
                # add synthetic data to data dict
                gen_data_dict[gen_type] = gen_data_dict[gen_type] + gen_data

                time.sleep(20)
                break

        # reference_field_name represents the field used to generate the data
        gen_data_dict['reference_field_name'] = (
            gen_data_dict['reference_field_name'] + [getattr(row, 'field_name')]*7
        )

    synthetic = pd.DataFrame.from_dict(gen_data_dict)
    synthetic.to_csv(f'../data/2_interim/1_synthetic/synthetic_data.csv', index=False)

if __name__=="__main__":
    main()