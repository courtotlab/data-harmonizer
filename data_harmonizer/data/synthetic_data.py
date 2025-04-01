import os
import ast
import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv
from data_harmonizer.data.schema_data import get_schema_features
from collections.abc import Callable

load_dotenv()

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

def retry_gen_data(
    llm_call: Callable[[str, OpenAI, str, int], list[str] | None:], 
    attribute: str, num_syn: int = 7, num_retries: int = 5
) -> list[str] | None:
    """Retry to generate synthetic data

    Parameters
    ----------
    llm_call : function
        LLM function, specific to the field feature, used to generate the synonyms
    attribute : str
        The field feature that needs synonyms generated
    num_syn : int, optional
        The expected number of synonyms generated for the field feature, by default 7
    num_retries : int, optional
        The number of retries before the function gives up, by default 5

    Returns
    -------
    list[str] | None
        Returns a list of strings with a length of num_syn. If the function 
        cannot create a list of strings with num_syn strings within 
        num_retries, None is returned.
    """
    attempt=1
    while attempt<=num_retries:
        try:
            # attempt to turn the string of a list into a literal list
            result_list_attempt = ast.literal_eval(
                llm_call(attribute, num_syn)
            )
            
            if len(result_list_attempt) == num_syn:
                return result_list_attempt
            else:
                attempt=attempt+1
        
        except:
            attempt=attempt+1

    return None

def get_gen_row_data_dict(
    row: tuple, gen_func: dict[str, Callable]
) -> dict[str, list[str]]:
    """Get the generated data for a single row of the dataframe

    Parameters
    ----------
    row : tuple
        Features which we want to generate synonyms for
    gen_func : dict[str, Callable]
        Dictionary where the key represents the feature and 
        the value represents the function used to generate the synonyms

    Returns
    -------
    dict[str, list[str]]
        Dictionary where the key represents the feature and 
        the value represents list of strings represents the generated synonyms

    See Also
    --------
    retry_gen_data
    """

    gen_row_data_dict = {}
    for field_feature in list(gen_func.keys()):
        # represents the value we want to get synonyms for
        attribute = getattr(row, field_feature)           
        
        gen_data = retry_gen_data(gen_func[field_feature], attribute)

        # if generating data for a feature fails, we don't need to 
        # try generating other features
        if gen_data is None:
            gen_row_data_dict = {}
            break
        else:
            gen_row_data_dict[field_feature] = gen_data

    return gen_row_data_dict


def main():
    gen_func = {
        'field_name': field_name_gen_openai,
        'field_description': field_desc_gen_openai
    }

    # used to store synthetic data based on features in gen_func
    gen_data_dict = {
        'reference_field_name': []
    }
    for key in gen_func.keys():
        gen_data_dict[key] = []

    # get target schema info which is used to generate synthetic data
    schema_df = get_schema_features()
    for row in schema_df.itertuples(index=False):
        # generate synonyms for a single row in the data frame and 
        # store synthetic data inside a dictionary
        gen_row_data_dict = get_gen_row_data_dict(row, gen_func)

        # proceed if data is correctly generated
        if len(gen_row_data_dict) == len(gen_func.keys()):        
            
            for key, val in gen_row_data_dict.values:
                # add synthetic data to data dict
                gen_data_dict[key] = (
                    gen_data_dict[key] + val
                )

            # reference_field_name represents the field used to generate the data
            gen_data_dict['reference_field_name'] = (
                gen_data_dict['reference_field_name'] + [getattr(row, 'field_name')]*7
            )

        # TODO: log issue with generating data
        else:
            pass

    synthetic = pd.DataFrame.from_dict(gen_data_dict)
    synthetic.to_csv(f'../data/2_interim/1_synthetic/synthetic_data.csv', index=False)

if __name__=="__main__":
    main()
