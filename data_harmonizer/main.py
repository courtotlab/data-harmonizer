import os
from dotenv import load_dotenv
from data_harmonizer.data.schema_data import get_schema_features

load_dotenv()

TARGET_LINKML_PATH = os.getenv('TARGET_LINKML_PATH')
SOURCE_LINKML_PATH = os.getenv('SOURCE_LINKML_PATH')

def main():

    target = get_schema_features(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', TARGET_LINKML_PATH)
    ))

    source = get_schema_features(os.path.abspath(
        os.path.join(os.path.dirname( __file__ ), '..', SOURCE_LINKML_PATH)
    ))

if __name__ == '__main__':
    main()
