# Data Harmonizer
## Project Organization Summary
    ├── .env
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── data
    │   ├── 1_raw
    │   ├── 2_interim
    │   └── 3_processed
    ├── data_harmonizer
    │   ├── data
    │   │   ├── schema_data.py
    │   │   ├── split_data.py
    │   │   └── synthetic_data.py
    │   ├── modeling
    │   │   └── train.py
    │   ├── __init__.py
    │   └── main.py
    ├── logs
    ├── models
    └── tests
        ├── test_linkml.yaml   
        ├── test_schema_data.py  
        ├── test_split_data.py 
        └── test_synthetic_data.py
