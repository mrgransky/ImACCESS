import pandas as pd
import re

def extract_year(description, raw_doc_date):
    if not pd.isna(raw_doc_date):  # Check if raw_doc_date is missing (None or NaN)
        return raw_doc_date

    if description is None:  # Check if description is None
        return None
    print(f"neither... check description")
    year_pattern = r'\b\d{4}\b'
    match = re.search(year_pattern, description)
    if match:
        return match.group()
    else:
        return None

# Sample DataFrame
data = {
    'description': [
        "west neebish channel lights michigan 22 date may 1916 district 11",
        None,
        "sandusky bay range beacon west ohio date 1885 district 10",
        "whitehall narrows vermont date october 13 1913 district",
        "bolivar point light station texas camera station 300 ft northwesterly date 11 1917 district",
        "st george reef light station california district 18",
        None,
        "this description does not have date",
        "looking southeast 50 ft date august 21 1919 district",
    ],
    'raw_doc_date':[
        1986,
        None,
        None,
        None,
        1945,
        1972,
        1972,
        None,
        2018,
    ],
}

df = pd.DataFrame(data)

df['doc_date'] = df.apply(lambda row: extract_year(row['description'], row['raw_doc_date']), axis=1)
print(df)