
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


#  Define Project Paths (Relative to the GitHub Repo Root)
# This replaces the Google Drive paths
df = pd.read_csv("Tourism.Project/data/tourism.csv")
df.drop(['CustomerID'],axis=1,inplace=True)


#  Perform Data Cleaning
def clean_data(data):
    # Fix Gender inconsistency
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

    # Standardize MaritalStatus
    if 'MaritalStatus' in data.columns:
        data['MaritalStatus'] = data['MaritalStatus'].replace('Unmarried', 'Single')

    # Drop unnecessary identifier columns
    cols_to_drop = ['Unnamed: 0', 'CustomerID']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Fill missing values with median
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].median())
    if 'MonthlyIncome' in data.columns:
        data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())

    return data

df_cleaned = clean_data(df)

# Separating target column
X = df_cleaned.drop(['ProdTaken','PitchSatisfactionScore','ProductPitched','NumberOfFollowups','DurationOfPitch','Agebin','Incomebin'],axis=1)
X = pd.get_dummies(X,drop_first=True)
y = df_cleaned['ProdTaken']

#  Split the cleaned dataset (Stratify ensures Prodtaken ratio stays same)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    df_cleaned,
    test_size=0.2,
    random_state=42,
    stratify=y
)


Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="casatish.kaikolar/Tourism.Project/Data",
        repo_type="dataset",
    )

# Retrieve the token from GitHub Secrets (passed as an environment variable in pipeline.yml)
hf_token = os.getenv('HF_TOKEN')

if hf_token:
    processed_dataset.push_to_hub(
        "casatishkaikolar/Tourism.Project/Data/VisitWithUs-Tourism-Dataset-Processed",
        token=hf_token
    )
    print("Cleaned data successfully pushed!")
else:
    print("Error: HF_TOKEN not found in environment variables.Check your GitHub")
