### About the Solution

This Python script automates the process of downloading, unzipping, cleaning, and transforming large-scale e-commerce product data. It also stores the final, cleaned data into a local SQLite database.

### Features
 - Downloads .gz-compressed CSV data via URL

 - Unzips and reads in chunks to handle large files

 - Cleans numeric and text fields, replaces junk descriptions

 - Normalizes text for further NLP use

 - Computes similarity between product name and description to cleaning irrelevant description

 - Saves the cleaned output into an SQLite database


### Project Structure
```
├── main.py               # Main script
├── .env                  # Contains the URL for data download
├── transformed_data.db   # Output SQLite DB (we get after running main.py)
└── README.md             # This file
```

### Installation Steps

You must have uv installed.
Visit the docs for installation.
https://docs.astral.sh/uv/getting-started/installation/

```
uv venv
source .venv/bin/activate # if linux
.venv/Scripts/activate # if windows
uv sync
```

Alternatively using PIP,
```
python -m venv venv
source .venv/bin/activate # if linux
.venv/Scripts/activate # if windows
pip install -r requirements.txt
```

Then create a .env file and add URL = <YOUR_FILE_URL>
In this case the assignment URL.

### Data Download & Processing Pipeline


Run ```python main.py``` in terminal. This contains all data cleaning and transformation steps.

The database schema is stored in ```db_schema.sql``` file.
