import pandas as pd
import numpy as np
import re
import logging
import time
import sqlite3
from sklearn.feature_extraction.text import HashingVectorizer
import swifter
import os
from dotenv import load_dotenv
import gzip
import requests
from tqdm import tqdm

# Setting Environment Variables

load_dotenv()
URL = os.getenv("URL")

# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def log_time(msg, start_time):
    elapsed = time.time() - start_time
    logging.info(f"{msg} — completed in {elapsed:.2f} seconds")

# Downloading file & unzipping

from tqdm import tqdm

def download_file(url):
    start = time.time()
    logging.info("⬇️ Downloading file...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if "content-disposition" in response.headers:
            content_disposition = response.headers["content-disposition"]
            filename = content_disposition.split("filename=")[-1].strip('\"')
        else:
            filename = url.split("/")[-1]
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        with open(filename, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=filename
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        logging.info(f"✅ File '{filename}' downloaded successfully.")
        log_time("✅ File downloading completed", start)
        return filename

    except requests.RequestException as e:
        logging.error(f"❌ Download failed: {e}")
        raise

def gunzip(source_filepath, dest_filepath, block_size=65536):
    start = time.time()
    logging.info("📂 Unzipping file...")

    try:
        file_size = os.path.getsize(source_filepath)
        with gzip.open(source_filepath, 'rb') as s_file, \
             open(dest_filepath, 'wb') as d_file, \
             tqdm(total=file_size, unit='B', unit_scale=True, desc="Unzipping") as bar:

            while True:
                block = s_file.read(block_size)
                if not block:
                    break
                d_file.write(block)
                bar.update(len(block))

        logging.info(f"✅ Unzipped to '{dest_filepath}'")
        log_time("✅ File unzipping completed", start)
    except (OSError, IOError) as e:
        logging.error(f"❌ Unzipping failed: {e}")
        raise

# Load & Filter CSV

def load_and_filter_csv(path="Tyroo-dummy-data.csv", chunk_size=50000):
    start = time.time()
    logging.info("🔍 Reading CSV in chunks...")

    def print_bad_rows(df):
        col_names = set(df.columns.astype(str))
        mask = df.astype(str).isin(col_names)
        bad_row_indices = mask.any(axis=1)
        logging.info(f"⚠️  Found {bad_row_indices.sum()} suspicious rows.")
        return bad_row_indices

    chunks = pd.read_csv(path, chunksize=chunk_size)
    df_list = []

    for i, chunk in enumerate(chunks):
        logging.info(f"📦 Processing chunk {i + 1}")
        bad_rows = print_bad_rows(chunk)
        chunk = chunk[~bad_rows].reset_index(drop=True)
        df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)
    log_time("✅ CSV loading and filtering", start)
    return df

# Basic Cleanup

def basic_cleanup(df):
    start = time.time()
    logging.info("🧹 Starting basic cleanup...")

    num_cols = [
        'platform_commission_rate', 'product_commission_rate',
        'bonus_commission_rate', 'promotion_price', 'current_price',
        'price', 'discount_percentage', 'number_of_reviews',
        'rating_avg_value', 'seller_rating'
    ]
    text_cols = [
        'venture_category3_name_en', 'venture_category2_name_en', 'venture_category1_name_en',
        'venture_category_name_local', 'brand_name', 'business_type', 'business_area',
        'product_name', 'seller_name'
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df.fillna({
        'is_free_shipping': '0',
        'availability': 'out of stock',
        'deeplink': '',
        'product_url': '',
        'seller_url': '',
        'description': np.nan,
        **{col: 'Unknown' for col in text_cols}
    }, inplace=True)

    df['is_free_shipping'] = df['is_free_shipping'].astype(bool)
    df['availability'] = df['availability'].map({'in stock': True, 'out of stock': False})

    image_cols = [col for col in df.columns if 'img' in col or 'image_url' in col]
    df[image_cols] = df[image_cols].fillna('')
    df['description'] = df['description'].fillna(df['product_name'])

    log_time("✅ Basic cleanup", start)
    return df

# Clean Description

def clean_descriptions(df):
    start = time.time()
    logging.info("🧼 Cleaning descriptions...")

    # Junk pattern filtering
    junk_regex = r'^(&nbsp;)+$|^-+$|^\.{1,2}$|^welcome to my shop.*$|^www.*$|^$|^no description currently.*$|^_*$'
    desc_str = df['description'].fillna('').str.strip().str.lower()
    junk_mask = desc_str.str.match(junk_regex)
    df.loc[junk_mask, 'description'] = df.loc[junk_mask, 'product_name']

    def clean_single_desc(row):
        desc = row['description']
        product_name = row['product_name']
        if pd.isna(desc):
            return product_name
        text = str(desc).strip().lower()
        text = re.sub(r'&[a-z]+;', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        if len(text) < 10:
            return product_name
        return text

    df['description'] = df.swifter.apply(clean_single_desc, axis=1)
    log_time("✅ Description cleaning", start)
    return df

# Normalize Text

def normalize_text(df):
    start = time.time()
    logging.info("🔠 Normalizing text fields...")

    def fast_clean(series):
        return (
            series.fillna('')
                  .str.encode('ascii', errors='ignore').str.decode('ascii')
                  .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip()
                  .str.lower()
        )

    df['product_name'] = fast_clean(df['product_name'])
    df['description'] = fast_clean(df['description'])

    log_time("✅ Text normalization", start)
    return df

# Vectorize & Match

def vectorize_and_compare(df):
    start = time.time()
    logging.info("📊 Vectorizing and computing similarities...")

    vectorizer = HashingVectorizer(
        stop_words='english',
        n_features=2**10,
        alternate_sign=False,
        norm='l2'
    )

    name_vecs = vectorizer.transform(df['product_name'])
    desc_vecs = vectorizer.transform(df['description'])

    similarities = (name_vecs.multiply(desc_vecs)).sum(axis=1).A1
    df['similarity'] = similarities
    df.loc[df['similarity'] == 0.0, 'description'] = df.loc[df['similarity'] == 0.0, 'product_name']
    df.drop(columns=['similarity'], inplace=True)

    log_time("✅ Similarity vectorization", start)
    return df

# Save to SQLite

def save_to_sqlite(df, db_name="transformed_data.db", table_name="products_cleaned"):
    start = time.time()
    logging.info("💾 Saving to SQLite...")

    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    log_time(f"✅ Data saved to SQLite table '{table_name}'", start)

def main():
    if not URL:
        raise ValueError("❌ URL environment variable not found.")
    zip_file_name = download_file(URL)
    file_name = zip_file_name.removesuffix(".gz")
    gunzip(zip_file_name, file_name)
    df = load_and_filter_csv(path=file_name)
    df = basic_cleanup(df)
    df = clean_descriptions(df)
    df = normalize_text(df)
    df = vectorize_and_compare(df)
    save_to_sqlite(df)
    logging.info("🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
