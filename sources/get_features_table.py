import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine
import os

def get_post_df():

    # Obtain db connection
    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

def df_to_sql(df, name):

    # Try to write DF into the db by chunks
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    try:

        print(("to_sql - start writing"))
        df.to_sql(name,
                  con=engine,
                  if_exists='replace',
                  index=False)
        print(("to_sql - successfully written"))
        conn.close()

    except:

        print(("to_sql - failed to write"))
        conn.close()

    return 0


def csv_to_sql(csv_name,table_name):


    # Try to write csv file into the db by chunks
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    try:

        print(("to_sql - start writing"))

        chunksize = int(os.getenv('CHUNKSIZE'))

        for chunk in pd.read_csv(csv_name, chunksize=chunksize):

            chunk.to_sql(table_name, engine, if_exists='append', index=False, method='multi')

        print(("to_sql - successfully written"))
        conn.close()

    except:

        print(("to_sql - failed to write"))
        conn.close()

    return 0


def load_features(features_name) -> pd.DataFrame:

    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print(("from sql - start loading"))
        for chunk_dataframe in pd.read_sql(features_name,
                                           conn, chunksize=int(os.getenv('CHUNKSIZE'))):

            chunks.append(chunk_dataframe)

        print(("from sql - loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Data loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

