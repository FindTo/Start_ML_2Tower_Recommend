import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, REAL, VARCHAR
from io import StringIO
import gc
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load env variables
load_dotenv()

"""Explicit SQL type mapping for the `feed_encoded` DataFrame.

Expected columns (see `learn_model.get_vector_df`):
['user_id','post_id','time_indicator','hour_sin','hour_cos',
 'weekday_sin','weekday_cos','month_sin','month_cos','is_weekend','target']
"""

FEED_FEATURES_TYPE_MAP = {
                            'user_id': Integer,           # int
                            'post_id': Integer,           # int
                            'time_indicator': REAL,       # float32
                            'hour_sin': REAL,             # float32
                            'hour_cos': REAL,             # float32
                            'weekday_sin': REAL,          # float32
                            'weekday_cos': REAL,          # float32
                            'month_sin': REAL,            # float32
                            'month_cos': REAL,            # float32
                            'is_weekend': Integer,        # 0/1
                            'target': Integer             # 0/1
                        }
                        
"""Type mapping for user_df_encoded_for_2towers.csv

Columns: user_id, main_topic_liked, main_topic_viewed, views_per_user, likes_per_user,
         gender, age, country, city_capital, exp_group
"""
USER_FEATURES_TYPE_MAP = {
    'user_id': Integer,              # int
    'main_topic_liked': Integer,     # label encoded
    'main_topic_viewed': Integer,    # label encoded  
    'views_per_user': REAL,          # float
    'likes_per_user': REAL,          # float
    'gender': Integer,               # int
    'age': Integer,                  # int
    'country': Integer,              # label encoded
    'city_capital': Integer,         # 0/1
    'exp_group': Integer             # int
}                       


"""Type mapping for post_df_encoded_for_2towers.csv

Columns: post_id, topic, text_length, post_likes, post_views, embed_0...embed_127 (128d embeddings)
"""
POST_FEATURES_TYPE_MAP = {
    'post_id': Integer,              # int
    'topic': Integer,                # label encoded
    'text_length': REAL,             # float
    'post_likes': Integer,           # int
    'post_views': Integer,           # int
    # 128 embedding columns (embed_0 to embed_127)
    **{f'embed_{i}': REAL for i in range(128)}  # float32 embeddings
}

# Download post_data table
def get_post_df():

    # Obtain db connection
    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

# Send DF to the DB using chunks - INSERT based method
def df_to_sql(df, name, if_exists: str = 'replace', dtype: dict | None = None, chunksize: int | None = None):

    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    try:
        print((f"to_sql - start writing {name}"))
        effective_chunksize = int(chunksize or os.getenv('CHUNKSIZE', '200000'))
        df.to_sql(
            name,
            con=engine,
            if_exists=if_exists,
            index=False,
            dtype=dtype,
            chunksize=effective_chunksize,
            method='multi',
        )
        print((f"to_sql - {name} successfully written"))
    except Exception as e:
        print((f"to_sql - failed to write {name}: {e}"))
    finally:
        conn.close()

    return 0


# Send .csv file to PostgreSQL using chunks and fast COPY method
def csv_to_sql(csv_path: str, table_name: str, type_map: dict, chunksize=5000, sep=";"):

    engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
    metadata = MetaData()

    try:
        # Read .csv header
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(sep)

        # Create new SQL table
        columns_def = [Column(col, type_map.get(col, VARCHAR(255))) for col in header]
        table = Table(table_name, metadata, *columns_def)
        metadata.drop_all(engine, [table])
        metadata.create_all(engine)
        print(f"Table {table_name} recreated")

        # Calculate num of lines and chunks for tqdm progress bar
        total_lines = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        total_chunks = (total_lines + chunksize - 1) // chunksize

        # Open raw_connection: low-level approach
        conn = engine.raw_connection()
        cursor = conn.cursor()

        for chunk in tqdm(
            pd.read_csv(csv_path, sep=sep, chunksize=chunksize, encoding="utf-8"),
            total=total_chunks,
            desc="COPY chunks",
            unit="chunk"
        ):
            # Conbert types
            for col in chunk.columns:
                if col in type_map:
                    if type_map[col] == Integer:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("int32")
                    elif type_map[col] == REAL:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
                    elif isinstance(type_map[col], VARCHAR):
                        chunk[col] = chunk[col].astype(str)

            # Ignore NaNs, save to buffer
            buffer = StringIO()
            chunk.to_csv(buffer, sep=sep, index=False, header=False)
            buffer.seek(0)

            # Copy buffer to the SQL table
            cursor.copy_expert(
                f"COPY {table_name} FROM STDIN WITH (FORMAT csv, DELIMITER '{sep}')",
                buffer
            )

            # Delete temp buffer, chunk and collect garbage
            buffer.close()
            del chunk
            gc.collect()

        # Finish connection
        conn.commit()
        cursor.close()
        conn.close()
        print("COPY completed successfully")

    finally:
        engine.dispose()

# Load big DF from the DB using chunks - ORM approach
def load_sql_df(table_name,
                type_map:dict) -> pd.DataFrame:
    engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
    chunksize = int(os.getenv('CHUNKSIZE'))
    chunks = []

    try:
        print(f"from sql - start loading {table_name}")

        #
        with engine.connect() as conn:
            iterator = pd.read_sql(table_name, conn, chunksize=chunksize)
            for chunk in iterator:
                # Convert types by TYPE_MAP in the chunk
                for col, col_type in type_map.items():
                    if col in chunk.columns:
                        if col_type == Integer:
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('int32')
                        elif col_type == VARCHAR(50):
                            chunk[col] = chunk[col].astype(str)
                        elif col_type == REAL:
                            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

                chunks.append(chunk)

        print(f"from sql - {table_name} loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Data loading error: {e}")

    df = pd.concat(chunks, ignore_index=True)
    total_memory = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nMemory size for the downloaded df: {total_memory:.2f} MB")
    print(df.shape)
    return df


def load_features(features_name) -> pd.DataFrame:

    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print((f"from sql - start loading {features_name}"))
        for chunk_dataframe in pd.read_sql(features_name,
                                           conn, chunksize=int(os.getenv('CHUNKSIZE'))):

            chunks.append(chunk_dataframe)

        print((f"from sql - {features_name} loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Data loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

