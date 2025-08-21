from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import desc
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet
from typing import List
from datetime import datetime
from get_features_table import load_features, get_post_df
from get_model import load_models
from learn_model import (ItemDataset,
                         ITEM_CAT_FEATURES,
                         USER_CAT_FEATURES)
from dotenv import load_dotenv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Load environment variables
load_dotenv()

def get_db():
    with SessionLocal() as db:
        return db

# Get item embeddings by post features df using NN model inference mode
def generate_item_embeddings(model, item_dataset, device='cpu'):
    model.eval()
    model.to(device)

    all_item_ids = []
    all_embeddings = []

    loader = DataLoader(item_dataset, batch_size=256)

    with torch.no_grad():
        for batch in tqdm(loader, desc='Item Embeddings', leave=False):
            item_cat = batch[0].to(device)         # shape [B, C]
            item_num = batch[1].to(device)         # shape [B, N]
            item_ids = batch[2]                    # для связи

            item_embeds = model.item_tower(item_cat, item_num)
            all_item_ids.extend(item_ids)
            all_embeddings.append(item_embeds.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    id_tensor = torch.stack(all_item_ids)
    return id_tensor.numpy().astype('int32'), embeddings_tensor.numpy().astype('float32')

# Get user embedding by user idm user df features and timestamp using NN model inference mode
def get_user_embedding(user_id, timestamp, user_df, model, device='cpu'):
    user_num_features = [x for x in user_df.columns.to_list() if x not in USER_CAT_FEATURES.keys()]

    # Get time features encoded to sin/cos
    def prepare_time_features(timestamp):
        # Sin-cos time encoding
        def encode_cyclic(df, col, period):
            sin = np.sin(2 * np.pi * df[col] / period)
            cos = np.cos(2 * np.pi * df[col] / period)
            return sin, cos

        # Time features dict
        time = {}

        # Datetime object init
        dt = datetime.strptime("2021-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

        # Convert timestamp to datetime object
        if isinstance(timestamp, datetime):
            dt=timestamp
        elif isinstance(timestamp, str):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        # Add time features
        time['day_of_week'] = dt.weekday()
        time['hour'] = dt.hour
        time['month'] = dt.month
        time['day'] = dt.day
        time['year'] = dt.year

        # Total accumulated time from 2021 up to the current, in hours
        time['time_indicator'] = (np.log1p((time['year'] - 2021) * 360 * 24 +
                                           time['month'] * 30 * 24 +
                                           time['day'] * 24 +
                                           time['hour'])
                                  )

        # New sin-cos encoded features
        time['hour_sin'], time['hour_cos'] = encode_cyclic(time, 'hour', 24)
        time['weekday_sin'], time['weekday_cos'] = encode_cyclic(time, 'day_of_week', 7)
        time['month_sin'], time['month_cos'] = encode_cyclic(time, 'month', 12)
        time['is_weekend'] = 1 if time['day_of_week'] in [5, 6] else 0

        time_features_columns = ['time_indicator',
                                 'hour_sin',
                                 'hour_cos',
                                 'weekday_sin',
                                 'weekday_cos',
                                 'month_sin',
                                 'month_cos',
                                 'is_weekend',
                                 ]

        # Delete unnecessary features
        time_list = [time[k] for k in time_features_columns if k in time]

        return time_list

    try:
        # Find user by input ID
        user_row = user_df.loc[user_id]

    except KeyError:
        # if incorrect user_id - 404 error
        raise HTTPException(404, detail=f"user {user_id} not found")

    # Convert to tensors using correct datatypes
    user_cat = torch.tensor([[user_row[cat] for cat in USER_CAT_FEATURES]], dtype=torch.long).to(device)
    user_num = torch.tensor([[user_row[num] for num in user_num_features]], dtype=torch.float32).to(device)
    time_num = torch.tensor([prepare_time_features(timestamp)], dtype=torch.float32).to(device)

    # Retrieve embedding
    model.eval()
    with torch.no_grad():
        embedding = model.user_tower(user_cat, user_num, time_num)

    return embedding.cpu().numpy().squeeze().astype('float32')

# Get normalized np vectors
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Sigmoid np function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Retrieve k most probably liked posts with ids, probabilities and cosine scores
def recommend_top_k(user_embedding: np.ndarray,
                    item_embeddings: np.ndarray,
                    item_ids: np.ndarray,
                    k: int = 5):
    # Normalize user's and item embeddings
    user_embedding_norm = user_embedding / np.linalg.norm(user_embedding)
    item_embeddings_norm = normalize_vectors(item_embeddings)

    # Calculate cosine similarity
    scores = item_embeddings_norm @ user_embedding_norm

    # Receive probability using sigmoid and scaled logit
    probabilities = sigmoid(scores * 5)

    # Sort by probability, descend
    sorted_idx = np.argsort(-probabilities)
    top_probs = probabilities[sorted_idx][:k]
    top_scores = scores[sorted_idx][:k]
    top_item_ids = item_ids[sorted_idx][:k]

    return top_item_ids, top_probs, top_scores

# Load user df with features and set index as user_id
user_df = load_features(os.getenv('USER_FEATURES_NN'))
user_df['user_id_idx'] = user_df['user_id']
user_df.set_index('user_id_idx', inplace=True)

# Load post df and create ItemDataset
post_df = load_features(os.getenv('POST_FEATURES_NN'))
item_dataset = ItemDataset(post_df,ITEM_CAT_FEATURES )

# Load 2Tower model
model = load_models(os.getenv('NN_MODEL_NAME'))

# Load original post df - for json answer
post_original_df = get_post_df()

# Create post embedding vectors as np array and post_id column
item_ids_np, item_embeds_np = generate_item_embeddings(model,item_dataset)

# Create app instance
app = FastAPI()

@app.get("/user/{id}", response_model = UserGet)
def get_user(id: int, db: Session = Depends(get_db)):

    data = db.query(User).filter(User.id == id).first()

    if data == None:

        raise HTTPException(404, "user not found")

    else:
        logger.info(data)
        return data

@app.get("/post/{id}", response_model = PostGet)
def get_post(id: int, db: Session = Depends(get_db)):

    data = db.query(Post).filter(Post.id == id).first()

    if data == None:

        raise HTTPException(404, "post not found")

    else:

        return data

@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    data = db.query(Feed).filter(Feed.user_id == id).order_by(desc(Feed.time)).limit(limit).all()
    logger.info(data)

    return data

@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    return db.query(Feed).filter(Feed.post_id == id).order_by(desc(Feed.time)).limit(limit).all()

# @app.get("/post/recommendations/", response_model=List[PostGet])
# def get_post_recomended(id: int, limit: int = 10, db: Session = Depends(get_db)):
#
#     post_liked = db.query(Post, func.count(Feed.user_id)).select_from(Feed
#                           ).join(Post, Feed.post_id == Post.id).filter(Feed.action == 'like'
#                                    ).group_by(Post
#                                               ).order_by(desc(func.count(Feed.user_id))).limit(limit).all()
#
#     return [x[0] for x in post_liked]

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    logger.info("Endpoint enter")

    # Create user embedding - timestamp convertion and single NN reference
    user_embed = get_user_embedding(id, time, user_df, model)

    logger.info("User ebedding created")

    post_ids, post_prob, post_scor  = recommend_top_k(user_embed,
                                                      item_embeds_np,
                                                      item_ids_np,
                                                      5)

    logger.info("Top posts are ready")
    # First n=limit posts from pull with max like probability
    posts_recommend = post_ids.tolist()
    post_prob = post_prob.tolist()

    logger.info(posts_recommend)
    logger.info(post_prob)

    posts_recommend_list = []

    # Making response by Pydantic using the obtained post IDs
    for i in posts_recommend:

        posts_recommend_list.append(PostGet(id=i,
                                        text=post_original_df[post_df['post_id'] == i].text.iloc[0],
                                        topic=post_original_df[post_df['post_id'] == i].topic.iloc[0])
                                )

    return posts_recommend_list
