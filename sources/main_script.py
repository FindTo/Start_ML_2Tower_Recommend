from learn_model import (autoencoder_train,
                         Post_Data,
                         get_vector_df,
                         get_embedd_df,
                         build_user_histories,
                         InteractionDatasetWithHistory,
                         whole_train_valid_cycle,
                         USER_CAT_FEATURES,
                         ITEM_CAT_FEATURES
                         )

from get_post_embeddings import make_roberta_embeddings, get_128d_embeddings
from get_features_table import (df_to_sql, 
                                csv_to_sql, 
                                FEED_FEATURES_TYPE_MAP,
                                USER_FEATURES_TYPE_MAP,
                                POST_FEATURES_TYPE_MAP)
from dotenv import load_dotenv
import pandas as pd
import os

N_EPOCHS = 3
LEARNING_RATE = 5e-4
BATCH_SIZE=256
IS_NEW_MODEL=True

if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader, random_split

    # Load env variables
    load_dotenv()
    
    # Load RoBerta and prepare text embeddings 768d
    df_post_embed = make_roberta_embeddings()
    
    # Train autoencoder
    autoencoder_model, _ = autoencoder_train(df_post_embed)
    
    # Create dataset object form post embeddings dataframe
    post_dataset = Post_Data(df_post_embed)
    
    # Receive 128d embeddings
    df_post_128d = get_128d_embeddings(autoencoder_model, post_dataset)

    # df_post_128d = get_embedd_df(is_csv=True)

    # Fetch DB data for user, post and feed
    user_features, post_features, feed_encoded = get_vector_df(df_post_128d,
                                                               feed_n_lines=1512000,
                                                               is_csv=False)

    # Create dicts with user's histories of interactions - random max_history=50 interactions per user
    user_hist_dict = build_user_histories(feed_encoded)

    # Create dataset for learning with self-attention, with last lim_hist=8 interactions before the current one
    dataset = InteractionDatasetWithHistory(feed_encoded,
                                            user_features,
                                            post_features,
                                            USER_CAT_FEATURES,
                                            ITEM_CAT_FEATURES,
                                            user_histories=user_hist_dict,
                                            lim_hist = 8)


    # Set seed in order to freeze train/test separation between epoch sets
    generator = torch.Generator().manual_seed(123)

    # Separate to train/test
    train_dataset, test_dataset = random_split(dataset,
                                               (int(len(dataset) * 0.8),
                                                len(dataset) - int(len(dataset) * 0.8)),
                                              generator=generator)

    # Create loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              pin_memory=True,
                              shuffle=True,
                              persistent_workers=False,
                              num_workers=4
                             )
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             pin_memory=True,
                             shuffle=False,
                             persistent_workers=False,
                             num_workers=4)

    # Train 2Tower model and save it locally
    model, history = whole_train_valid_cycle(train_loader,
                                             test_loader,
                                             epochs=N_EPOCHS,
                                             lr=LEARNING_RATE,
                                             is_new_model=IS_NEW_MODEL)

    # Send features df's to the SQL database - for web service operation
    csv_to_sql('feed_df_encoded_for_2towers.csv',
          os.getenv('FEED_FEATURES_NN'),
          FEED_FEATURES_TYPE_MAP)

    csv_to_sql('user_df_encoded_for_2towers.csv',
                os.getenv('USER_FEATURES_NN'), 
                USER_FEATURES_TYPE_MAP, 
                )

    csv_to_sql('post_df_encoded_for_2towers.csv',
                os.getenv('POST_FEATURES_NN'), 
                POST_FEATURES_TYPE_MAP, 
                )


