import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

# Parameters
USER_CAT_FEATURES = {
    'main_topic_liked': {'vocab_size': 7, 'embed_dim': 4},
    'main_topic_viewed': {'vocab_size': 7, 'embed_dim': 4},
    'country': {'vocab_size': 11, 'embed_dim': 6},
    'exp_group': {'vocab_size': 5, 'embed_dim': 4},
    'user_id': {'vocab_size': 168552, 'embed_dim': 64}
}

USER_NUM_FEATURES_CNT = 5

ITEM_CAT_FEATURES = {
    'post_id': {'vocab_size': 7319, 'embed_dim': 32},
    'topic': {'vocab_size': 7, 'embed_dim': 4}
}

ITEM_NUM_FEATURES_CNT = 131
TIME_FEATURES_CNT = 8

EMBEDDING_DIM = 64
BATCH_SIZE = 1024
N_EPOCHS = 7
LEARNING_RATE = 1e-3

# Download user's data from the SQL database
def get_user_df():

    user = pd.read_sql("SELECT * FROM public.user_data;", os.getenv('DATABASE_URL'))
    print(user.head())
    return user

# Download posts data from the SQL database
def get_post_df():

    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

# Download transformed BERT embeddings for posts (128D) from SQL database or from local
def get_embedd_df(is_csv=False, sep=';'):

    if is_csv:
        embedds = pd.read_csv('df_post_128d_embedd_with_id_pure.csv', sep=sep)
    else:
        # Loading post embedd
        embedds = pd.read_sql(f"SELECT * FROM {os.getenv('EMBEDD_DF_NAME')};", os.getenv('DATABASE_URL'))

    print(embedds.head())
    return embedds

def get_vector_df(post_embed, feed_n_lines=1512000, is_csv=False, sep=';'):

    if is_csv:

        user_features=pd.read_csv('user_df_encoded_for_2towers.csv', sep=sep)
        post_features = pd.read_csv('post_df_encoded_for_2towers.csv', sep=sep)
        feed_encoded = pd.read_csv('feed_df_encoded_for_2towers.csv', sep=sep)

        return user_features, post_features, feed_encoded
    else:

        # Download source dataframes from the DB
        user = get_user_df()
        post = get_post_df()

        feed = pd.read_sql(f"SELECT * FROM public.feed_data order by random() LIMIT {feed_n_lines};",
                           os.getenv('DATABASE_URL'))

        feed = feed.drop_duplicates()
        print(feed.head())

        # Working with User df
        # Set first categorial columns
        categorical_columns = []
        categorical_columns.append('country')
        categorical_columns.append('exp_group')

        # New boolean feature - pick only main cities in the countries
        capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                    'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
        user.city = user.city.apply(lambda x: 1 if x in capitals else 0)
        user = user.rename(columns={"city": "city_capital"})

        # User df is ready
        user.head()
        num_user_full = user['user_id'].nunique()
        print(f'Num of unique users:{num_user_full}')

        # Working with Post df
        # Add scaled text length feature
        post['text_length'] = np.log1p(post['text'].apply(len))

        # Delete source texts - it's been encoded into embeddings
        post = post.drop(['text'], axis=1)

        # Working with Feed df
        # Rename Action column
        feed = feed.rename(columns={"action": "action_class"})

        # Merge df's together into one master df
        df = pd.merge(
            feed,
            post,
            on='post_id',
            how='left'
        )

        df = pd.merge(
            df,
            user,
            on='user_id',
            how='left'
        )

        # Feature-counter for post likes by feed sample
        df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)
        df['post_likes'] = df.groupby('post_id')['action_class'].transform('sum')

        # Feature-counter for post views by feed sample
        df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
        df['post_views'] = df.groupby('post_id')['action_class'].transform('sum')
        df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

        # Parse Datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sorting by timestamp
        df = df.sort_values('timestamp')

        # Getting features from the timestamp
        df['day_of_week'] = df.timestamp.dt.dayofweek
        df['hour'] = df.timestamp.dt.hour
        df['month'] = df.timestamp.dt.month
        df['day'] = df.timestamp.dt.day
        df['year'] = df.timestamp.dt.year

        # Integral time indicator starting from th beginning of 2021 - min date of data
        df['time_indicator'] = np.log1p(
            (df['year'] - 2021) * 360 * 24 + df['month'] * 30 * 24 + df['day'] * 24 + df['hour'])

        # Sin-cos time encoding
        def encode_cyclic(df, col, period):
            sin = np.sin(2 * np.pi * df[col] / period)
            cos = np.cos(2 * np.pi * df[col] / period)
            return sin, cos

        # New sin-cos encoded features
        df['hour_sin'], df['hour_cos'] = encode_cyclic(df, 'hour', 24)
        df['weekday_sin'], df['weekday_cos'] = encode_cyclic(df, 'day_of_week', 7)
        df['month_sin'], df['month_cos'] = encode_cyclic(df, 'month', 12)
        df['is_weekend'] = df.day_of_week.apply(lambda x: 1 if x == 5 or x == 6 else 0)

        # Delete unnecessary features
        df = df.drop(['timestamp',
                      'day_of_week',
                      'hour',
                      'month',
                      'day',
                      'year'
                      ], axis=1)

        # Making new features: top liked anf viewed topics
        main_liked_topics = df[df['action_class'] == 1].groupby(['user_id'])['topic'].agg(
            lambda x: np.random.choice(x.mode())).to_frame().reset_index()
        main_liked_topics = main_liked_topics.rename(columns={"topic": "main_topic_liked"})
        main_viewed_topics = df[df['action_class'] == 0].groupby(['user_id'])['topic'].agg(
            lambda x: np.random.choice(x.mode())).to_frame().reset_index()
        main_viewed_topics = main_viewed_topics.rename(columns={"topic": "main_topic_viewed"})

        # Merge new features to the master DF
        df = pd.merge(df, main_liked_topics, on='user_id', how='left')
        df = pd.merge(df, main_viewed_topics, on='user_id', how='left')

        # Fill NaNs with mode
        df['main_topic_liked'].fillna(df['main_topic_liked'].mode().item(), inplace=True)
        df['main_topic_viewed'].fillna(df['main_topic_viewed'].mode().item(), inplace=True)

        # Add new features to the cat columns list
        categorical_columns.append('main_topic_viewed')
        categorical_columns.append('main_topic_liked')

        # Making new features: likes counters per user by data sample
        likes_per_user = df.groupby(['user_id'])['action_class'].agg(pd.Series.sum).to_frame().reset_index()
        likes_per_user = likes_per_user.rename(columns={"action_class": "likes_per_user"})

        # Making new features: views counters per user by data sample
        df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
        df['views_per_user'] = df.groupby('user_id')['action_class'].transform('sum')
        df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

        # Merge new features to the master DF
        df = pd.merge(df, likes_per_user, on='user_id', how='left')

        num_user_df = df['user_id'].nunique()
        print(f'Num of unique users in the dataset:{num_user_df}')

        num_post_df = df['post_id'].nunique()
        print(f'Num of unique posts in the dataset:{num_post_df}')

        # There are repeated lines in the dataset: one with action_class==1 and target==0,
        # another with target==1 and action class==1
        # Learning is going to be performed by target, so need to align them
        df['target'] = df['target'].astype('int32')
        df['action_class'] = df['action_class'].astype('int32')
        df['target'] = df['target'] | df['action_class']

        # Delete unnecessary features
        df = df.drop(['action_class',
                      'os',
                      'source'], axis=1)
        print('Categotial columns list:')
        print(categorical_columns)
        print('Master dataset - final:')
        print(df.head)

        # Making separated df's for the items, users and feed

        post_features = df[['post_id',
                            'topic',
                            'text_length',
                            'post_likes',
                            'post_views']]

        user_features = df[['user_id',
                            'main_topic_liked',
                            'main_topic_viewed',
                            'views_per_user',
                            'likes_per_user']]

        feed_encoded = df[['user_id',
                           'post_id',
                           'time_indicator',
                           'hour_sin',
                           'hour_cos',
                           'weekday_sin',
                           'weekday_cos',
                           'month_sin',
                           'month_cos',
                           'is_weekend',
                           'target']]

        # Download post's text embeddings dataset
        # Kaggle version - isn't used here
        # post_embed = pd.read_csv('/kaggle/input/df-post-roberta-128d-new/df_post_128d_embedd_with_id_pure (1).csv', sep=';')

        # Merge embeddings with post features df
        post_features = post_features.merge(post_embed, on='post_id', how='outer').drop_duplicates().reset_index(drop=True)

        # Fill NANs after merge with median values
        post_features.post_likes = post_features.post_likes.fillna(post_features.post_likes.median())
        post_features.post_views = post_features.post_views.fillna(post_features.post_views.median())
        post_features.drop(['text_length', 'topic'], axis=1, inplace=True)
        post_features = post_features.merge(post[['post_id', 'text_length', 'topic']], on='post_id', how='outer')

        user_features = user_features.merge(user[['user_id',
                                                  'gender',
                                                  'age',
                                                  'country',
                                                  'city_capital',
                                                  'exp_group']],
                                            on='user_id',
                                            how='outer').drop_duplicates().reset_index(drop=True)

        # Fill NaNs after merge for user data with median or mode
        user_features.views_per_user = user_features.views_per_user.fillna(user_features.views_per_user.median())
        user_features.likes_per_user = user_features.likes_per_user.fillna(user_features.likes_per_user.median())
        user_features.main_topic_viewed = user_features.main_topic_viewed.fillna(user_features.main_topic_viewed.mode()[0])
        user_features.main_topic_liked = user_features.main_topic_liked.fillna(user_features.main_topic_liked.mode()[0])

        # It's necessary to execute label encoding - for embedding creation for categorial features

        # List of topic categories
        topic_cats = df.topic.unique()
        topic_cats = [str(item) for item in topic_cats]
        topic_cats.append('new_cat')

        # List of country categories
        country_cats = df.country.unique()
        country_cats = [str(item) for item in country_cats]
        country_cats.append('new_cat')

        # Label encoders for strings
        le_topic, le_country = LabelEncoder(), LabelEncoder()
        le_topic.fit(topic_cats)
        le_country.fit(country_cats)

        # Encoding categorial features
        user_features.main_topic_liked = le_topic.transform(user_features.main_topic_liked)
        user_features.main_topic_viewed = le_topic.transform(user_features.main_topic_viewed)
        user_features.country = le_country.transform(user_features.country)
        post_features.topic = le_topic.transform(post_features.topic)

        # Print heads before returning df's
        print(user_features.head())
        print(post_features.head())
        print(feed_encoded.head())

        user_features.to_csv('user_df_encoded_for_2towers.csv', index=False, sep=';')
        post_features.to_csv('post_df_encoded_for_2towers.csv', index=False, sep=';')
        feed_encoded.to_csv('feed_df_encoded_for_2towers.csv', index=False, sep=';')

    return user_features, post_features, feed_encoded


# Autoencoder class - for reducing space of  BERT-like embeddings
class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, latent_dim)  # 128D latent space
        )
        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Class for post's data: separate data an ID
class Post_Data(Dataset):
    def __init__(self, data_source, is_csv=False, sep=';'):
        if is_csv:
            df = pd.read_csv(data_source, sep=sep)
        else:
            if not isinstance(data_source, pd.DataFrame):
                raise TypeError("If is_csv=False => data_source must be pd.DataFrame")
            df = data_source.copy()

        self.post_id = df['post_id']
        self.data = df.drop(['post_id'], axis=1)

    def __getitem__(self, idx):
        vector = self.data.loc[idx]
        post_id = self.post_id.loc[idx]

        vector = torch.FloatTensor(vector)
        post_id = torch.FloatTensor([post_id])

        return vector, post_id

    def __len__(self):
        return len(self.post_id)


def autoencoder_plot_stats(
        history: dict,
        title: str,
):
    plt.figure(figsize=(12, 4))

    # График Loss и MAE
    plt.subplot(1, 2, 1)
    plt.title(title + ' loss')
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.legend()

    # График процентилей MAE
    plt.subplot(1, 2, 2)
    plt.title(title + ' Percentiles: 25-50-75')
    plt.plot(history["train_mae_25p"], label="25th train")
    plt.plot(history["train_mae_median"], label="50th train")
    plt.plot(history["train_mae_75p"], label="75th train")
    plt.plot(history["test_mae_25p"], label="25th test")
    plt.plot(history["test_mae_median"], label="50th test")
    plt.plot(history["test_mae_75p"], label="75th test")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

# MAE accuracy for autoencoder training visualisation
def mae_accuracy(preds, x, is_test=False):
    mae = torch.abs(preds - x).mean(dim=1)  # MAE per sample
    mae_np = mae.cpu().detach().numpy()

    if not is_test:
        # return varios MAE metrics per batch
        return {
            "train_mae_mean": np.mean(mae_np),
            "train_mae_median": np.median(mae_np),
            "train_mae_25p": np.percentile(mae_np, 25),
            "train_mae_75p": np.percentile(mae_np, 75),
        }
    else:
        return {
            "test_mae_mean": np.mean(mae_np),
            "test_mae_median": np.median(mae_np),
            "test_mae_25p": np.percentile(mae_np, 25),
            "test_mae_75p": np.percentile(mae_np, 75),
        }

# Autoencoder inference mode - to get metrics
@torch.inference_mode()
def autoencoder_evaluate(model, loader, loss_fn, history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    epoch_loss = 0
    epoch_mae_metrics = {"test_mae_mean": [], "test_mae_median": [], "test_mae_25p": [], "test_mae_75p": []}

    for x, _ in tqdm(loader, desc='Test'):
        x = x.to(device)

        output = model(x)[0]

        loss = loss_fn(output, x)

        epoch_loss += loss.item()

        accuracy = mae_accuracy(output.detach().cpu(), x.detach().cpu(), is_test=True)

        for k in epoch_mae_metrics:
            epoch_mae_metrics[k].append(accuracy[k])

    history["test_loss"].append(epoch_loss / len(loader))

    for k in epoch_mae_metrics:
        history[k].append(np.mean(epoch_mae_metrics[k]))

    return history


# Train autoencoder using post's BERT-like embeddings - the whole cycle
def autoencoder_train(data_source, is_csv=False, sep=';', lr=1e-3, n_epoch=20):
    history = {
        "train_loss": [],
        "train_mae_mean": [],
        "train_mae_median": [],
        "train_mae_25p": [],
        "train_mae_75p": [],
        "test_loss": [],
        "test_mae_mean": [],
        "test_mae_median": [],
        "test_mae_25p": [],
        "test_mae_75p": [],

    }

    # Create dataset
    dataset = Post_Data(data_source,is_csv=is_csv, sep=sep)
    train_dataset, test_dataset = random_split(dataset,
                                               (int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8))
                                               )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    # Create the model
    model = Autoencoder(input_dim=768, latent_dim=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE loss
    model.train()

    for epoch in range(n_epoch):

        epoch_loss = 0
        epoch_mae_metrics = {"train_mae_mean": [],
                             "train_mae_median": [],
                             "train_mae_25p": [],
                             "train_mae_75p": []
                             }

        for x, _ in tqdm(train_loader, desc='Train'):
            x = x.to(device)

            optimizer.zero_grad()

            output = model(x)[0]

            loss = loss_fn(output, x)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            accuracy = mae_accuracy(output.detach().cpu(), x.detach().cpu())

            for k in epoch_mae_metrics:
                epoch_mae_metrics[k].append(accuracy[k])

        history["train_loss"].append(epoch_loss / len(train_loader))

        for k in epoch_mae_metrics:
            history[k].append(np.mean(epoch_mae_metrics[k]))

        autoencoder_evaluate(model, test_loader, loss_fn, history)

        clear_output()
        autoencoder_plot_stats(history, 'Posts autoencoder 768->128')

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Test Loss: {history['test_loss'][-1]:.4f} | "
            f"Train MAE: {history['train_mae_mean'][-1]:.4f} (median: {history['train_mae_median'][-1]:.4f}) | "
            f"Test MAE: {history['test_mae_mean'][-1]:.4f} (median: {history['test_mae_median'][-1]:.4f}) | "
            f"Train 25-75p: [{history['train_mae_25p'][-1]:.4f}, {history['train_mae_75p'][-1]:.4f}]"
            f"Test 25-75p: [{history['test_mae_25p'][-1]:.4f}, {history['test_mae_75p'][-1]:.4f}]"
        )

    torch.save(model.state_dict(), 'post_autoencoder_drop_0_3_0_2_pure.pt')

    return model, history

# Dataset for NN inference mode - get prepared post data
class ItemDataset(Dataset):
    def __init__(self,
                 item_features,
                 item_cat):
        self.item_features = item_features

        # Lists of categoral and mumerical column names
        self.item_cat_columns = item_cat
        self.item_num_columns = [x for x in self.item_features.columns.to_list() if
                                 x not in self.item_cat_columns.keys()]

    def __len__(self):
        return len(self.item_features)

    def __getitem__(self, idx):
        item_row = self.item_features.iloc[idx]
        item_id = item_row['post_id']

        # Retrieving post features
        item_cat = torch.tensor([item_row[f] for f in self.item_cat_columns], dtype=torch.long)
        item_num = torch.tensor([item_row[f] for f in self.item_num_columns], dtype=torch.float)

        return item_cat, item_num, item_id

# Dataset for NN learning based on the interaction history
class InteractionDataset(Dataset):
    def __init__(self,
                 interactions,
                 user_features,
                 item_features,
                 user_cat,
                 item_cat):
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features

        # Add an extra id column for fast access via index
        self.user_features['user_id_idx'] = self.user_features['user_id']
        self.user_features.set_index('user_id_idx', inplace=True)
        self.item_features['post_id_idx'] = self.item_features['post_id']
        self.item_features.set_index('post_id_idx', inplace=True)

        # Lists of categoral and mumerical column names
        self.user_cat_columns = user_cat
        self.user_num_columns = [x for x in self.user_features.columns.to_list() if
                                 x not in self.user_cat_columns.keys()]
        self.item_cat_columns = item_cat
        self.item_num_columns = [x for x in self.item_features.columns.to_list() if
                                 x not in self.item_cat_columns.keys()]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        user_id = interaction['user_id']
        post_id = interaction['post_id']
        target = interaction['target']

        # Retrieving user features
        user_row = self.user_features.loc[user_id]

        user_cat = torch.tensor([user_row[f] for f in self.user_cat_columns], dtype=torch.long)
        user_num = torch.tensor([user_row[f] for f in self.user_num_columns], dtype=torch.float)

        # Retrieving time features
        time_num = torch.tensor(interaction[['time_indicator',
                                                        'hour_sin',
                                                        'hour_cos',
                                                        'weekday_sin',
                                                        'weekday_cos',
                                                        'month_sin',
                                                        'month_cos',
                                                        'is_weekend'
                                                        ]].values, dtype=torch.float)

        # Retrieving post features
        item_row = self.item_features.loc[post_id]

        item_cat = torch.tensor([item_row[f] for f in self.item_cat_columns], dtype=torch.long)
        item_num = torch.tensor([item_row[f] for f in self.item_num_columns], dtype=torch.float)

        return user_cat, user_num, time_num, item_cat, item_num, torch.tensor(target, dtype=torch.float)


# User tower - NN for user embedding
class UserTower(nn.Module):
    def __init__(self):
        super().__init__()

        # Cat features - to embeddings with selected dim
        self.embeddings = nn.ModuleDict()
        self.dropout = nn.Dropout(0.1)

        for feature, params in USER_CAT_FEATURES.items():
            self.embeddings[feature] = nn.Embedding(
                params['vocab_size'] + 1,
                params['embed_dim']
            )

        # Fully connected layers
        total_embed_dim = sum(params['embed_dim'] for params in USER_CAT_FEATURES.values())

        self.fc1 = nn.Sequential(

            nn.Linear(total_embed_dim + USER_NUM_FEATURES_CNT + TIME_FEATURES_CNT, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.fc2 = nn.Sequential(

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Linear(96, EMBEDDING_DIM)

    def forward(self, cat_features, num_features, time_features):
        # Cat features - to embeddings
        embeddings = []
        for i, (feature, _) in enumerate(USER_CAT_FEATURES.items()):
            emb = self.dropout(self.embeddings[feature](cat_features[:, i]))
            embeddings.append(emb)

        # Concat all features tensors
        x = torch.cat(embeddings + [num_features] + [time_features], dim=1)

        x = self.fc1(x)
        residual = x  # bypass vector for skip connextion
        x = self.fc2(x)
        x = x + residual  # skip connection
        x = self.fc3(x)
        return self.output(x)


# Item tower - NN for post embedding
class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()

        # Cat features - to embeddings with selected dim
        self.embeddings = nn.ModuleDict()
        self.dropout = nn.Dropout(0.1)

        for feature, params in ITEM_CAT_FEATURES.items():
            self.embeddings[feature] = nn.Embedding(
                params['vocab_size'] + 1,
                params['embed_dim']
            )

        # Fully connected layers
        total_embed_dim = sum(params['embed_dim'] for params in ITEM_CAT_FEATURES.values())

        self.fc1 = nn.Sequential(

            nn.Linear(total_embed_dim + ITEM_NUM_FEATURES_CNT, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.fc2 = nn.Sequential(

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Linear(96, EMBEDDING_DIM)

    def forward(self, cat_features, num_features):

        # Cat features - to embeddings
        embeddings = []
        for i, (feature, _) in enumerate(ITEM_CAT_FEATURES.items()):
            emb = self.dropout(self.embeddings[feature](cat_features[:, i]))
            embeddings.append(emb)

        # Concat all features tensors
        x = torch.cat(embeddings + [num_features], dim=1)

        x = self.fc1(x)
        residual = x  # bypass vector for skip connection
        x = self.fc2(x)
        x = x + residual  # skip connection
        x = self.fc3(x)
        return self.output(x)


# Full TwoTower Model
class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_cat, user_num, time_num, item_cat, item_num):
        user_embedding = self.user_tower(user_cat, user_num, time_num)
        item_embedding = self.item_tower(item_cat, item_num)

        # Using cosine similarity - for FAISS search in the future
        sim = nn.CosineSimilarity(dim=1)(user_embedding, item_embedding)

        # Scaling for sigmoid
        logits = sim * nn.Parameter(torch.tensor(5.0))

        # # Scalar product for vectors - sigmoid is to be added externally
        # logits = torch.sum(user_embedding * item_embedding, dim=1)

        return logits


# Choose the best probability threshold for max accuracy
def find_best_threshold(y_true, y_pred_probs):
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_acc = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    print("Best threshold for accuracy:", best_thresh)
    return best_thresh

# Binary accuracy and ROC calculation
def _metrics(preds, y):
    y_true = np.concatenate(y)
    y_pred = torch.sigmoid(torch.from_numpy(np.concatenate(preds))).numpy()
    y_pred_label = (y_pred >= find_best_threshold(y_true, y_pred )).astype(int)
    acc = accuracy_score(y_true, y_pred_label)
    roc = roc_auc_score(y_true, y_pred)
    return acc, roc


# Train epoch of NN learning
def train(model, train_loader, optimizer, loss_fn, device) -> float:
    model.train()

    train_loss = 0
    # train_accuracy = 0
    y_true_list = []
    logits_list = []

    for batch in tqdm(train_loader, desc='Train', leave=False):
        batch = [x.to(device, non_blocking=True) for x in batch]
        u_c, u_n, t_n, i_c, i_n, y = batch

        optimizer.zero_grad()

        output = model(u_c, u_n, t_n, i_c, i_n)
        # output, y = output.cpu(), y.cpu()

        loss = loss_fn(output, y)
        train_loss += loss.item()

        y_true_list.append(y.detach().cpu().numpy())
        logits_list.append(output.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

    train_loss /= len(train_loader)
    train_accuracy, train_roc = _metrics(logits_list, y_true_list)

    return train_loss, train_accuracy, train_roc


# Evaluate NN accuracy anf ROC for test data
@torch.inference_mode()
def evaluate(model, test_loader, loss_fn, device):
    model.eval()

    test_loss = 0
    y_true_list = []
    logits_list = []

    for batch in tqdm(test_loader, desc='Evaluation', leave=False):
        batch = [x.to(device, non_blocking=True) for x in batch]
        u_c, u_n, t_n, i_c, i_n, y = batch

        output = model(u_c, u_n, t_n, i_c, i_n)

        y_true_list.append(y.detach().cpu().numpy())
        logits_list.append(output.detach().cpu().numpy())

        loss = loss_fn(output, y)

        test_loss += loss.item()

    test_loss /= len(test_loader)
    train_accuracy, train_roc = _metrics(logits_list, y_true_list)

    return test_loss, train_accuracy, train_roc


# Make epoch plots for ACC and ROC
def plot_history(hist):
    epochs = range(1, len(hist['train_roc']) + 1)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['train_roc'], label='Train ROC‑AUC')
    plt.plot(epochs, hist['test_roc'], label='Test  ROC‑AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC‑AUC')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['train_acc'], label='Train Acc')
    plt.plot(epochs, hist['test_acc'], label='Test  Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# Main co-rutine for NN training and validation with plots
def whole_train_valid_cycle(train_loader, test_loader, epochs=N_EPOCHS, lr=LEARNING_RATE,
                            device='cuda' if torch.cuda.is_available() else 'cpu', is_new_model=True):
    print(device)

    # Create NN model objects
    user_tower = UserTower()
    item_tower = ItemTower()
    model = TwoTowerModel(user_tower, item_tower)

    # If model is presented locally
    if not is_new_model:

        towers_state_dict = torch.load(
            '2Towers_cosine_4layer_drop_1_3_3_2_BCE_dec_1e4.pt',
            weights_only=True)
        model.load_state_dict(towers_state_dict)

    model.to(device)

    # Optimizer with regularization - weight_decay
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # For imbalanced classes - 20% of likes - set pos_weight
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {'train_roc': [], 'test_roc': [],
               'train_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_roc = train(model, train_loader, opt, loss_fn, device)
        te_loss, te_acc, te_roc = evaluate(model, test_loader, loss_fn, device)

        history['train_roc'].append(tr_roc)
        history['test_roc'].append(te_roc)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)

        clear_output()

        print(f'E{epoch:02d}: '
              f'train ROC {tr_roc:.4f}  ACC {tr_acc:.4f} | '
              f'test ROC {te_roc:.4f}  ACC {te_acc:.4f}')

        plot_history(history)

    torch.save(model.state_dict(), '2Towers_cosine_4layer_drop_1_3_3_2_BCE_dec_1e4.pt')

    return model, history




