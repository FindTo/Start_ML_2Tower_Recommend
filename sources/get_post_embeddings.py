from dotenv import load_dotenv
import pandas as pd
import os
from sqlalchemy import create_engine
import torch
from transformers import RobertaModel, BertModel, DistilBertModel, DataCollatorWithPadding, AutoTokenizer
from learn_model import get_post_df, Autoencoder, Post_Data
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm

# Load DF with features from DB using chunks
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

        raise RuntimeError(f"Loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

# Send DF to the DB  - no chunks
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

# Choose NN model for text encoding
def get_model(model_name):
    assert model_name in ['bert', 'roberta', 'distilbert']

    checkpoint_names = {
        'bert': 'bert-base-cased',  # https://huggingface.co/bert-base-cased
        'roberta': 'roberta-base',  # https://huggingface.co/roberta-base
        'distilbert': 'distilbert-base-cased'  # https://huggingface.co/distilbert-base-cased
    }

    model_classes = {
        'bert': BertModel,
        'roberta': RobertaModel,
        'distilbert': DistilBertModel
    }

    return AutoTokenizer.from_pretrained(checkpoint_names[model_name]), model_classes[model_name].from_pretrained(
        checkpoint_names[model_name])

# Inference mode for HuggingFace BERT-like encoder
@torch.inference_mode()
def get_embeddings_labels(model, loader, device='cpu'):
    model.eval()

    total_embeddings = []

    for batch in tqdm(loader):
        # labels.append(batch['labels'].unsqueeze(1))

        batch = {key: batch[key].to(device) for key in ['attention_mask', 'input_ids']}

        embeddings = model(**batch)['last_hidden_state'][:, 0, :]

        total_embeddings.append(embeddings.cpu())

    return torch.cat(total_embeddings, dim=0)

# Generate BERT-like embeddings 768d
def make_roberta_embeddings():

    df_post = get_post_df()

    tokenizer, model = get_model('roberta')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    dataset = Dataset.from_pandas(df_post['text'].to_frame())

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenization(example):
        return tokenizer.batch_encode_plus(example['text'],
                                           add_special_tokens=True,
                                           return_token_type_ids=False,
                                           truncation=True,
                                           padding='max_length'
                                           )

    token_dataset = dataset.map(tokenization, batched=True)

    token_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(token_dataset, batch_size=64, collate_fn=data_collator, pin_memory=True, shuffle=False)

    embeddings = get_embeddings_labels(model, loader, device)

    embedd_df = pd.DataFrame(embeddings)

    df_post_embed = pd.concat([df_post, embedd_df], axis=1)
    df_post_embed.drop(['text', 'topic'], axis=1, inplace=True)
    df_post_embed.to_csv('df_post_roberta_pure.csv', sep=';', index=False)

    return df_post_embed

def get_128d_embeddings(model:Autoencoder, dataset:Post_Data):
    model.eval()

    device = next(model.parameters()).device
    post_128d = model(torch.FloatTensor(dataset.data.values).to(device))[1]
    df_post_128d = pd.DataFrame(post_128d.detach().to('cpu')).astype("float").add_prefix('embed_', axis=1)
    df_post_128d['post_id'] = dataset.post_id
    df_post_128d.to_csv('df_post_128d_embedd_with_id_pure.csv', sep=';', index=False)

    return df_post_128d











