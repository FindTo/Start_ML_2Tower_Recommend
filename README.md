Karpov Courses StartML Project, Deep Learinig module.

https://karpov.courses/ml-start

---

# Introduction

It's the web service to get recommendations for users in social network.
Retrieving data from tables User, Posts and Feed, the list of top recommended posts per selected user can be obtained 
by http request.

Here the 2tower neural network model was implemented: features of user and item are being processed 
and used as inputs for user-tower NN and item-tower NN. Both of them generate 64D embeddings, dot product or cosine similarity 
between these vectors means score of potential interaction - is it going to be a like or not. Score can be converted
into probability of like using sigmoid function. The top of scores for user and posts - e.g. top of like 
probabilities - is being taken for recommendation. Such approach requires only single NN inference per query for the 
chosen user and single scalar multiplication between user embedding and vector of all post embeddings, with quicksort of 
the taken score list. That means much faster response time - about 10ms, with decent flexibility of feature processing:
MLP or transformer with interaction history per user may be used. 

# Input SQL tables structure

## User_data

- age - User age (in profile)
- city - User city (in profile)
- country - User country (in profile)
- exp_group - Experimental group: some encrypted category
- gender - User's gender
- user_id - Unique user ID
- os - Operating system of the device used to access the social network
- source - Whether the user came to the app from organic traffic or from advertising

##  Post_text_df 

- post_id - Unique post identifier
- text - Text content of the post
- topic - Main topic

##  Feed_data 

- timestamp - The time when the view was made.
- user_id - The ID of the user who made the view.
- post_id - The ID of the viewed post.
- action - The type of action: view or like.
- target - 1 for views if a like was made almost immediately after the view, otherwise 0. The value is omitted for 
like actions.

---

# Modules overview 

**Python ver. 3.12**

- There is a dockerfile and .dockerignore inside the project, which are aimed for server version with **Python 3.11** and 
**Torch 2.71+cpu**. Docker version is supposed for web service demonstration at 
https://startml2towerrecommend-production.up.railway.app/, where you can try to send query and receive a response.

- **Starting from v.2.0 NN model architechture is changed.** Now it combines static user features and limitied historical data for user embedding generation (UserTower). The ItemTower MLP is simplified. Feed_data table is being used for user histories generetion - in build_user_histories() function, which aggregates the table into a dictionary of interaction record per user, limited by max_history. The dictionary is being taken as an input for **transformer layer of UserTower** - to produce history based embedding and fuse it with the static one, based on user features. The folowing algorythm is the same. This approach demonstrate higher ROC-AUC and overall accuracy on condition of reach history data, with nearly all users mentioned witn non-zero history. Otherwise - there won't be a significant hitrate boost. In current conditions - 1.5M records dataset with random sampling is reach by unique users - 97% of all in user_data - but historical data length might not be sufficient.

The **main_script.py** contains all feature preparation functions and calls, you can launch it cnd check process. 
But .env file will be necessary for DB connections and output features file naming.
The basic pipelane: generate BERT-like embeddings for post texts using HuggingFace Roberta, then compress it to 128D. 
An MLP autoencoder is being used for this purpose. After that fetching process starts from the DB tables User_data,
Post_text_df and Feed_data, to prepare features vectors for 2Tower NN train. Afterward, the NN training process starts, 
producing model .pt file and history of epochs with ROC & accuracy at train/test. Also, there is an option to download
local user and posts from the corresponding NN towers outputs.

The **app.py** is used for web service operation. User_ID and timestamp are being received as an input. Then endpoint
function generates user tower embedding, normalizes if necessary and dots it with all item embeddings. The obtained list of scores helps to create the top of n posts. Activation using `uvicorn app:app --port 8000`, or with another port. 
Example of http query (GET method): 
http://startml2towerrecommend-production.up.railway.app/post/recommendations/?id=121245&time=2021-01-06 12:41:55

File **learn_model.py** contains classes and functions for 2Tower and MLP autoencoder training, including datasets 
preparations. That's all is used in **main_script.py**.

Module **get_model.py** is only aimed for .pt file downloading on remote or locally.

File **get_post_embeddings.py** contains functions for post embedding generation.

In **get_features_table.py** some functions for SQL upload/download can be found, for dataframes and CSV files. It's being used in web service at startup and after NN model learnig in main_script.py

Some modules - like **database.py**, **schema.py**, **table_feed.py**, **table_post.py** and **table_user.py** are used 
for setting SQLAlchemy ORM and Pydantic data formats.

# Endpoints

- **/user/{id}**: find relevant user info by id={id} and return as JSON
- **/post/{id}**: find relevant post info  by id={id} and return as JSON
- **/user/{id}/feed/?limit=LIMIT**: should return all actions from the feed for the user with id = {id}, sorted by 
actuality with limit=LIMIT
- **/post/{id}/feed/?limit=LIMIT**: should return all actions from the feed for the post with id = {id}, sorted by 
actuality with limit=LIMIT
- **/post/recommendations/?id=ID&time=TIMESTAMP&limit=LIMIT**: should return the top limit=LIMIT of recommended posts for the user with
user_id=ID at moment of time=TIMESTAMP (string datetime format %Y-%m-%d %H:%M:%S). There is the core endpoint for
recommendations.








 

