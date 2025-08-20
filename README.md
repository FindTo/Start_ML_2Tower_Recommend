Karpov Courses StartML Project, Deep Learinig module.

https://karpov.courses/ml-start

---

# Introduction

It's the web service to get recommendations for users in social network.
Retrieving data from tables User, Posts and Feed, the list of top recommended posts per selected user can be obtained 
by http request.

---

# Input dataframe structure:

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

The **main_script.py** contains all feature preparation functions and calls, you can launch it cnd check process. 
But .env file will be necessary for DB connections and output features file naming.
The basic pipelane: generate BERT-like embeddings for post texts using HuggingFace Roberta, then compress it to 128D. 
An MLP autoencoder is being used for this purpose. After that fetching process starts from the DB tables User_data,
Post_text_df and Feed_data, to prepare features vectors for 2Tower NN train. Afterward, the NN training process starts, 
producing model .pt file and history of epochs with ROC & accuracy at train/test. Also, there is an option to download
local user and posts from the corresponding NN towers outputs.

The **app.py** is used for web service operation. User_ID and timestamp are being received as an input. Then endpoint
function generates user tower embedding, normalizes and dots it with all item embeddings. The obtained list of scores 
therefore helps to create a top of n posts. Activation using `uvicorn app:app --port 8000`, or with another port. 
Example of http query (GET method): 
http://startml2towerrecommend-production.up.railway.app/post/recommendations/?id=121245&time=2021-01-06 12:41:55

File **learn_model.py** contains classes and functions for 2Tower and MLP autoencoder training, including datasets 
preparations. That's all is used in **main_script.py**.

Module **get_model.py** is only aimed for .pt file downloading on remote or locally.

File **get_post_embeddings.py** contains functions for post embedding generation and df download/upload with remote DB.

Some modules - like **database.py**, **schema.py**, **table_feed.py**, **table_post.py** and **table_user.py** are used 
for setting SQLAlchemy ORM and Pydantic data formats.








 

