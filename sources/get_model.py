import os
import torch
from dotenv import load_dotenv
from learn_model import (UserTower, 
                        ItemTower, 
                        TwoTowerModel,
                        USER_CAT_FEATURES,
                        USER_NUM_FEATURES_CNT,
                        TIME_FEATURES_CNT,
                        HISTORY_LENGTH,
                        EMBEDDING_DIM)

# Load environment variables
load_dotenv()

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path

    return MODEL_PATH

def load_models(model_path, user_cat_features=USER_CAT_FEATURES,
                            user_num_features_cnt=USER_NUM_FEATURES_CNT,
                            time_features_cnt=TIME_FEATURES_CNT,
                            embedding_dim: int = EMBEDDING_DIM,
                            history_length: int = HISTORY_LENGTH):
                            
    model_path = get_model_path(model_path)
    item_tower = ItemTower()
    user_tower = UserTower(user_cat_features=user_cat_features,
                           user_num_features_cnt=user_num_features_cnt,
                           time_features_cnt=time_features_cnt,
                           item_tower=item_tower,
                           history_length=history_length,
                           embedding_dim=embedding_dim)                      
    model = TwoTowerModel(user_tower, item_tower)

    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device('cpu'),
                                     weights_only=False)
                          )
    model.eval()

    return model
