import os
import torch
from dotenv import load_dotenv
from learn_model import UserTower, ItemTower, TwoTowerModel

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path

    return MODEL_PATH

def load_models(model_path):
    model_path = get_model_path(model_path)
    user_tower = UserTower()
    item_tower = ItemTower()
    model = TwoTowerModel(user_tower, item_tower)

    model.load_state_dict(torch.load(model_path,
                    map_location=torch.device('cpu')))
    model.eval()

    return model
