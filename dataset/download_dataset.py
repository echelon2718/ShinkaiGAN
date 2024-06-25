######### UNCOMMENT TO DOWNLOAD DATASET ##########
import opendatasets as od
import os
import shutil

def download_dataset():
    os.makedirs('./.kaggle', exist_ok=True)
    shutil.move('./<YOUR_KAGGLE_API_KEY_JSON>.json', './.kaggle/<YOUR_KAGGLE_API_KEY_JSON>.json')
    os.chmod('./.kaggle/<YOUR_KAGGLE_API_KEY_JSON>.json', 0o600)
    od.download("https://www.kaggle.com/datasets/kevinputrasantoso/makoto-shinkai-dataset-new") # Still in private due to copyright