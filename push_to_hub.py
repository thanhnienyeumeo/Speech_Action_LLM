repo_name = 'Colder203/Human_Robot_Interaction'

import os
from huggingface_hub import HfApi, HfFolder, Repository

checkpoint_path = 'speech_to_text_model_final'
api = HfApi()


list_dir = os.listdir(checkpoint_path)
for file in list_dir:
    #check if file is a directory
    if os.path.isdir(os.path.join(checkpoint_path, file)):
        for sub_file in os.listdir(os.path.join(checkpoint_path, file)):
            api.upload_file(
                path_or_fileobj= os.path.join(checkpoint_path, file, sub_file),
                path_in_repo=os.path.join(file, sub_file),
                repo_id = repo_name,
                repo_type='model'
            )
    else:
        api.upload_file(
            path_or_fileobj= os.path.join(checkpoint_path, file),
            path_in_repo=file,
            repo_id = repo_name,
            repo_type='model'
        )