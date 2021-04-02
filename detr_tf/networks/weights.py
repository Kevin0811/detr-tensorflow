import os
import requests


WEIGHT_NAME_TO_CKPT = {
    "detr": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/checkpoint",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index"
    ]
}

def load_weights(model, weights: str):
    """ 
    Load weight on a given model
    weights are supposed to be sotred in the weight folder at the root of the repository. If weights
    does not exists, but are publicly known, the weight will be download from gcloud.
    """
    if not os.path.exists('weights'):
        # if folder is not exist add folder
        os.makedirs('weights')
    
    if "ckpt" in "weights":
        # load the model if it is already in the file
        model.load(weights)
    elif weights in WEIGHT_NAME_TO_CKPT:
        # otherwise down it from link

        # storage path
        wdir = f"weights/{weights}"

        if not os.path.exists(wdir):
            # if folder is not exist add folder
            os.makedirs(wdir)
        for f in WEIGHT_NAME_TO_CKPT[weights]:
            fname = f.split("/")[-1]
            if not os.path.exists(os.path.join(wdir, fname)):
                # start downloading
                print("Download....", f)
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(wdir, fname), 'wb').write(r.content)
        print("Load weights from", os.path.join(wdir, f"{weights}.ckpt"))
        
        # load model
        l = model.load_weights(os.path.join(wdir, f"{weights}.ckpt"))
        l.expect_partial()
    else:
        # print error
        raise Exception(f'Cant load the weights: {weights}')