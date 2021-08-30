import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from data_modules import *

def inception_score(dataloader, N, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    
    if len(next(iter(dataloader))) >= 2:
        for i, batch in enumerate(dataloader, 0):
            batch = batch[0].type(dtype)
            batch_size_i = batch.size()[0]

            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)
    else :
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batch_size_i = batch.size()[0]

            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def IS(args, G, dataset, resize=True, splits=1):
    _, generated_loader, num = Generator_Imgs(args, G, dataset)
    return inception_score(generated_loader, num, cuda=args.device, batch_size=args.batch_size, resize=resize, splits=splits)
