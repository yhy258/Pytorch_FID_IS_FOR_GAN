import os

import torch
from torch.utils.data import DataLoader

"""
    주어진 Generator에 대해서 Data들을 샘플링하고, 이를 Dataset화 시켜줌.
"""
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, generated_imgs):
        self.datas = generated_imgs

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]
    
"""
    메모리가 부족하다면..
"""
# class CustomDataset(torch.utils.data.Dataset): 
#     def __init__(self, n, G, batch_size, dim):
#         self.G = G
#         self.batch_size = batch_size
#         self.n = n
#     def __len__(self):
#         return self.n
#     def __getitem__(self, idx): 
#         z = torch.randn(self.batch_size, dim, device=torch.device('cuda:0'))
#         datas = G(z)
#         return datas[idx]
    
    

"""
    CheckPoint 불러오기
"""


def load_checkpoint(model, model_dir, name):
    path = os.path.join(model_dir, name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=name, path=path
    ))
    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    iteration = checkpoint['iteration']
    return iteration


def generate_img(batch_size, G, dim):
    with torch.no_grad():
        z = torch.randn(batch_size, dim, device=torch.device('cuda:0'))
        fake_img = G(z)
    return fake_img


def gen_n_images(n, G, batch_size, dim):
    images = []
    for i in range(n // batch_size + 1):
        images.append(generate_img(batch_size, G, dim))
    images = torch.cat(images, dim=0)
    return images[:n]

def Generator_Imgs(args, G, dataset):

    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True
    )

    generated_imgs = gen_n_images(len(dataset), G, args.batch_size, args.dims)
    generated_dataset = CustomDataset(generated_imgs)

    generated_loader = DataLoader(
        dataset=generated_dataset, batch_size=args.batch_size, shuffle=True
    )

    return dataloader, generated_loader, len(dataset)
