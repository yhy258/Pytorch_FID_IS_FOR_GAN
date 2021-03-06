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


def generate_img(batch_size, G, dim, device):
    with torch.no_grad():
        z = torch.randn(batch_size, dim, device=torch.device(device))
        fake_img = G(z)
    return fake_img


def gen_n_images(n, G, batch_size, dim, img_size, device):
    images = torch.zeros((n, *img_size))
    
    for i in range(n // batch_size): # 메모리 줄이기.
        images[i*batch_size : (i+1)*batch_size] = generate_img(batch_size, G, dim, device)
        
    return images[:n]

def Generator_Imgs(args, G, dataset):

    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True
    )

    generated_imgs = gen_n_images(len(dataset), G, args.batch_size, args.z_dim, args.img_size, args.device)
    generated_dataset = CustomDataset(generated_imgs)

    generated_loader = DataLoader(
        dataset=generated_dataset, batch_size=args.batch_size, shuffle=True
    )

    return dataloader, generated_loader, len(dataset)
