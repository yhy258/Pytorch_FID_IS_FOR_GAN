from calculate_fid import FID
from inception_score import IS

class Config:
    z_dim = 128
    img_size = (3, 32, 32)
    batch_size = 50 #(len(dataset)//n)
    device = "cuda:0"
    dims = 2048
    inverse = False

args = Config()

"""
    Pretrained Generator를 불러와주세요!
    사용했던 Dataset을 불러와주세요! (Custom Data라면 torch.utils.data.Dataset을 상속받은 형태로)
    
    EX ) torchvision datasets API, CIFAR10
    dataset = datasets.CIFAR10(
        root='Datasets',
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]),
        download=True
    )
"""

dataset = None
G = None

if __name__ == "__main__":
    if dataset == None or G == None:
        pass
    else :
        fid_value = FID(args, G, dataset)
        mean, _ = IS(args, G, dataset)
        print("[FID : {:.4f}]".format(fid_value))
        print("[IS : {:.4f}]".format(mean))

