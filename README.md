# FID_IS_FOR_GAN

It was difficult for me to set the model and dataset in using FID and IS codes.  
Personally, it was much more intuitive and easier to use to put pre-trained models and datasets as direct codes.  
  
GANs vary in architecture configuration depending on which dataset and which model they use.  
I think it would be more convenient for individuals to write directly on main.py.  


개인적으로 서치하면 나오는 FID 및 IS 코드를 사용하는데 있어서, model을 설정하는 자유도, 그리고 dataset을 설정하는데 이해하기 좀 어려웠었다. (느끼기에)  
그래서 나중에 내가 쓰기도 하고 공유도 할겸 올리게 되었다.  
직접 코드로서 pretrained model과 dataset을 넣을 수 있게 하는게 개인적으로는 훨씬 직관적이고 사용하기 쉬웠다.  
  
GAN은 어떤 dataset과 어떤 모델을 사용하는지에 따라 Architecture 구성이 천차만별이다.  
개개인이 main.py에 직접 써넣는게 더 편할것 같아 이렇게 구성. (무엇보다도 내가 제일 편했음.)
  
# How To Use??
  
main.py의 파일을 보면 pretrained Generator와 내가 직접 사용했던 dataset을 적을 수 있도록 각 변수에 None으로 줬다.  
거기에 pretrained Generator와 dataset을 넣고 실행하면 된다.  
pretrainted Generator를 넣으라는 말은 Generator를 불러오고 torch.load로 pretrained model의 parameters를 불러오라는 것.  
  
Just Input Model and Dataset! (in main.py)  
G : pretrained generator  
dataset : dataset

# For Pytorch
