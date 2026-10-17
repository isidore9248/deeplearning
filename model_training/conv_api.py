import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def img_preprocess():
    img_rgb = plt.imread('./lenna.png')
    print(f'shape:{img_rgb.shape}')
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


    # PNG 图片可能包含透明度（alpha）通道；卷积层只需要 RGB 三通道。
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]


    # 把图像转换为张量格式  img -> 张量
    img_tensor=torch.tensor(img_rgb,dtype=torch.float)

    # 图像做处理时，需要把通道数放在第一个位置：C H W
    # 更改的维度形式为 原数组的 ( 第二个维度， 第零个维度， 第一个维度)
    img_tensor=img_tensor.permute(2,0,1) 

    # 因为这里只有1张图, 所以我们给它在增加1个维度, 从 CHW -> (1, C, H, W), 1张3通道的 H*W像素的 图
    img_tensor=img_tensor.unsqueeze(dim=0)

    print(f'shape:{img_tensor.shape}')  

    return img_tensor


def conv_process(img_tensor):

    conv1=nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=2,padding=0)

    # 卷积时会检查(1, C, H, W)的C是否符合要求
    conv_img=conv1(img_tensor)
    print(f'conv_img shape:{conv_img.shape}')  

    return conv_img


def conv_img_show(conv_img):
    conv_img_show=conv_img[0]  # C H W 
    # 转换通道顺序, 从 CHW -> HWC
    conv_img_show=conv_img_show.permute(1,2,0)  # H W C
    # 把张量转换为 图像数据格式
    conv_img_show=conv_img_show.detach().numpy() 

    for i in range(conv_img_show.shape[2]):
        plt.imshow(conv_img_show[:,:,i])
        plt.axis('off')
        plt.show()
    return 


if __name__=="__main__":
    img_tensor=img_preprocess()
    conv_img=conv_process(img_tensor)
    conv_img_show(conv_img)
