import torch
import torch.nn as nn

inputs_single_channel = torch.tensor([     # 1 通道C
    [                       # 3 高度H
        [0, 1, 2],          # 3 宽度W
        [3, 4, 5],
        [6, 7, 8]
    ]
])


inputs_multi_channel = torch.tensor([     # 3 通道C
    [                       # 通道1, HW 3,3
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ],

    [                       # 通道2, HW 3,3
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ],

    [                       # 通道3, HW 3,3
        [11, 22, 33],
        [44, 55, 66],
        [77, 88, 99]
    ]
])


def pool_single_process(inputs):

    maxpool=nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    output=maxpool(inputs)
    print(f' shape: {output.shape}')   
    return output


def pool_multi_process(inputs):

    avgpool=nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    output=avgpool(inputs)
    print(f' shape: {output.shape}')   
    return output


if __name__ == '__main__':
    pool_single_process(inputs_single_channel)
    pool_multi_process(inputs_multi_channel)