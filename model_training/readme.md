# model training (conda3-4.12.0 + CUDA11.8)

## anaconda环境配置

### 修改Anaconda创建虚拟环境的默认安装位置

- 修改.condarc配置文件

  1. 找到或创建用户目录下的.condarc文件（通常位于C:\\Users\用户名\\）

  ```
  envs_dirs:
  - D:\\Anaconda3\\envs
  ```

  2. （可选）修改包缓存位置

  ```
  pkgs_dirs:
  - D:\\Anaconda3\\pkgs
  ```

  3. 查看安装位置

  ```Shell
  conda env list
  ```

### 迁移现有环境

如果想将已有环境从C盘迁移到新位置：

直接复制环境文件夹(如从``C:\\Users\\用户名\\.conda\\envs\\env_name``到``D:\\Anaconda3\\envs\\env_name``)

确保新路径已包含在envs_dirs配置中

Conda会自动识别迁移后的环境

### 环境变量

- anaconda

``D:\anaconda\install``
``D:\anaconda\install\Library\mingw-w64\bin``
``D:\anaconda\install\Library\usr\bin``
``D:\anaconda\install\Library\bin``
``D:\anaconda\install\Scripts``

- CUDA

``D:\cuda\install\11.8\CUDA\bin``
``D:\cuda\install\11.8\CUDA\libnvvp``
``D:\cuda\development\extras\CUPTI\lib64``
``D:\cuda\development\cudnn\bin``

- CUDA 和 Cudnn 安装包
  通过网盘分享的文件：tensorflow环境安装包
  链接:
  ```
  https://pan.baidu.com/s/14jjpbvNik-pWI540pn8d9Q?pwd=cxci 提取码: cxci
  ```

### 虚拟环境搭建

1. 创建虚拟环境

```Shell
conda create -n tensorflow python=3.9
```

2. 安装tensorflow-gpu版本

```Shell
pip install tensorflow-gpu==2.10.0 -i https://mirrors.aliyun.com/pypi/simple
```

3. 安装其他依赖

```Shell
pip install -i https://mirrors.aliyun.com/pypi/simple pandas scikit-learn matplotlib numpy==1.21.5
```

4. 问题记录：

- 缺少windowsapi动态库(zlibwapi.dll)
  - 解决方法：
    https://blog.csdn.net/weixin_42166222/article/details/130625663?ops_request_misc=%257B%2522request%255Fid%2522%253A%25226c5b850d6b7c1c184709886c3c7081f1%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=6c5b850d6b7c1c184709886c3c7081f1&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-130625663-null-null.142^v101^control&utm_term=Could%20not%20locate%20zlibwapi.dll.%20Please%20make%20sure%20it%20is%20in%20your%20library%20path%21&spm=1018.2226.3001.4187

## pytorch环境配置

### 安装CUDA11.8支持的pytorch版本 (不要通过requirements.txt安装，否则为cpu版本)

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

- 如果在安装依赖时，提示需要t64.exe的问题，报错如下

```
ValueError: Unable to find resource t64.exe in package pip._vendor.distlib
```

- 解决方法：

  1. 删除损坏的 distlib 包

  ```Shell
  rm -rf D:\your env path\your env name\Lib\site-packages\pip\_vendor\distlib
  ```

  2. 重新下载distlib 包 (使用阿里源)

  ```Shell
  pip download distlib pip setuptools wheel -d . -i https://mirrors.aliyun.com/pypi/simple/
  ```

  解压后手动复制``t64.exe``到

  ```
  D:\your env path\your env name\Lib\site-packages\pip\_vendor\distlib
  ```
- 网络问题安装失败

  1. 离线安装包参考

  ```
  https://download.pytorch.org/whl/torch_stable.html
  ```

  ```
  https://pan.baidu.com/s/1KY-BjGf7pKIzWl46dXS4gw?pwd=dasb
  ```

  ```
  提取码: dasb
  ```

  2. 安装

  ```Shell
  python -m pip install xxx.whl
  ```

# Reasoning and deployment (tensorRT10.5 + CUDA12.6)
