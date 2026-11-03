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

# inference (ONNX runtime1.31 + Opencv4.13.0)

本文记录在 Windows x64 上安装 **ONNX Runtime C++ CPU 版**、配置 Visual Studio 2026 CMake 项目，以及排查常见错误的完整步骤。

本文示例使用：

- ONNX Runtime：`1.31.0`
- 安装类型：CPU、Windows x64
- 安装目录：`D:\onnxruntime`
- IDE：Visual Studio Community 2026（Visual Studio 18）
- 构建系统：CMake

> CPU 版不需要安装 CUDA 或 cuDNN。


## 1. 准备 Visual Studio 开发环境

在 Visual Studio Installer 中确认已经安装以下组件：

- 使用 C++ 的桌面开发
- MSVC x64/x86 C++ 生成工具
- Windows 10 或 Windows 11 SDK
- 用于 Windows 的 C++ CMake 工具

在 PowerShell 中可检查 CMake：

```powershell
cmake --version
cmake --help
```

Visual Studio 2026 对应的 CMake 生成器名称应为：

```text
Visual Studio 18 2026
```

## 2. 下载并安装 ONNX Runtime CPU x64

官方发布页：

- https://github.com/microsoft/onnxruntime/releases
- https://onnxruntime.ai/docs/get-started/with-cpp

下载 CPU x64 压缩包：

```text
onnxruntime-win-x64-1.31.0.zip
```

对应的官方下载地址：

```text
https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip
```


安装完成后，目录结构应类似：

```text
D:\onnxruntime\onnxruntime-win-x64-1.31.0
├─ include
│  ├─ onnxruntime_c_api.h
│  └─ onnxruntime_cxx_api.h
├─ lib
│  ├─ onnxruntime.dll
│  ├─ onnxruntime.lib
│  └─ onnxruntime_providers_shared.dll
├─ LICENSE
└─ VERSION_NUMBER
```

检查安装版本：

```powershell
Get-Content D:\onnxruntime\onnxruntime-win-x64-1.31.0\VERSION_NUMBER
```

## 3. 配置ORT用户环境变量

建议设置：

```text
ONNXRUNTIME_ROOT=D:\onnxruntime\onnxruntime-win-x64-10
```

并把以下目录加入当前用户的 `PATH`：

```text
D:\onnxruntime\onnxruntime-win-x64-1.31.0\lib
```

设置完成后，请重新启动 Visual Studio 和终端，使新环境变量生效。

检查结果：

```powershell
[Environment]::GetEnvironmentVariable("ONNXRUNTIME_ROOT", "User")
[Environment]::GetEnvironmentVariable("Path", "User")
```

## 4. 下载安装Opencv4.13.0

官方发布页
- https://github.com/opencv/opencv/releases#release-4.13.0


解压后得到：
```text

D:\opencv\4.13\opencv
│
├── build
│   ├── include
│   │   └── opencv2
│   └── x64
│       └── vc16
│           ├── bin
│           └── lib
└── sources
```

## 5. 配置Opencv用户环境变量

新建变量：
```text
OpenCV_DIR=D:\opencv\4.13\opencv\build
```

并把以下目录加入当前用户的 PATH
```text
D:\opencv\4.13\opencv\build\x64\vc16\bin
```

验证
```bash
echo %OpenCV_DIR%
where opencv_world4130.dll
```


## 6. 创建 Visual Studio CMake 项目

一个最小项目可包含：

```text
onnxruntime-cpp-smoke-test
├─ include
├─ src
├─ CmakeLists.txt
├─ out
└─ models
```


## 7. 常见错误处理

### 7.1 移动项目后 CMakeCache 路径不一致

典型错误：

```text
The current CMakeCache.txt directory is different than the directory
where CMakeCache.txt was created.
```

原因：`build` 中的 CMake 缓存记录了项目移动前的绝对路径。

在 Visual Studio 中处理：

1. 打开“项目”菜单。
2. 选择“删除缓存并重新配置”。
3. 等待 CMake 重新生成。

命令行处理：

```powershell
cmake --preset vs2026-x64 --fresh
cmake --build --preset release
```


### 7.3 找不到 onnxruntime_cxx_api.h

检查：

```powershell
Test-Path D:\onnxruntime\include\onnxruntime_cxx_api.h
```

如果返回 `False`，说明压缩包没有正确解压，或者 `ONNXRUNTIME_ROOT` 指向了错误目录。

### 7.4 链接时找不到 onnxruntime.lib

检查：

```powershell
Test-Path D:\onnxruntime\lib\onnxruntime.lib
```

同时确认 Visual Studio、项目和 ONNX Runtime 包都使用相同架构。本示例全部使用 x64。

### 7.5 运行时提示缺少 onnxruntime.dll

确认以下文件存在：

```text
D:\onnxruntime\lib\onnxruntime.dll
```

本示例的 `CMakeLists.txt` 会在构建后自动把 DLL 复制到可执行文件目录。也可以把 `D:\onnxruntime\lib` 加入用户 `PATH`。

## 8. CPU 版与 GPU 版的区别

本文安装的是 CPU x64 版本：

```text
onnxruntime-win-x64-1.31.0.zip
```

它不依赖 CUDA/cuDNN，适合先验证 C++ API 和模型推理流程。以后若切换到 CUDA 版本，需要额外匹配 ONNX Runtime、CUDA、cuDNN 以及显卡驱动版本，不能只替换一个 DLL。

## 9. 当前项目位置
ONNX Runtime 安装目录：

```text
D:\onnxruntime\onnxruntime-win-x64-1.31.0
```


# inference (tensorRT10.5 + CUDA12.6)
