# rkface_insight
这是一个在RV1106上使用insightface SDK进行人脸检测和识别的测试程序，部份代码参考了RKMPI、InsightFace C++代码。

## 编译
本工程默认使用VSCode + CMake + WSL方式进行交叉编译，你可以通过下列不同的方式进行编译。

1、WSL Ubuntu
将源码复制到WSL环境的Ubuntu系统下，在VSCode中通过WSL扩展打开源码目录，配置好交叉编译工具后直接Build .vscode下配置有直接将编译好的程序上传到板卡的任务，可以根据你的情况进行修改。

2、Ubuntu
将源码复制到Ubuntu下，确保交叉编译环境可用
```
cd rkface_insightface

mkdir build && cd build
cmake ..
make
```

## 程序的运行
将编译好的程序上传到板卡上执行./rkface_insight，并保证其目录结构如下：

```
rkface_insight
├── faces-----------------------人脸注册图片
|    ├── Zhang.png
|    └── Xia.png
└── model-----------------------模型文件 
    └── Pikachu
如果需要板子加电自动运行，需要修改板卡的应用启动脚本。
```