# tensorrt_custom_plugin
custom  plugin for tensorrt, include pRelu, leakyRelu, Slice...
for tensorrt 5.0.2.6, I think it available for tensorrt other version such as 4.0. but for your best performance, use the newest tensorrt version



## extras

1. add the inference sample, use the build guide to build your inference.
2. add the arcface caffe model, if you want to run the sample, download the [arcface](https://drive.google.com/drive/folders/1fshKBjLtnsjwnCbb4xkiTUKB8uLFFyvn?usp=sharing) model
3. base inference include 
   - uffmodel -> tensorrt model
   - caffemodel-> tensorrt model
   - onnx -> tensorrt model



## depends

1. opencv >= 3.0
2. cuda >= 8.0



## build

```
git clone https://github.com/ruiyoua/tensorrt_custom_plugin

mkdir build

cd build

cmake.. && make
```

## run

```
./bin/tensorrt-custom
```

