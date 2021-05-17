# torchserve-for-efficientnet


## 安装库
```
sudo apt install openjdk-11-jdk
conda create -n serve-g python=3.8
source activate serve-g

pip install sentencepiece   
pip install efficientnet_pytorch
pip install torchserve torch-model-archiver

git clone https://github.com/pytorch/serve.git
cd serve

1、gpu：
python ./ts_scripts/install_dependencies.py --cuda=cu110
2、cpu：
python ./ts_scripts/install_dependencies.py

```

## 创建.mar文件
```
torch-model-archiver --model-name efficientnet-b1 \
--version 1.0 \
--serialized-file efficientnet-b1.pt \
--extra-files ./index_to_name.json,./MyHandler.py \
--handler my_handler.py  \
--export-path model-store -f
```

## 启动模型服务
```
torchserve --start --ncs --model-store model-store --models efficientnet-b1.mar
```

## 查看模型列表
```
curl "http://localhost:8081/models" 
```

## 测试
```
curl http://127.0.0.1:8080/predictions/efficientnet-b1 -T cat.jpg
```

## 停止服务
```
torchserve --stop
```

