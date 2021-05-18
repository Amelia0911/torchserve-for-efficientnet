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
首先对模型进行格式转换，代码：export_model.py。
```
torch-model-archiver --model-name efficientnet-b1 \
--version 1.0 \
--serialized-file efficientnet-b1.pt \
--extra-files ./index_to_name.json,./MyHandler.py \
--handler my_handler.py  \
--export-path model-store -f

参数说明：
torch-model-archiver --model-name efficientnet \     #打包的包名称，.mar
--version 1.0 \                                      #设定版本
--serialized-file efficientNet.pt \                  #导出的模型名称
--extra-files ./index_to_name.json,./MyHandler.py \  #附加文件：模型label；handler-整个推理过程(预处理+推理+后处理)
--handler my_handler.py  \                           #server-handler：推理流程
--export-path model-store -f                         #生成到制定文件夹-覆盖形式
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

输出：
{
  "tabby": 0.6616447567939758,
  "Egyptian_cat": 0.13378049433231354,
  "tiger_cat": 0.09632444381713867,
  "paper_towel": 0.006356021389365196,
  "Persian_cat": 0.005184500478208065
}
```

## 停止服务
```
torchserve --stop
```

