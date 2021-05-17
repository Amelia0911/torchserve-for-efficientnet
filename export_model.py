import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet #使用pip安装的EfficientNet包，进行模型转换

model = EfficientNet.from_name('efficientnet-b1') #初始化网络结构
print(model)
model.set_swish(memory_efficient=False)

#自定义类别
# in_features = model._fc.in_features #取出fc层
# model._fc = nn.Linear(in_features, 80)

if __name__ == '__main__':

    weights = 'efficientnet-b1.pth'
    image = torch.rand(1, 3, 240, 240)

    if 1:
        device = torch.device('cuda')
        image = image.to(device)
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weights).items()})
        model.load_state_dict(torch.load(weights))
        model.cuda()

    else:
        device = torch.device('cpu')
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weights, map_location=torch.device('cpu')).items()})  # 多卡训练的模型
        model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))  # 多卡训练的模型
        model.cpu()

    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, image)
        traced_model.save('efficientnet-b1.pt')

    print("Done !")
