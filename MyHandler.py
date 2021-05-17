import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label

class MyHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_one_image(self, data):
        image = data.get("data")
        if image is None:
            image = data.get("body")
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(device=self.device)
        print("device: ", self.device)
        return image

    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def postprocess(self, images):
        outs = self.model.forward(images)
        probs = F.softmax(outs, dim=1)
        probs, classes = torch.topk(probs, 5, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        print("probs = ", probs)
        print("classes = ", classes)
        return map_class_to_label(probs, self.mapping, classes)