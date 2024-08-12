from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models


class FeatureExtract:
    def __init__(self):
        self.device = self.get_device()
        self.model = self.load_model()
        self.preprocess = self.preprocess()

    @staticmethod
    def preprocess():
        return transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                   ])
    
    def get_device(self):
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
            
    def load_model(self):
        resnet = models.resnet50()
        resnet.fc = torch.nn.Linear(in_features=2048, out_features=238)
        resnet.to(self.device)
        loaded_state_dict = torch.load("forensicsUI/models/resnet50_state.pth", map_location=self.device)
        resnet.load_state_dict(loaded_state_dict)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        resnet.eval()
        return resnet

    def extract_vector(self, img_array):
        image = Image.fromarray(img_array).convert('RGB')
        image = self.preprocess(image)
        image = torch.unsqueeze(image, 0).to(self.device)  # Move image to MPS device
        with torch.no_grad():
            features = self.model(image)
        feature_vector = torch.flatten(features, start_dim=1)
        return feature_vector.cpu().numpy()
    
    