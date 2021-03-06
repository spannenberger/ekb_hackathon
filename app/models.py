from mmdet.apis import init_detector, inference_detector
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.preprocessing import LabelEncoder
from mmdet.apis import inference_detector
from mmcv import Config
import pandas as pd
import torch

class MmDetectionModel:
    """Реализация модели MMDet"""
    
    def __init__(self, model_path: str, config_path: str):
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        config = Config.fromfile(config_path)
        
        self.detection_model = init_detector(
            config, 
            checkpoint=model_path, 
            device=device
            )
    
    def get_model_prediction(self, img):
        result = inference_detector(self.detection_model, img)

        return result
    
    
class ViTMetricModel:
    """Реализация Visual Transformer"""
    
    def __init__(self, extractor, model_path: str, csv_path: str):
        
        extractor = 'google/vit-base-patch16-384' # фичи экстрактор
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # пока пусть так будет, нужно сделать не так явно
        df = pd.read_csv(csv_path) # считываем csv со средними эмбеддингами для каждого класса
        label_encoder = LabelEncoder()
        self.classes = label_encoder.fit_transform(df.columns)
        self.base = df.values.T

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(extractor)
        self.metric_model = ViTModel.from_pretrained(model_path)
        self.metric_model.to(self.device)

    def get_model_prediction(self, img):
        img = self.feature_extractor(img, return_tensors="pt")
        img.to(self.device)
        # инференс модели и получение предикта
        self.metric_model.eval()
        with torch.no_grad():
            prediction = self.metric_model(**img).pooler_output
            prediction = prediction[0].cpu().detach().numpy()
        
        return prediction
