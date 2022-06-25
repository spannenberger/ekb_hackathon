from app.models import MmDetectionModel, ViTMetricModel
from app.business_logic import DetectionHandler, MetricLearningHandler
from app.ekb_solution import TeethHackSolution
from sqlalchemy.orm import sessionmaker
from app.db import engine
import numpy as np
import random
import torch
import os

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
    
def init_models():
    """Initialize models and handlers"""
    
    # Возможно стоит подумать над созданием конфига, пока не уверен
    detection_model_path = os.getenv('DETECTION_MODEL', '')
    detection_config_path = os.getenv('DETECTION_CONFIG', '')
    extractor = os.getenv('METRIC_EXTRACTOR', '')
    metric_model_path = os.getenv('METRIC_MODEL', '')
    csv_path = os.getenv('METRIC_CSV_PATH', '')

    classes_dict = {0: 'teeth', 1: 'caries'}

    detection_model = MmDetectionModel(detection_model_path, detection_config_path) # we can change detection class here
    metric_model = ViTMetricModel(extractor, metric_model_path, csv_path) # we can change metric learning class here

    detection_handler = DetectionHandler(detection_model, threshold=0.5)
    metric_handler = MetricLearningHandler(metric_model, classes_dict, n_neighbors=1)
    habarovsk_handler = TeethHackSolution(detection_handler, metric_handler)
    
    return habarovsk_handler, classes_dict

def init_db():
    # TODO добавить обработку сигнала завершения с close коннекшена
    Session = sessionmaker(engine)
    conn = engine.connect()
    session = Session(bind=conn)
    
    return session


