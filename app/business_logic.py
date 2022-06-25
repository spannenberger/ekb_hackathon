from app.utils import read_image_request, image_augmentations
from sklearn.neighbors import KNeighborsClassifier
from abc import ABC, abstractmethod
from tqdm import tqdm


class Handler(ABC):
    """Родительский хэндлер с основными методами, требующие реализации"""

    def process(self, input, **kwargs):
        """Запуск всех необходимых этапов хэндлера"""
        
        preprocessed_input = self.preprocess(input)
        model_output = self.forward_model(preprocessed_input)
        return self.business_logic(model_output, **kwargs)

    @abstractmethod
    def preprocess(self, input):
        """Обработки входных данных для дальнейшей обработки моделью"""
        
        pass

    @abstractmethod
    def forward_model(self, preprocessed_input):
        """Получаем предикты модели"""
        
        pass

    @abstractmethod
    def business_logic(self, model_output, **kwargs):
        """Логика хэндлера"""
        
        pass


class DetectionHandler(Handler):
    """Хэндлер детекции"""

    def __init__(
        self,
        detection_model,
        threshold
        ) -> None:
        
        """Инициализируем модель детекции"""
        
        self.detection_model = detection_model
        self.threshold = threshold

    def preprocess(self, input):
        """ Обработка полученного изображения

        Args:
            input: np.array - входное изображение

        Return:
            img: np.array - обработанное изображение
        """
        
        img = read_image_request(input)
        # добавить применение аугментаций из albumentations по конфигу, 
        # чтобы не менять код здесь в зависимости от примененных аугментаций на обучении
        img = image_augmentations(img)
        
        return img

    def forward_model(self, preprocessed_input):  
        """Предикт обработанного изображения моделью
        
        Args:
            preprocessed_input: np.array - обработанное изображение
            
        Return:
            predictions: List - предикт модели детекции
        """
        
        predictions = self.detection_model.get_model_prediction(preprocessed_input)
        return predictions

    def business_logic(self, model_output):
        """Основная логика хэндлера детекции

        Args:
            model_output: List - предикт модели детекции

        Return:
            all_bboxes: обработанные предикты модели по заданному threshold
        """
        all_bboxes = []
        model_output = model_output[0].squeeze()
        
        if model_output.shape[0] == 0:
            return all_bboxes

        for bbox in model_output:
            if bbox[-1] > self.threshold: 
                all_bboxes.append({
                    'bbox': {
                        'x1': int(bbox[0]), 
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]), 
                        'y2': int(bbox[3])
                        },
                    'probability': int(bbox[-1] * 100)
                    })
                
        return all_bboxes

class MetricLearningHandler(Handler):
    """Хэндлер метрик-лернинга"""

    def __init__(
        self,
        metric_model,
        classes_dict: dict,
        n_neighbors: int=1
        ) -> None:
        """Init metric learning model"""
        
        self.n_neighbors = n_neighbors
        self.classes_dict = classes_dict
        self.metric_model = metric_model

    def preprocess(self, input):
        """ По полученным bbox кропаем изображения 

        Args:
            input: Tuple[np.array, List] - Аугментированное изображение и предикты модели детекции

        Return:
            cutted_img: List[np.array] - список из всех кропнутых изображений
        """
        
        bboxes, img = input
        cutted_images = []
        
        for bbox in tqdm(bboxes):
            topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
            botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

            cutted_images.append(img[topLeftCorner[1]:botRightCorner[1], topLeftCorner[0]:botRightCorner[0]])
            
        return cutted_images

    def forward_model(self, preprocessed_input):
        """Предикт модели на всех кропнутых изображениях
        
        Args:
            preprocessed_input: List[np.array] - список из всех кропнутых изображений
            
        Return:
            predictions: List - список эмбеддингов модели metric learning
        """
        
        predictions = []
        
        for img in tqdm(preprocessed_input):
            predictions.append(self.metric_model.get_model_prediction(img))
            
        return predictions

    def business_logic(self, model_output):
        """Основная логика хэндлера metric learning

        Args:
            model_output: List - список эмбеддингов модели metric learning

        Return:
            metric_result: Dict - словарь ближайших изображений по расстояниям
        """
        
        metric_result = {
            'distances': [],
            'labels': []
        }
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='cosine')
        self.knn.fit(self.metric_model.base, self.metric_model.classes)
        # import pdb;pdb.set_trace()
        for metric_predict in tqdm(model_output):
            metric_predict = metric_predict.reshape(1, -1)
            distance, label = self.knn.kneighbors(metric_predict, n_neighbors=1, return_distance=True)
            metric_result['distances'].append(distance.squeeze().item())
            metric_result['labels'].append(label.squeeze().item())
            
        return metric_result