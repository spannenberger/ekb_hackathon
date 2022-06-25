from app.business_logic import DetectionHandler, MetricLearningHandler

class TeethHackSolution:
    """Хэндлер для детекции моржей"""
    
    def __init__(self, detection_handler: DetectionHandler, metric_handler: MetricLearningHandler):
        
        self.detection_handler = detection_handler
        self.metric_handler = metric_handler
        
    def process(self, request, classes_dict):
        """Основная логика решения, с использованием необходимых хэндлеров
        
        Args:
            request: bytearray - байтовое представление изображения, полученное из тг бота
            classes_dict: Dict - словарь классов
            
        Return:
            img: np.array - изображение после всех обработок
            result: Dict - словарь с результатом работы сервиса 
        """
        
        img = self.detection_handler.preprocess(request)
        detection_result = self.detection_handler.forward_model(img)
        result = self.detection_handler.business_logic(detection_result)
        
        metric_input = (result, img)
        metric_result = self.metric_handler.process(metric_input)
        # import pdb;pdb.set_trace()
        for idx in range(len(result)):
            
            # if metric_result['distances'][idx] > 0.01:
            result[idx].update({'class_name': classes_dict[metric_result['labels'][idx]]})
                
            # else:
            #     result[idx].update({'class_name': classes_dict[1]})
        
        return result, img