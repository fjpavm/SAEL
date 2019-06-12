class SituationalEmotionPredictor:
    def __init__(self, in_eventList, in_defaultPrediction):
        self.m_predictiveData = []
        self.m_eventList = in_eventList
        self.m_defaultPredictions = in_defaultPrediction

    def replacePredictiveData(self, in_predictiveData):
        self.m_predictiveData = in_predictiveData.copy()
        
    def addPredictiveData(self, in_newPredictiveData):
        self.m_predictiveData.extend(in_newPredictiveData)
        
    def train(self):
        raise NotImplementedError()
    
    def predict(self, in_situation):
        raise NotImplementedError()