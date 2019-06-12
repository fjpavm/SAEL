from SituationalEmotionPredictor import SituationalEmotionPredictor
import random

class FrequencyPredictor(SituationalEmotionPredictor):
    def __init__(self, in_eventList, in_defaultPrediction):
        super().__init__(in_eventList, in_defaultPrediction)
        self.m_predictionOutputs = in_defaultPrediction.keys()
        self.m_predictionCounts = { k:{v:1} for k,v in in_defaultPrediction.items() }
        self.m_predictionTotals = { k:1 for k in self.m_predictionOutputs }
        
    def train(self):
        totalDataCount = len(self.m_predictiveData)
        self.m_predictionTotals = { k:0 for k in self.m_predictionOutputs}
        self.m_predictionCounts = { k:{} for k in self.m_predictionOutputs} # initialising counts to empty
        for data in self.m_predictiveData:
            for k in self.m_predictionOutputs:
                value = self.m_defaultPredictions[k] if not k in data.m_predictiveMap else data.m_predictiveMap[k]
                if value == -1: # using -1 as ignore tag
                    continue
                self.m_predictionCounts[k][value] = self.m_predictionCounts[k].get(value, 0) + 1
                self.m_predictionTotals[k] = self.m_predictionTotals[k]+1
    
    def predict(self, in_situation):
        returnPrediction = {}
        for k in self.m_predictionOutputs:
            randInt = random.randrange(0, self.m_predictionTotals[k])
            for value, count in self.m_predictionCounts[k].items():
                if randInt < count:
                    returnPrediction[k] = value
                    break
                else:
                    randInt -= count
        return returnPrediction