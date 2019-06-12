import JsonUtils

class Situation:
    def __init__(self, in_startEvent, in_emotionID = -1):
        self.m_startEvent = in_startEvent
        self.m_endEvent = -1
        self.m_predictiveMap = {}
        self.m_emotionID = in_emotionID
        
    def toJson(self):
        situationJson = dict()
        situationJson['startEvent'] = self.m_startEvent
        situationJson['endEvent'] = self.m_endEvent
        situationJson['emotionID'] = self.m_emotionID
        situationJson['predictiveMap'] = self.m_predictiveMap
        return situationJson
    
    @classmethod
    def fromJson(cls, in_situationJson):
        startEvent = in_situationJson['startEvent']
        self = cls(startEvent)
        self.m_endEvent = in_situationJson['endEvent']
        self.m_emotionID = in_situationJson['emotionID']
        self.m_predictiveMap = JsonUtils.jsonKeys2int(in_situationJson['predictiveMap'])
        return self
        
        
    def print(self):
        print('Situation start:' + str(self.m_startEvent) + ' end:' + str(self.m_endEvent) + ('' if not bool(self.m_predictiveMap) else ' predictions: ' + str(self.m_predictiveMap)) + ('' if self.m_emotionID == -1 else ' emotionID:' + str(self.m_emotionID)))