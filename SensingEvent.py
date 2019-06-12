class SensingEvent:
    def __init__(self, in_inputInfo, in_inputValues, in_originalSituationID = -1):
        self.m_info = in_inputInfo
        self.m_values = in_inputValues
        self.m_originalSituationID = in_originalSituationID
        
    def toJson(self):
        sensingEventJson = dict()
        sensingEventJson['info-ref'] = id(self.m_info)
        sensingEventJson['values'] = self.m_values
        sensingEventJson['origSituationID'] = self.m_originalSituationID
        return sensingEventJson
        
    # used to create object from json
    @classmethod
    def fromJson(cls, in_inputInfo, in_sensingEventJson):
        inputInfoRef = in_sensingEventJson['info-ref']
        if inputInfoRef in in_inputInfo:
            in_inputInfo = in_inputInfo[inputInfoRef]
        values = in_sensingEventJson['values']
        originalSituationID = in_sensingEventJson['origSituationID']
        return cls(in_inputInfo, values, originalSituationID)
    
    def print(self):
        print('SensingEvent values:' + str(self.m_values) + ('' if self.m_originalSituationID == -1 else ' original situation: ' + str(self.m_originalSituationID)))