from SensingEvent import SensingEvent
from Situation import Situation
import JsonUtils

def situationMapOrListToJson(in_situationMapOrList):
    if isinstance(in_situationMapOrList, dict):
        return {k:v.toJson() for k,v in in_situationMapOrList.items()}
    else:
        return [v.toJson() for v in in_situationMapOrList]
    
def situationMapOrListFromJson(in_situationMapOrListJson):
    if isinstance(in_situationMapOrListJson, dict):
        return {k:Situation.fromJson(v) for k,v in in_situationMapOrListJson.items()}
    else:
        return [Situation.fromJson(v) for v in in_situationMapOrListJson]

class SituationSegmenter:
    def __init__(self, situationDuration = 30*5, detectionInterval = 6*5, predictiveIntervalRatio = 2, predictFromNonCompletedSituations = True):
        self.m_situationDuration = situationDuration
        self.m_situationDetectionInterval = detectionInterval
        self.m_predictiveIntervalRatio = predictiveIntervalRatio
        self.m_predictiveInterval = predictiveIntervalRatio*situationDuration
        self.m_predictFromNonCompletedSituations = predictFromNonCompletedSituations
        self.m_events = []
        self.m_startedNeutralSituations = {}
        self.m_currentEmotionalSituations = {}
        self.m_completedNeutralSituations = {}
        self.m_predictingNeutralSituations = [] # used to store all completed situations with predictive anotations
        self.m_finishedEmotionalSituations = [] # used to store all emotional situations once finished
        
    def toJson(self):
        situationSegmenterJson = dict()
        # special map to avoid input info duplication for each event
        situationSegmenterJson['eventInputInfoMap'] = {id(sensingEvent.m_info):sensingEvent.m_info for sensingEvent in self.m_events}
        
        situationSegmenterJson['situationDuration'] = self.m_situationDuration
        situationSegmenterJson['situationDetectionInterval'] = self.m_situationDetectionInterval
        situationSegmenterJson['predictiveIntervalRatio'] = self.m_predictiveIntervalRatio
        situationSegmenterJson['predictiveInterval'] = self.m_predictiveInterval
        situationSegmenterJson['predictFromNonCompletedSituations'] = self.m_predictFromNonCompletedSituations
        situationSegmenterJson['events'] = [event.toJson() for event in self.m_events]
        situationSegmenterJson['startedNeutralSituations'] = situationMapOrListToJson(self.m_startedNeutralSituations)
        situationSegmenterJson['currentEmotionalSituations'] = situationMapOrListToJson(self.m_currentEmotionalSituations)
        situationSegmenterJson['completedNeutralSituations'] = situationMapOrListToJson(self.m_completedNeutralSituations)
        situationSegmenterJson['predictingNeutralSituations'] = situationMapOrListToJson(self.m_predictingNeutralSituations)
        situationSegmenterJson['finishedEmotionalSituations'] = situationMapOrListToJson(self.m_finishedEmotionalSituations)
        
        return situationSegmenterJson
    
    @classmethod
    def fromJson(cls, situationSegmenterJson):
        sitSeg = cls()
        eventInputInfoMap = JsonUtils.jsonKeys2int(situationSegmenterJson['eventInputInfoMap'])
        sitSeg.m_situationDuration = situationSegmenterJson['situationDuration']
        sitSeg.m_situationDetectionInterval = situationSegmenterJson['situationDetectionInterval']
        sitSeg.m_predictiveIntervalRatio = situationSegmenterJson['predictiveIntervalRatio']
        sitSeg.m_predictiveInterval = situationSegmenterJson['predictiveInterval']
        sitSeg.m_predictFromNonCompletedSituations = situationSegmenterJson['predictFromNonCompletedSituations']
        sitSeg.m_events = [SensingEvent.fromJson(eventInputInfoMap, eventJson) for eventJson in situationSegmenterJson['events'] ]
        sitSeg.m_startedNeutralSituations = JsonUtils.jsonKeys2int( situationMapOrListFromJson( situationSegmenterJson['startedNeutralSituations'] ) )
        sitSeg.m_currentEmotionalSituations = JsonUtils.jsonKeys2int( situationMapOrListFromJson( situationSegmenterJson['currentEmotionalSituations'] ) )
        sitSeg.m_completedNeutralSituations = JsonUtils.jsonKeys2int( situationMapOrListFromJson( situationSegmenterJson['completedNeutralSituations'] ) )
        sitSeg.m_predictingNeutralSituations = situationMapOrListFromJson( situationSegmenterJson['predictingNeutralSituations'] )
        sitSeg.m_finishedEmotionalSituations = situationMapOrListFromJson( situationSegmenterJson['finishedEmotionalSituations'] )

        return sitSeg
    
    
    def addNewEvent(self, in_event):
            currentEventNr = len(self.m_events)
            self.m_events.append(in_event)
            
            # update neutral situations 
            #  - finished started situations when duration reaches situationDuration
            moveToCompleted = [k for k in self.m_startedNeutralSituations if (currentEventNr - k) >= self.m_situationDuration]
            newCompletedSituations = [self.m_startedNeutralSituations[k] for k in moveToCompleted]
            for k in moveToCompleted: 
                situation = self.m_startedNeutralSituations.pop(k)
                situation.m_endEvent = situation.m_startEvent + self.m_situationDuration
                self.m_completedNeutralSituations[situation.m_endEvent] = situation
            #  - start new neutral situation if detectionInterval reached
            if currentEventNr % self.m_situationDetectionInterval == 0:
                situation = Situation(currentEventNr)
                self.m_startedNeutralSituations[currentEventNr] = situation
            #  - update completed neutral situations that fall outside the prediction interval
            #    (predictive tags are added when emotinal situations are running)
            moveToPredicting = [k for k in self.m_completedNeutralSituations if (currentEventNr - k) >= self.m_predictiveInterval]
            newPredictingSituations = [self.m_completedNeutralSituations[k] for k in moveToPredicting]
            for k in moveToPredicting: 
                situation = self.m_completedNeutralSituations.pop(k)
                self.m_predictingNeutralSituations.append(situation)
                
            # newCompletedSituations will need predicting 
            # newPredictingSituations are new training examples
            return (newCompletedSituations,  newPredictingSituations) 

                
    def registerActiveEmotions(self, in_emotionIDList):
        currentEventNr = len(self.m_events)
        # Start emotional situations when new IDs are active 
        for eID in in_emotionIDList:
            if self.m_currentEmotionalSituations.get(eID) == None:
                self.m_currentEmotionalSituations[eID] = Situation(currentEventNr, eID)
                # Add predicting tags to completed neutral situations (predicts start of emotional situations)
                for situation in self.m_completedNeutralSituations.values():
                    situation.m_predictiveMap[eID] = 1
                # Also add tag to non-completed situations if configured to do so
                if self.m_predictFromNonCompletedSituations:
                    for situation in self.m_startedNeutralSituations.values():
                        situation.m_predictiveMap[eID] = 1

        # Stop emotional situations who's IDs are no longer active
        moveToFinished = [k for k in self.m_currentEmotionalSituations if in_emotionIDList.count(k) == 0]
        for k in moveToFinished:
            situation = self.m_currentEmotionalSituations.pop(k)
            situation.m_endEvent = currentEventNr - 1
            self.m_finishedEmotionalSituations.append(situation)
                
            