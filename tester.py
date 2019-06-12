import JsonUtils
import ResultEvaluationUtils
from SituationSegmenter import SituationSegmenter
from SensingEvent import SensingEvent
from FrequencyPredictor import FrequencyPredictor
from TreePredictor import TreePredictor
import time
from multiprocessing import Process
        
def generateSensingEvent(timeIndex, sequenceInstanceJson, inputsForPredictionIndices, inputInfo):
        inputValues = [sequenceInstanceJson[str(inputIndex)][timeIndex] for inputIndex in inputsForPredictionIndices]
        return SensingEvent(inputInfo, inputValues, sequenceInstanceJson['situation_id'][timeIndex])
    
def generateEmotionIDs(timeIndex, sequenceInstanceJson, aversiveInputIndexs, aversiveThresholdValues):
        inputAversiveValues = [sequenceInstanceJson[str(inputIndex)][timeIndex] for inputIndex in aversiveInputIndexs]
        numAversiveIndexes = len(inputAversiveValues)
        emotionIDs = [inputIndex for inputIndex in range(numAversiveIndexes) if inputAversiveValues[inputIndex] < aversiveThresholdValues[inputIndex]]
        return emotionIDs

def comparePredictions(in_truth, in_prediction):
    if in_truth == -1:
        return 'ig'
    if in_prediction == 1:
        if in_prediction == in_truth:
            return 'tp'
        else:
            return 'fp'
    else:
        if in_prediction == in_truth:
            return 'tn'
        else:
            return 'fn'

def calculateCumulativePredictionTypes(in_numAvailableGroundTruth, in_predictionResults):

    allEmotionIDs = in_predictionResults[0][1].keys()
    truePositives = {k:[] for k in allEmotionIDs}
    falsePositives = {k:[] for k in allEmotionIDs}
    trueNegatives = {k:[] for k in allEmotionIDs}
    falseNegatives = {k:[] for k in allEmotionIDs}
    cumulativeResults = {k:dict({'tp':0,'fp':0,'tn':0,'fn':0,'ig':0}) for k in allEmotionIDs}
    for i in range(in_numAvailableGroundTruth):
        correctPrediction = {k:in_predictionResults[i][0].m_predictiveMap.get(k, 0) for k in allEmotionIDs} # assumes defaulting to 0
        actualPrediction = {k:in_predictionResults[i][1][k] for k in allEmotionIDs}
        comparison = {k: comparePredictions(correctPrediction[k], actualPrediction[k]) for k in allEmotionIDs}
        for k in allEmotionIDs:
            cumulativeResults[k][comparison[k]] += 1
        for k in allEmotionIDs:
            truePositives[k].append(cumulativeResults[k]['tp'])
            falsePositives[k].append(cumulativeResults[k]['fp'])
            trueNegatives[k].append(cumulativeResults[k]['tn'])
            falseNegatives[k].append(cumulativeResults[k]['fn'])
    return {'tp':truePositives, 'fp':falsePositives, 'tn':trueNegatives, 'fn':falseNegatives}


def recreateResultsAtIndexFromCumulativePredictionTypes(in_index, in_cumulativePredictionTypes):
    return {k: {'tp': in_cumulativePredictionTypes['tp'][k][in_index], 'fp': in_cumulativePredictionTypes['fp'][k][in_index], 'tn': in_cumulativePredictionTypes['tn'][k][in_index], 'fn': in_cumulativePredictionTypes['fn'][k][in_index]} for k in in_cumulativePredictionTypes['tp'].keys()}

def calculateCumulativeMeasure(in_measure, in_cumulativePredictionTypes, in_initialIgnoreRatio = 0.0, in_nonEvalValue = 0.0):
    allEmotionIDs = [k for k in in_cumulativePredictionTypes['tp'].keys()]
    totalGroundTruth = len(in_cumulativePredictionTypes['tp'][allEmotionIDs[0]])
    baseIndex = int(in_initialIgnoreRatio*totalGroundTruth)
    baseResult = recreateResultsAtIndexFromCumulativePredictionTypes(baseIndex, in_cumulativePredictionTypes)
    measureResults = {k:[in_nonEvalValue for i in range(baseIndex+1)] for k in allEmotionIDs}
    for k in measureResults:
        for i in range(baseIndex+1,totalGroundTruth):
            currentResult = recreateResultsAtIndexFromCumulativePredictionTypes(i, in_cumulativePredictionTypes)
            funcReturn =  ResultEvaluationUtils.resultFormatCall(in_measure, in_result=currentResult[k], in_baseResult=baseResult[k])
            measureResults[k].append(funcReturn)
    return measureResults

def calculateLastResultMeasure(in_measure, in_cumulativePredictionTypes, in_initialIgnoreRatio = 0.0):
    allEmotionIDs = [k for k in in_cumulativePredictionTypes['tp'].keys()]
    totalGroundTruth = len(in_cumulativePredictionTypes['tp'][allEmotionIDs[0]])
    baseIndex = int(in_initialIgnoreRatio*totalGroundTruth)
    baseResult = recreateResultsAtIndexFromCumulativePredictionTypes(baseIndex, in_cumulativePredictionTypes)
    finalResult = recreateResultsAtIndexFromCumulativePredictionTypes(-1, in_cumulativePredictionTypes)
    measureResults = {}
    for k in allEmotionIDs:
        funcReturn =  ResultEvaluationUtils.resultFormatCall(in_measure, in_result=finalResult[k], in_baseResult=baseResult[k])
        measureResults[k] = funcReturn
    return measureResults
    
def runSituationSequenceInstanceTest(filename, sitSeg, emotionalPredictors, inputsForPredictionIndices, inputsForPredictionInfo, aversiveInputIndexs, aversiveThresholdValues, extraInfo):
    situationSequenceInstance = JsonUtils.readJsonFromFile( filename)
    sequenceInstanceJson = situationSequenceInstance['sequenceInstance']
    instanceLength = len(sequenceInstanceJson['situation_id'])
    #instanceLength = int(instanceLength/10)
    print(filename + ' ' + JsonUtils.jsonToString(extraInfo, indent=None))
    print('situation sequence lenth is ' + str(instanceLength))
    emotionalPredictorsResults = { predictorName : [] for predictorName in emotionalPredictors}
        
    totalTrainingTime = { predictorName : 0.0 for predictorName in emotionalPredictors}
    maxTrainingTime = { predictorName : 0.0 for predictorName in emotionalPredictors}
    maxPredictionTime = { predictorName : 0.0 for predictorName in emotionalPredictors}
    fullst = time.time()
        
    for timeIndex in range(instanceLength):
        sensingEvent = generateSensingEvent(timeIndex, sequenceInstanceJson, inputsForPredictionIndices, inputsForPredictionInfo)
        # keep tract of emotional status from inputs (i.e. simple amigdala threshold tracking)
        emotionIDs = generateEmotionIDs(timeIndex, sequenceInstanceJson, aversiveInputIndexs, aversiveThresholdValues)
        sitSeg.registerActiveEmotions(emotionIDs)
        # add new events to segmenter
        newCompletedSituations, newPredictiveData = sitSeg.addNewEvent(sensingEvent)
        if bool(newPredictiveData):
            #print('training at time ' + str(timeIndex))
            for predictorName, predictor in emotionalPredictors.items():
                st = time.time()
                predictor.replacePredictiveData(sitSeg.m_predictingNeutralSituations)
                predictor.train()
                ft = time.time()
                runningTime = ft-st
                maxTrainingTime[predictorName] = max(runningTime, maxTrainingTime[predictorName])
                totalTrainingTime[predictorName] += runningTime
        if bool(newCompletedSituations):
            #print('predicting at time ' + str(timeIndex))
            for predictorName, predictor in emotionalPredictors.items():
                for situation in newCompletedSituations:
                    st = time.time()
                    prediction = predictor.predict(situation)
                    emotionalPredictorsResults[predictorName].append((situation, prediction))
                    ft = time.time()
                    runningTime = ft-st
                    maxPredictionTime[predictorName] = max(runningTime, maxPredictionTime[predictorName])
    fullft = time.time()
    runningTime = fullft-fullst
    print('total run time for sequence ' + str(runningTime))
    for predictorName in emotionalPredictors:
        print(predictorName + "trained for " + str(totalTrainingTime[predictorName]) + ", max training:" + str(maxTrainingTime[predictorName]) + " predicting:" + str(maxPredictionTime[predictorName]))
    
    compiledResults = {'results':emotionalPredictorsResults, 'time': {'totalTrain': totalTrainingTime, 'maxTrain': maxTrainingTime, 'maxPredict': maxPredictionTime} }
    return compiledResults

def createResultReport(emotionalPredictors, results, sitSeg):
    resultsReportJson = {}
    emotionalPredictorsResults = results['results']
    resultsReport = {}
    for predictorName in emotionalPredictors:
        predictorResultsReport = {}
        cumulativePredictionTypes = calculateCumulativePredictionTypes(len(sitSeg.m_predictingNeutralSituations), emotionalPredictorsResults[predictorName])
        predictorResultsReport['cumulativePredictionTypes'] = cumulativePredictionTypes
        predictorResultsReport['predictions'] = {k: [situationPredictionTuple[1][k] for situationPredictionTuple in emotionalPredictorsResults[predictorName]] for k in emotionalPredictorsResults[predictorName][0][1]}
        finalMeasures = {}
        finalMeasures['recall'] = calculateLastResultMeasure(ResultEvaluationUtils.recall, cumulativePredictionTypes, in_initialIgnoreRatio=0.2)
        finalMeasures['precision'] = calculateLastResultMeasure(ResultEvaluationUtils.precision, cumulativePredictionTypes, in_initialIgnoreRatio=0.2)
        finalMeasures['f1Score'] = calculateLastResultMeasure(ResultEvaluationUtils.f1Measure, cumulativePredictionTypes, in_initialIgnoreRatio=0.2)
        finalMeasures['f2Score'] = calculateLastResultMeasure(ResultEvaluationUtils.f2Measure, cumulativePredictionTypes, in_initialIgnoreRatio=0.2)
        predictorResultsReport['finalMeasures'] = finalMeasures
        predictorResultsReport['time'] = {'totalTrain': results['time']['totalTrain'][predictorName], 'maxTrain': results['time']['maxTrain'][predictorName], 'maxPredict': results['time']['maxPredict'][predictorName]}
        resultsReport[predictorName] = predictorResultsReport
    resultsReportJson['resultsReport'] = resultsReport
    resultsReportJson['situationSegmenter'] = sitSeg.toJson()
    return resultsReportJson

def runTest(useAversive, useNoOverlapDataset, useNonComplete, useReducedPredictionInterval):
    aversiveInputIndexs = [2]
    aversiveThresholdValues = [0.5]
    
    pathToSituationSequenceInstances = 'SitSeq_Overlaps/'
    if(useNoOverlapDataset):
        pathToSituationSequenceInstances = 'SitSeq_NoOverlaps/'
    
    inputsForPredictionIndices = [0,1,2,3] # All inputs
    if(not useAversive):
        inputsForPredictionIndices = [v for v in inputsForPredictionIndices if v not in aversiveInputIndexs]
    
        
    predictiveIntervalRatio = 2.0
    if(useReducedPredictionInterval):
        predictiveIntervalRatio =1.1
    #compose output dir nam
    outputDir = 'Out_'
    outputDir += 'Overlaps' if not useNoOverlapDataset else 'NoOverlaps'
    outputDir += '_Aversive' if useAversive else '_NoAversive'
    outputDir += '_NonComplete' if useNonComplete else '_NoNonComplete'
    outputDir += '_ReducedInterval' if useReducedPredictionInterval else '_NoReducedInterval'
    outputDir += '/'
    
    situationFileJson = JsonUtils.readJsonFromFile(pathToSituationSequenceInstances + "situations.json")
    inputsForPredictionInfo = [situationFileJson['inputs'][indx] for indx in inputsForPredictionIndices]
    
    for i in range(10):
        sitSeg = SituationSegmenter(predictiveIntervalRatio=predictiveIntervalRatio, predictFromNonCompletedSituations=useNonComplete) 
        emotionalPredictors = { 'frequency' : FrequencyPredictor(sitSeg.m_events, in_defaultPrediction={0:0}),
                                'treeOriginal' :  TreePredictor(sitSeg.m_events, in_defaultPrediction={0:0}, in_originalCompression=True),
                                'treeNew' :  TreePredictor(sitSeg.m_events, in_defaultPrediction={0:0}, in_originalCompression=False)
                              }
        sitSeqInstFilename = pathToSituationSequenceInstances+'sitSeq_'+str(i)+'.json'
        
        extraInfo = {}
        extraInfo['sequenceInstanceIndex'] = i
        extraInfo['useAversive'] = useAversive
        extraInfo['useNoOverlapDataset'] = useNoOverlapDataset
        extraInfo['useNonComplete'] = useNonComplete
        extraInfo['useReducedPredictionInterval'] = useReducedPredictionInterval
        
        results = runSituationSequenceInstanceTest(sitSeqInstFilename, sitSeg, emotionalPredictors, inputsForPredictionIndices, inputsForPredictionInfo, aversiveInputIndexs, aversiveThresholdValues, extraInfo)

        emotionalPredictors['treeOriginal'].exportPNG(outputDir+'treeOriginal_sitSeq_'+str(i)+'.png',0)
        emotionalPredictors['treeNew'].exportPNG(outputDir+'treeNew_sitSeq_'+str(i)+'.png',0)
        
        resultsReportJson = createResultReport(emotionalPredictors, results, sitSeg)

        resultsReportJson['extraInfo'] = extraInfo
        JsonUtils.writeJsonToFile(outputDir+'sitSeq_'+str(i)+'.results', resultsReportJson)
 
# subProcess function code adapted from https://gist.github.com/awesomebytes/0483e65e0884f05fb95e314c4f2b3db8
def subProcess(fn):
    """Can be uses as decorator to make a function call another process.
    Needs import
    from multiprocessing import Process"""
    def wrapper(*args, **kwargs):
        processObj = Process(target=fn, args=args, kwargs=kwargs)
        #thread.start()
        return processObj
    return wrapper
 
if __name__ == '__main__':
    
    useAversive = False
    useNoOverlapDataset = True
    useNonComplete = False
    useReducedPredictionInterval = True
    booleanChoices = [False, True]
    
    testProcesses = []
    for useAversive in booleanChoices:
        for useNoOverlapDataset in booleanChoices:
            for useNonComplete in booleanChoices:
                for useReducedPredictionInterval in booleanChoices:
                    testProcesses.append(subProcess(runTest)(useAversive,useNoOverlapDataset,useNonComplete,useReducedPredictionInterval))
    
    runningProcesses = []
    maxConcurrent = 3
    for i,proc in enumerate(testProcesses):
        if i % maxConcurrent == 0:
            for runningProcess in runningProcesses:
                runningProcess.join()
            runningProcesses.clear()
        proc.start()
        runningProcesses.append(proc)
    
            
         
