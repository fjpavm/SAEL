import JsonUtils
import ResultEvaluationUtils
from multiprocessing import Process
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt

def createFigure(title='', xLabel='', yLabel=''):
    fig = plt.figure()
    #fig = matplotlib.figure.Figure
    fig.suptitle(title,fontsize=14, fontweight='bold')
    graph = fig.add_subplot(1,1,1)
    #graph = matplotlib.axes.Axes  
    graph.set_xlabel(xLabel, fontsize = 14)
    graph.set_ylabel(yLabel, fontsize=14)
    return fig

def deleteFigure(fig):
    plt.close(fig)

# Code adapted from https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
def autolabel(graph, bars, xpos='left'):
    """
    Attach a text label above each bar in *bars*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for bar in bars:
        height = bar.get_height()
        graph.annotate('{:10.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

def checkNoneMakeArrayWithDefaultValue(in_array, in_size, in_default = None):
    # expand None values into arrays with default value
    if in_array == None:
        return [in_default for i in range(in_size)]
    else:
        return in_array

def addMultiBarChart(fig, in_yValueArrays, in_yValueErrs = None, in_barGroupLabels = None, in_barLegends = None, in_barColours = None, in_barGroupLabelsRotation = 0, in_writeYValues = True):
    graph = fig.add_subplot(1,1,1)
    #graph = matplotlib.axes.Axes
    numValues = len(in_yValueArrays[0])
    xValues = numpy.arange(numValues)
    numBars = len(in_yValueArrays)
    barWidth = 1.0/(numBars+1)
    barStartOffset = 0.5 - barWidth
     
    in_yValueErrs = checkNoneMakeArrayWithDefaultValue(in_yValueErrs, numBars)
    #in_barGroupLabels = checkNoneMakeArrayWithDefaultValue(in_barGroupLabels, numBars)
    addLegends = in_barLegends != None
    in_barLegends = checkNoneMakeArrayWithDefaultValue(in_barLegends, numBars)
    in_barColours = checkNoneMakeArrayWithDefaultValue(in_barColours, numBars)

    bars = []
    for i in range(numBars):
        bar = graph.bar(xValues - barStartOffset + i*barWidth, in_yValueArrays[i], barWidth, yerr=in_yValueErrs[i], label=in_barLegends[i], color=in_barColours[i])
        bars.append(bar)
    
    if in_writeYValues:
        for bar in bars:
            autolabel(graph, bar)
    
    graph.set_xticks(xValues)
    if in_barGroupLabels != None:
        graph.set_xticklabels(in_barGroupLabels, rotation=in_barGroupLabelsRotation)
    if addLegends:
        graph.legend(bbox_to_anchor=(1.01, 1), loc=2)
        

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

def processResults(useAversive, useNoOverlapDataset, useNonComplete, useReducedPredictionInterval):

    #compose results dir nam
    resultsDir = 'Results/Out_'
    resultsDir += 'Overlaps' if not useNoOverlapDataset else 'NoOverlaps'
    resultsDir += '_Aversive' if useAversive else '_NoAversive'
    resultsDir += '_NonComplete' if useNonComplete else '_NoNonComplete'
    resultsDir += '_ReducedInterval' if useReducedPredictionInterval else '_NoReducedInterval'
    resultsDir += '/'
    
    print('summarysing dir ' +  resultsDir)
    
    resultFiles = [resultsDir + file for file in os.listdir(resultsDir) if file.endswith('.results')]
    summaryJson = {}
    for resultFile in resultFiles:
        resultsReportJson = JsonUtils.readJsonFromFile(resultFile)
        resultsReport = resultsReportJson['resultsReport']
        for predictorName in resultsReport:
            finalMeasures = resultsReport[predictorName]['finalMeasures']
            for measure in finalMeasures:
                if measure not in summaryJson:
                    summaryJson[measure] = {}
                if predictorName not in summaryJson[measure]:
                    summaryJson[measure][predictorName] = {}
                    summaryJson[measure][predictorName]['values'] = []
                summaryJson[measure][predictorName]['values'].append(finalMeasures[measure]['0'])
            
            timeMeasures = resultsReport[predictorName]['time']
            for timeMeasure in timeMeasures:
                timeMeasureName = 'time.' + timeMeasure
                if timeMeasureName not in summaryJson:
                    summaryJson[timeMeasureName] = {}
                if predictorName not in summaryJson[timeMeasureName]:
                    summaryJson[timeMeasureName][predictorName] = {}
                    summaryJson[timeMeasureName][predictorName]['values'] = []
                summaryJson[timeMeasureName][predictorName]['values'].append(timeMeasures[timeMeasure])
    
    for measure in summaryJson:
        measures = summaryJson[measure]
        for predictorName in measures:
            values = measures[predictorName]['values']
            measures[predictorName]['mean'] = numpy.mean(values)
            measures[predictorName]['std'] = numpy.std(values)
            
    extraInfo = {}
    extraInfo['useAversive'] = useAversive
    extraInfo['useNoOverlapDataset'] = useNoOverlapDataset
    extraInfo['useNonComplete'] = useNonComplete
    extraInfo['useReducedPredictionInterval'] = useReducedPredictionInterval
    
    reportSummaryJson = {}
    reportSummaryJson['summary'] = summaryJson
    reportSummaryJson['extraInfo'] = extraInfo

    JsonUtils.writeJsonToFile(resultsDir+'results.summary', reportSummaryJson)
    
    ### Create Multi Bar Grapths
    # Performance Measures graph
    figureTitle = ''
    figureTitle += 'Overlaps' if not useNoOverlapDataset else 'NoOverlaps'
    figureTitle += '_Aversive' if useAversive else '_NoAversive'
    figureTitle += '_NonComplete' if useNonComplete else '_NoNonComplete'
    figureTitle += '_ReducedInterval' if useReducedPredictionInterval else '_NoReducedInterval'
    fig = createFigure(title=figureTitle, xLabel='Performance Measures', yLabel='Performance Value')
    
    predictorNames = []
    firstMeasureJson = next(iter(summaryJson.values()))
    predictorNames = [predictorName for predictorName in firstMeasureJson]
    barGroupLabels = [measure for measure in summaryJson if not measure.startswith('time')]
    yValueArrays = [ [summaryJson[measure][predictorName]['mean'] for measure in barGroupLabels] for predictorName in predictorNames]
    yValueErrs = [ [summaryJson[measure][predictorName]['std'] for measure in barGroupLabels] for predictorName in predictorNames]
    addMultiBarChart(fig, yValueArrays, in_yValueErrs = yValueErrs, in_barGroupLabels = barGroupLabels, in_barLegends = predictorNames, in_barColours = None)
    
    figureFilename = resultsDir + 'Graphs/PerformanceMeasureGroups.png'
    JsonUtils.ensure_dir(figureFilename)
    fig.savefig(figureFilename, bbox_inches='tight')
    deleteFigure(fig)
    
    # Time Measures graph
    figureTitle = ''
    figureTitle += 'Overlaps' if not useNoOverlapDataset else 'NoOverlaps'
    figureTitle += '_Aversive' if useAversive else '_NoAversive'
    figureTitle += '_NonComplete' if useNonComplete else '_NoNonComplete'
    figureTitle += '_ReducedInterval' if useReducedPredictionInterval else '_NoReducedInterval'
    fig = createFigure(title=figureTitle, xLabel='Time Measures', yLabel='Time (s)')
    
    predictorNames = []
    firstMeasureJson = next(iter(summaryJson.values()))
    predictorNames = [predictorName for predictorName in firstMeasureJson]
    barGroupLabels = [measure for measure in summaryJson if measure.startswith('time') and measure != 'time.totalTrain']
    yValueArrays = [ [summaryJson[measure][predictorName]['mean'] for measure in barGroupLabels] for predictorName in predictorNames]
    yValueErrs = [ [summaryJson[measure][predictorName]['std'] for measure in barGroupLabels] for predictorName in predictorNames]
    
    addMultiBarChart(fig, yValueArrays, in_yValueErrs = yValueErrs, in_barGroupLabels = barGroupLabels, in_barLegends = predictorNames, in_barColours = None)
    
    figureFilename = resultsDir + 'Graphs/TimeGroups.png'
    JsonUtils.ensure_dir(figureFilename)
    fig.savefig(figureFilename, bbox_inches='tight')
    deleteFigure(fig)

def processSummaries():
    booleanChoices = [False, True]
    
    summaries = []
    for useAversive in booleanChoices:
        for useNoOverlapDataset in booleanChoices:
            for useNonComplete in booleanChoices:
                for useReducedPredictionInterval in booleanChoices:
                    resultsDir = 'Results/Out_'
                    resultsDir += 'Overlaps' if not useNoOverlapDataset else 'NoOverlaps'
                    resultsDir += '_Aversive' if useAversive else '_NoAversive'
                    resultsDir += '_NonComplete' if useNonComplete else '_NoNonComplete'
                    resultsDir += '_ReducedInterval' if useReducedPredictionInterval else '_NoReducedInterval'
                    resultsDir += '/'
                    summaryFiles = [resultsDir + file for file in os.listdir(resultsDir) if file.endswith('.summary')]
                    summaries.append(JsonUtils.readJsonFromFile(summaryFiles[0]))
    
    predictorNames = []
    firstSummaryJson = summaries[0]['summary']
    firstMeasureJson = next(iter(firstSummaryJson.values()))
    predictorNames = [predictorName for predictorName in firstMeasureJson]
    measureNames = list(firstSummaryJson)
    
    graphsDir = 'Results/Graphs/'
    
    for measure in measureNames:
         # Time Measures graph
        figureTitle = measure
        yLabel =  'Time (s)' if measure.startswith('time') else measure + ' Performance'
        fig = createFigure(title=figureTitle, xLabel='Test Parameters', yLabel=yLabel)
    
        def barGeoupLabelCreator(useNoOverlapDataset, useAversive, useNonComplete, useReducedPredictionInterval):
            label = ''
            label += 'O[y]' if not useNoOverlapDataset else 'O[n]'
            label += '_A[y]' if useAversive else '_A[n]'
            label += '_NC[y]' if useNonComplete else '_NC[n]'
            label += '_RI[y]' if useReducedPredictionInterval else '_RI[n]'
            return label
        
        barGroupLabels = [barGeoupLabelCreator(reportSummaryJson['extraInfo']['useNoOverlapDataset'],
                                               reportSummaryJson['extraInfo']['useAversive'],
                                               reportSummaryJson['extraInfo']['useNonComplete'],
                                               reportSummaryJson['extraInfo']['useReducedPredictionInterval']) for reportSummaryJson in summaries]
        yValueArrays = [ [reportSummaryJson['summary'][measure][predictorName]['mean'] for reportSummaryJson in summaries] for predictorName in predictorNames]
        yValueErrs = [ [reportSummaryJson['summary'][measure][predictorName]['std'] for reportSummaryJson in summaries] for predictorName in predictorNames]
    
        addMultiBarChart(fig, yValueArrays, in_yValueErrs = yValueErrs, in_barGroupLabels = barGroupLabels, in_barLegends = predictorNames, in_barColours = None, in_barGroupLabelsRotation = 'vertical', in_writeYValues = False)

        figureFilename = graphsDir + measure + '.png'
        JsonUtils.ensure_dir(figureFilename)
        fig.savefig(figureFilename, bbox_inches='tight')
        deleteFigure(fig)
 
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
    

    booleanChoices = [False, True]
    
    testProcesses = []
    for useAversive in booleanChoices:
        for useNoOverlapDataset in booleanChoices:
            for useNonComplete in booleanChoices:
                for useReducedPredictionInterval in booleanChoices:
                    #processResults(useAversive,useNoOverlapDataset,useNonComplete,useReducedPredictionInterval)
                    testProcesses.append(subProcess(processResults)(useAversive,useNoOverlapDataset,useNonComplete,useReducedPredictionInterval))
    
    runningProcesses = []
    maxConcurrent = 7
    for i,proc in enumerate(testProcesses):
        if i % maxConcurrent == 0:
            for runningProcess in runningProcesses:
                runningProcess.join()
            runningProcesses.clear()
        proc.start()
        runningProcesses.append(proc)
    
    for runningProcess in runningProcesses:
        runningProcess.join()
    
    processSummaries()
    
    print('FINISHED ')
         
