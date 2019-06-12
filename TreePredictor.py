from SituationalEmotionPredictor import SituationalEmotionPredictor
from sklearn import tree
import numpy
from scipy import stats
from scipy import signal

import io 
import pydot
import JsonUtils

class TreePredictor(SituationalEmotionPredictor):
    def __init__(self, in_eventList, in_defaultPrediction, in_originalCompression = True):
        super().__init__(in_eventList, in_defaultPrediction)
        #min_sample_leaf_ratio = 1.0/150.0
        #self.m_predictionTrees = {k : tree.DecisionTreeClassifier(min_samples_leaf=min_sample_leaf_ratio) for k in in_defaultPrediction}
        self.m_predictionTrees = {k : tree.DecisionTreeClassifier() for k in in_defaultPrediction}
        self.m_trained = False
        if in_originalCompression:
            self.compressSituation = self.compressSituationOriginal
            self.createCompressionLabels = self.createCompressionOriginalLabels
        else:
            self.compressSituation = self.compressSituationNew
            self.createCompressionLabels = self.createCompressionNewLabels

    def compressSituationOriginal(self, in_situation):
        compressedSituation = []
        sitArray = [self.m_eventList[i].m_values for i in range(in_situation.m_startEvent, in_situation.m_endEvent)]
        sitMean = numpy.mean(sitArray, 0)
        compressedSituation.extend(sitMean)
        sitSkewness = stats.skew(sitArray,0,bias=True)
        # Deal with nan in skewness by assuming signal between 0 and 1 leading to -4 and +4 on extreme means
        sitSkewness = [sitSkewness[i] if not numpy.isnan(sitSkewness[i]) else 8*sitMean[i]-4 for i in range(len(sitSkewness))]
        compressedSituation.extend(sitSkewness)
        # Count number of peaks
        sitNumPeaks = [len(signal.find_peaks(input)[0]) for input in zip(*sitArray)]
        compressedSituation.extend(sitNumPeaks)
        return compressedSituation
    
    def createCompressionOriginalLabels(self):
        compressedSituationLabels = []
        meanLabels = [info['name'] + ' Mean' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(meanLabels)
        skewnessLabels = [info['name'] + ' Skewness' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(skewnessLabels)
        numPeaksLabels = [info['name'] + ' NumPeaks' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(numPeaksLabels)
        return compressedSituationLabels
    
    def compressSituationNew(self, in_situation):
        compressedSituation = []
        sitArray = [self.m_eventList[i].m_values for i in range(in_situation.m_startEvent, in_situation.m_endEvent)]
        sitMean = numpy.mean(sitArray, 0)
        # separate situation inputs into above and below mean to get top and bottom means
        # and the 'time' indexes for those above and bettow these means to get an average 'time index'
        topMean = []
        bottomMean = []
        topIndexMean = []
        bottomIndexMean = []
        for i,mean in enumerate(sitMean):
            tmArray = [ sitArray[k][i] for k in range(len(sitArray)) if  sitArray[k][i] > mean]
            tmArray.append(mean)
            tm = numpy.mean(tmArray)
            topMean.append(tm)
            
            tidxArray = [ k for k in range(len(sitArray)) if  sitArray[k][i] >= tm]
            topIndexMean.append(numpy.mean(tidxArray))
            
            bmArray = [ sitArray[k][i] for k in range(len(sitArray)) if  sitArray[k][i] < mean]
            bmArray.append(mean)
            bm = numpy.mean(bmArray)
            bottomMean.append(bm)
            
            bidxArray = [ k for k in range(len(sitArray)) if  sitArray[k][i] <= bm]
            bottomIndexMean.append(numpy.mean(bidxArray))
            
        compressedSituation.extend(topMean)
        compressedSituation.extend(topIndexMean)
        compressedSituation.extend(bottomMean)
        compressedSituation.extend(bottomIndexMean)

        # Count number of peaks
        #sitNumPeaks = [len(signal.find_peaks(input)[0]) for input in zip(*sitArray)]
        #compressedSituation.extend(sitNumPeaks)
        return compressedSituation
    
    def createCompressionNewLabels(self):
        compressedSituationLabels = []
        meanTopLabels = [info['name'] + ' Top Mean' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(meanTopLabels)
        indexTopLabels = [info['name'] + ' Top Index' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(indexTopLabels)
        meanBottomLabels = [info['name'] + ' Bottom Mean' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(meanBottomLabels)
        indexBottomLabels = [info['name'] + ' Bottom Index' for info in self.m_eventList[0].m_info]
        compressedSituationLabels.extend(indexBottomLabels)
        #numPeaksLabels = [info['name'] + ' NumPeaks' for info in self.m_eventList[0].m_info]
        #compressedSituationLabels.extend(numPeaksLabels)
        return compressedSituationLabels

        
    def train(self):
        # create list of compressed situations
        data = [self.compressSituation(situation) for situation in self.m_predictiveData]
        # make result lists for each emotion tag
        results = { k : [defaultPrediction if not k in situation.m_predictiveMap else situation.m_predictiveMap[k] for situation in self.m_predictiveData] for k , defaultPrediction in self.m_defaultPredictions.items()}
        # fit prediction trees to each result list
        for k in results:
            reducedData = [sit for i, sit in enumerate(data) if results[k][i] != -1] # using -1 as ignore tag
            reducedResults = [res for res in results[k] if res != -1] # using -1 as ignore tag
            self.m_predictionTrees[k].fit(reducedData, reducedResults)
        self.m_trained = True
    
    def predict(self, in_situation):
        returnPrediction = {}
        compressedSituation = self.compressSituation(in_situation)
        # change from list to expected numpy array format for single sample
        compressedSituation = numpy.array(compressedSituation).reshape(1, -1)
        for k, value in self.m_defaultPredictions.items():
            if self.m_trained:
                value = self.m_predictionTrees[k].predict(compressedSituation).tolist()[0] #.tolist()[0] turns it to standard python types for writing to json
            returnPrediction[k] = value
            
        return returnPrediction
    

    
    # Adapted from answer to https://stackoverflow.com/questions/27817994/visualizing-decision-tree-in-scikit-learn
    def exportPNG(self, filename, emotionID):
        
        JsonUtils.ensure_dir(filename)
        dotfile = io.StringIO()
        tree.export_graphviz(self.m_predictionTrees[emotionID], out_file=dotfile, feature_names=self.createCompressionLabels(), filled=True)
        pydot.graph_from_dot_data(dotfile.getvalue())[0].write_png(filename)
    
    
