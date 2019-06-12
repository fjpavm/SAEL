# Based on definitions from paper:
# "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation"
# Journal of Machine Learning Technologies, Vol 2 Issue 1, 2011, pages 37-63

# true positive rate a.k.a. sensitivity
def recall(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    realPositives = float(in_truePositives+in_falseNegatives)
    return 1.0 if realPositives == 0 else float(in_truePositives/realPositives)

# true negative rate a.k.a. specificity
def invRecall(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    realNegatives = float(in_trueNegatives+in_falsePositives)
    return 1.0 if realNegatives == 0 else float(in_trueNegatives/realNegatives)

# false positive rate
def fallout(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    realPositives = float(in_truePositives+in_falseNegatives)
    return 0.0 if realPositives == 0 else float(in_falsePositives/realPositives)

# false negative rate
def missRate(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    realNegatives = float(in_trueNegatives+in_falsePositives)
    return 0.0 if realNegatives == 0 else float(in_falseNegatives/realNegatives)

# true positive accuracy a.k.a. confidence
def precision(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    predictedPositives = float(in_truePositives+in_falsePositives)
    return 1.0 if predictedPositives == 0 else float(in_truePositives/predictedPositives)
# true negative accuracy a.k.a.
def invPrecision(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    predictedNegatives = float(in_trueNegatives+in_falseNegatives)
    return 1.0 if predictedNegatives == 0 else float(in_trueNegatives/predictedNegatives)

def accuracy(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    totalCount = float(in_truePositives+in_falsePositives+in_trueNegatives+in_falseNegatives)
    return float(in_truePositives + in_trueNegatives)/totalCount

def prevalance(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    totalCount = float(in_truePositives+in_falsePositives+in_trueNegatives+in_falseNegatives)
    realPositives = float(in_truePositives+in_falseNegatives)
    return realPositives/totalCount

def bias(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    totalCount = float(in_truePositives+in_falsePositives+in_trueNegatives+in_falseNegatives)
    predictedPositives = float(in_truePositives+in_falsePositives)
    return predictedPositives/totalCount

def skew(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    realNegatives = float(in_trueNegatives+in_falsePositives)
    realPositives = float(in_truePositives+in_falseNegatives)
    return realNegatives/realPositives

# Using alternative definition from "The truth of the F-measure" by Yutaka Sasaky, 2007 
def f1Measure(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    P = precision(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives)
    R = recall(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives)
    return 0.0 if (P+R) == 0 else 2*(P*R)/(P+R)

# Using definition from "The truth of the F-measure" by Yutaka Sasaky, 2007 
def fBetaMeasure(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives, in_beta = 1):
    P = precision(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives)
    R = recall(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives)
    betaSquared = in_beta*in_beta
    return 0.0 if (P+R) == 0 else (betaSquared + 1)*(P*R)/(betaSquared*P+R)

# Using definition from "The truth of the F-measure" by Yutaka Sasaky, 2007 
def f2Measure(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives):
    return fBetaMeasure(in_truePositives, in_falsePositives, in_trueNegatives, in_falseNegatives, in_beta = 2)

# Wrapper to call one of the functions of this Utils with the results in a dictionary format as follows:
# 'tp' - key for true positive count
# 'fp' - key for false positive count
# 'tn' - key for true negative count
# 'fn' - key for false negative count
# The function parameter in_baseResult represents an initial count to start from, is optional, and has the same format
def resultFormatCall(in_func, in_result, in_baseResult = None):
    if in_baseResult:
        return in_func(in_result['tp']-in_baseResult['tp'], in_result['fp']-in_baseResult['fp'], in_result['tn']-in_baseResult['tn'], in_result['fn']-in_baseResult['fn'])
    else:
        return in_func(in_result['tp'], in_result['fp'], in_result['tn'], in_result['fn'])