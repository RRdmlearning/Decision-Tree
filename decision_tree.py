import csv
from collections import defaultdict
import pydotplus
import numpy as np


# Important part
class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data


def calculateDiffCount(datas):
    # 将输入的数据汇总(input dataSet)
    # return results Set{type1:type1Count,type2:type2Count ... typeN:typeNCount}

    results = {}
    for data in datas:
        # data[-1] means dataType
        if data[-1] not in results:
            results[data[-1]] = 1
        else:
            results[data[-1]] += 1
    return results


def gini(rows):
    # 计算gini值(Calculate GINI)

    length = len(rows)
    results = calculateDiffCount(rows)
    imp = 0.0
    for i in results:
        imp += results[i] / length * results[i] / length
    return 1 - imp


def splitDatas(rows, value, column):
    # 根据条件分离数据集(splitDatas by value,column)
    # return 2 part(list1,list2)

    list1 = []
    list2 = []
    if (isinstance(value, int) or isinstance(value, float)):  # for int and float type
        for row in rows:
            if (row[column] >= value):
                list1.append(row)
            else:
                list2.append(row)
    else:  # for String type
        for row in rows:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)

    return (list1, list2)


def buildDecisionTree(rows, evaluationFunction=gini):
    # 递归建立决策树,当gain = 0 时停止递归
    # bulid decision tree by recursive function
    # stop recursive function when gain = 0
    # return tree

    currentGain = evaluationFunction(rows)
    column_length = len(rows[0])
    rows_length = len(rows)
    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for col in range(column_length - 1):
        col_value_set = set([x[col] for x in rows])
        for value in col_value_set:
            list1, list2 = splitDatas(rows, value, col)
            p = len(list1) / rows_length
            gain = currentGain - p * evaluationFunction(list1) - (1 - p) * evaluationFunction(list2)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)

    dcY = {'impurity': '%.3f' % currentGain, 'samples': '%d' % rows_length}

    # stop or not stop
    if best_gain > 0:
        trueBranch = buildDecisionTree(best_set[0], evaluationFunction)
        falseBranch = buildDecisionTree(best_set[1], evaluationFunction)
        return Tree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY)
    else:
        return Tree(results=calculateDiffCount(rows), summary=dcY, data=rows)


def prune(tree, miniGain, evaluationFunction=gini):
    # 剪枝, when gain < mini Gain，合并(merge the trueBranch and the falseBranch)

    if tree.trueBranch.results == None: prune(tree.trueBranch, miniGain, evaluationFunction)
    if tree.falseBranch.results == None: prune(tree.falseBranch, miniGain, evaluationFunction)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.trueBranch.data)
        len2 = len(tree.falseBranch.data)
        len3 = len(tree.trueBranch.data + tree.falseBranch.data)
        p = float(len1) / (len1 + len2)
        gain = evaluationFunction(tree.trueBranch.data + tree.falseBranch.data) - p * evaluationFunction(
            tree.trueBranch.data) - (1 - p) * evaluationFunction(tree.falseBranch.data)
        if (gain < miniGain):
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculateDiffCount(tree.data)
            tree.trueBranch = None
            tree.falseBranch = None


def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


#下面是辅助代码画出树
#Unimportant part
#plot tree and load data
def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s?' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s?' % (szCol, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def dotgraph(decisionTree):
    global dcHeadings
    dcNodes = defaultdict(list)
    """Plots the obtained decision tree. """

    def toString(iSplit, decisionTree, bBranch, szParent="null", indent=''):
        if decisionTree.results != None:  # leaf node
            lsY = []
            for szX, n in decisionTree.results.items():
                lsY.append('%s:%d' % (szX, n))
            dcY = {"name": "%s" % ', '.join(lsY), "parent": szParent}
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append(['leaf', dcY['name'], szParent, bBranch, dcSummary['impurity'],
                                    dcSummary['samples']])
            return dcY
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s' % (szCol, decisionTree.value)
            trueBranch = toString(iSplit + 1, decisionTree.trueBranch, True, decision, indent + '\t\t')
            falseBranch = toString(iSplit + 1, decisionTree.falseBranch, False, decision, indent + '\t\t')
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append([iSplit + 1, decision, szParent, bBranch, dcSummary['impurity'],
                                    dcSummary['samples']])
            return

    toString(0, decisionTree, None)
    lsDot = ['digraph Tree {',
             'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
             'edge [fontname=helvetica] ;'
             ]
    i_node = 0
    dcParent = {}
    for nSplit, lsY in dcNodes.items():
        for lsX in lsY:
            iSplit, decision, szParent, bBranch, szImpurity, szSamples = lsX
            if type(iSplit) == int:
                szSplit = '%d-%s' % (iSplit, decision)
                dcParent[szSplit] = i_node
                lsDot.append('%d [label=<%s<br/>impurity %s<br/>samples %s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                                                          decision.replace(
                                                                                                              '>=',
                                                                                                              '&ge;').replace(
                                                                                                              '?', ''),
                                                                                                          szImpurity,
                                                                                                          szSamples))
            else:
                lsDot.append('%d [label=<impurity %s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                                                                szImpurity,
                                                                                                                szSamples,
                                                                                                                decision))

            if szParent != 'null':
                if bBranch:
                    szAngle = '45'
                    szHeadLabel = 'True'
                else:
                    szAngle = '-45'
                    szHeadLabel = 'False'
                szSplit = '%d-%s' % (nSplit, szParent)
                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                                                                    i_node, szAngle,
                                                                                                    szHeadLabel))
                else:
                    lsDot.append('%d -> %d ;' % (p_node, i_node))
            i_node += 1
    lsDot.append('}')
    dot_data = '\n'.join(lsDot)
    return dot_data

def loadCSV(file):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    dcHeader = {}
    if bHeader:
        lsHeader = next(reader)
        for i, szY in enumerate(lsHeader):
                szCol = 'Column %d' % i
                dcHeader[szCol] = str(szY)
    return dcHeader, [[convertTypes(item) for item in row] for row in reader]



bHeader = True
# the bigger example
dcHeadings, trainingData = loadCSV('fishiris.csv') # demo data from matlab
decisionTree = buildDecisionTree(trainingData, evaluationFunction=gini)
result = plot(decisionTree)
prune(decisionTree, 0.4) # notify, when a branch is pruned (one time in this example)
result = plot(decisionTree)
dot_data = dotgraph(decisionTree)
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("iris.pdf")
graph.write_png("prune.png")