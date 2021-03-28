import nltk
from nltk.corpus import stopwords
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
porter = nltk.stem.PorterStemmer()

def read(fileName): #Function reading a txt file and givind as output the number of occurence of each word
    f = open(fileName)
    m = f.read()
    token = nltk.word_tokenize(m,preserve_line=False)
    token = [element for element in token if not element in stopwords.words()] #Question 5 remove this line to include the stopword

    stem = [porter.stem(w) for w in token]
    stem = np.asarray(stem)
    stemUnique = np.unique(stem)
    stem = list(stem)

    occurence = [[element,stem.count(element)] for element in stemUnique]
    occurence.sort(key = lambda x:x[1])
    occurence = occurence[::-1]
    return occurence


def convert(occurences): # Enable cloud word plotting
    txt = []
    for element in occurences:
        for i in range(element[1]):
            txt.append(element[0])
    return " ".join(txt)


def plot_cloud(wordcloud): #Plot tag clound

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()



def plotTagCloud(occurenceList): #Plot tag cloud depending on the number of occurence of each word
    wordcloud = WordCloud(width = 500, height = 500, background_color='pink',collocations = False, random_state=10).generate(convert(occurenceList))
    plot_cloud(wordcloud)



def takeCorpus(folderName): # Plot tag clound for each file (Taking the 50 first element)
    path = "./"+folderName+"/*.txt"
    filesNames=glob.glob(path)[0:15]
    result = dict()
    for fileName in filesNames:
        result[fileName] = read(fileName)
    for occurenceList in result:
        plotTagCloud(result[occurenceList][0:50])

#takeCorpus("nasa")


def termFrequency(fileName): # Compute termFrequency from a txt file
    f = open(fileName)
    m = f.read()
    token = nltk.word_tokenize(m,preserve_line=False)
    token = [element for element in token if not element in stopwords.words()] #Question 5 remove this line to include the stopword

    stem = [porter.stem(w) for w in token]
    stem = np.asarray(stem)
    stemUnique = np.unique(stem)
    stem = list(stem)

    occurence = [[element,stem.count(element)] for element in stemUnique]
    occurence.sort(key = lambda x:x[1])
    occurence = occurence[::-1]
    newOccurence = []

    for word in occurence:
        if not (len(word[0]) == 1 or word[0] == "''"):
            newOccurence.append(word)

    frequencyDictionary = dict()
    for word in newOccurence:
        frequencyDictionary[word[0]] = word[1]

    maximumFrequency = newOccurence[0][1]

    termFrequency = dict()

    for word in newOccurence:
        termFrequency[word[0]] = round(word[1]/maximumFrequency,2)


    return termFrequency,frequencyDictionary


def inverseDocumentFrequency(folderName): # Compute inverse Document Frequency from nasa folder
    N = 15 #Number of taken documents

    path = "./"+folderName+"/*.txt"
    filesNames=glob.glob(path)[0:N]

    InverseFrequencyDictionary = dict()

    FrequencyDictionaries = dict()

    for fileName in filesNames:
        _, frequencyDictionary = termFrequency(fileName)
        FrequencyDictionaries[fileName] = frequencyDictionary


    for fileName in FrequencyDictionaries:

        a = FrequencyDictionaries[fileName]
        inverseFrequency = dict()
        for word in a:
            n_i = 0
            for name in FrequencyDictionaries:
                b = FrequencyDictionaries[name]
                if word in b:
                    n_i += 1


            inverseFrequency[word] = round(np.log(N/n_i),2)

        InverseFrequencyDictionary[fileName] = inverseFrequency

    return InverseFrequencyDictionary # retourne un dictionnaire contenant pour chaque mot de chaque txt, log(N/ni) -> N = nbr document (15)


#Question 3 and 6 depending if we include stop word or not.


P = [3,5,10]  # Number of taken weights
N = 15  # Numbers of taken documents
folderName = "nasa"
path = "./" + folderName + "/*.txt"
filesNames = glob.glob(path)[0:N]
pathKey = "./" + folderName + "/*.key"
KeysNames = glob.glob(pathKey)[0:N]
inverseFreq = inverseDocumentFrequency(folderName)

for i,f in enumerate(filesNames):
    weightDictionary = dict()
    print("For filename:",f," we have: ")
    termFreq,_ = termFrequency(f)
    print(" ")
    print("Term Frequency: ")
    print(termFreq)
    print(" ")
    print("Inverse Term Frequency: ")
    print(inverseFreq[f])
    print(" ")
    for word in termFreq:
        weightDictionary[word] = termFreq[word]*inverseFreq[f][word]

    print("Weight: ")
    sortedWeightDictionary = dict(sorted(weightDictionary.items(), key=lambda item: item[1],reverse=True))
    print(sortedWeightDictionary)
    for p in P:
        pWeights = list(sortedWeightDictionary)[0:p]
        print(" ")
        print(p," potential key words: ")
        print(pWeights)
        print(" ")
    f = open(KeysNames[i])
    m = f.read()
    print("KeyWords: ")
    print(" ")
    print(m)
    print("************************************")


## Question 4
N = 15
def createTermDocumentMatrix(folderName,p):
    inverseFreq = inverseDocumentFrequency(folderName)
    path = "./" + folderName + "/*.txt"
    filesNames = glob.glob(path)[0:N]
    termDocumentMatrix = []

    for files in filesNames:
        weightDictionary = dict()
        termFreq,DictFreq = termFrequency(files)
        for word in termFreq:
            weightDictionary[word] = termFreq[word]*inverseFreq[files][word]
            sortedWeightDictionary = dict(sorted(weightDictionary.items(), key=lambda item: item[1], reverse=True))

        pWeights = list(sortedWeightDictionary)[0:p]
        termDocumentMatrix.append(pWeights)

    termDocumentMatrix = np.asarray(termDocumentMatrix)
    T = list(termDocumentMatrix.flatten())

    return termDocumentMatrix,T #Return termDocumentMatrix and T the same matrix but flaten

def compQueryBoolean(query,termDocumentmMatrix):
    p = len(termDocumentmMatrix[0])
    validFiles = []
    scores = []

    for i,document in enumerate(termDocumentmMatrix):
        score = 0
        for word in query:
            if word in document:
                score += 1
        scores.append([score,i])

    scores.sort(key = lambda x:x[0])
    scores = scores[::-1]

    path = "./" + "nasa"+ "/*.txt"
    filesNames=glob.glob(path)[0:15]

    for score in scores:
        m = []
        index = score[1]
        m.append(filesNames[index])
        m.append(score[0])
        validFiles.append(m)
    return validFiles


def queryBooleanRepresentation(k,T,p):
    queries = []

    for i in range(k):
        query = []
        r = random.randint(1,p)# Taille des sous query
        for j in range(r):
            m = random.randint(0,len(T)-1) # index de l'élément choisis pour la query
            query.append(T[m])
        queries.append(query)

    return queries

def queryVectorRepresentation(k,p,N):
    size = N * p
    queries = []

    for i in range(k):
        query = []
        for j in range(size):
            m = random.randint(0,1)
            query.append(m)
        queries.append(query)

    return queries

def comptVectorRepresentation(folderName,p):
    inverseFreq = inverseDocumentFrequency(folderName)
    path = "./" + folderName + "/*.txt"
    filesNames = glob.glob(path)[0:15]
    _,T = createTermDocumentMatrix(folderName,p)
    vectorRepresentation = dict()
    for files in filesNames:
        weightDictionary = dict()
        termFreq, DictFreq = termFrequency(files)
        for word in termFreq:
            weightDictionary[word] = termFreq[word] * inverseFreq[files][word]

        documentVector = []
        for word in T:
            if word in weightDictionary:
                documentVector.append(weightDictionary[word])
            else:
                documentVector.append(0)

        vectorRepresentation[files] = documentVector
    return vectorRepresentation


def similarityMeasure(queryVector,documentVector):
    return np.dot(documentVector,queryVector)/(np.linalg.norm(documentVector)*np.linalg.norm(queryVector))

def compQueryVector(folderName,p,queries):

    vecteurRepresentation = comptVectorRepresentation(folderName,p)
    results = []

    for query in queries:
        scores = []
        for fileName in vecteurRepresentation:
            score = similarityMeasure(query,vecteurRepresentation[fileName])
            scores.append([fileName,score])

        scores.sort(key=lambda x: x[1])
        scores = scores[::-1]
        results.append(scores)

    return results

# Q4
folderName = "nasa"
N = 15
p = 10
k = 4
termDocumentMatrix,T = createTermDocumentMatrix(folderName,p)


query = ['optic', 'integr', 'fabric', 'code-v', 'patran', 'electron', 'analys', 'nastran', 'thermal', 'possibl']
indexes = []
for element in query:
    indexes.append(T.index(element))

vectorQuery = []
for i in range(150):
    if i in indexes:
        vectorQuery.append(1)
    else:
        vectorQuery.append(0)

print("Boolean Query: ",query)
print("Same Boolean Query as Vector Query: ",vectorQuery)

print(" ")
print("Result from Boolean model: ",compQueryBoolean(query,termDocumentMatrix))
print("Result from Vector model: ",compQueryVector(folderName,p,[vectorQuery]))






























