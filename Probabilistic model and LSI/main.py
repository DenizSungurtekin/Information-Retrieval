import nltk
from nltk.corpus import stopwords
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
porter = nltk.stem.PorterStemmer()


def read(fileName):  # Function reading a txt file and givind as output the number of occurence of each word
    f = open(fileName)
    m = f.read()
    token = nltk.word_tokenize(m, preserve_line=False)
    token = [element for element in token if
             not element in stopwords.words()]  # Question 5 remove this line to include the stopword

    stem = [porter.stem(w) for w in token]
    stem = np.asarray(stem)
    stemUnique = np.unique(stem)
    stem = list(stem)

    occurence = [[element, stem.count(element)] for element in stemUnique]
    occurence.sort(key=lambda x: x[1])
    occurence = occurence[::-1]
    return occurence


def convert(occurences):  # Enable cloud word plotting
    txt = []
    for element in occurences:
        for i in range(element[1]):
            txt.append(element[0])
    return " ".join(txt)


def plot_cloud(wordcloud):  # Plot tag clound

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def plotTagCloud(occurenceList):  # Plot tag cloud depending on the number of occurence of each word
    wordcloud = WordCloud(width=500, height=500, background_color='pink', collocations=False, random_state=10).generate(
        convert(occurenceList))
    plot_cloud(wordcloud)


def takeCorpus(folderName, firstElement, filesNumbers):  # Plot tag clound for each file (Taking the 50 first element)
    path = "./" + folderName + "/*.txt"
    filesNames = glob.glob(path)[0:filesNumbers]
    result = dict()
    for fileName in filesNames:
        result[fileName] = read(fileName)
    for occurenceList in result:
        plotTagCloud(result[occurenceList][0:firstElement])


def termFrequency(fileName):  # Compute termFrequency from a txt file
    f = open(fileName)
    m = f.read()
    token = nltk.word_tokenize(m, preserve_line=False)
    token = [element for element in token if
             not element in stopwords.words()]  # Question 5 remove this line to include the stopword

    stem = [porter.stem(w) for w in token]
    stem = np.asarray(stem)
    stemUnique = np.unique(stem)
    stem = list(stem)

    occurence = [[element, stem.count(element)] for element in stemUnique]
    occurence.sort(key=lambda x: x[1])
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
        termFrequency[word[0]] = round(word[1] / maximumFrequency, 2)

    return termFrequency, frequencyDictionary



def inverseDocumentFrequency(folderName):  # Compute inverse Document Frequency from nasa folder
    N = 15  # Number of taken documents

    path = "./" + folderName + "/*.txt"
    filesNames = glob.glob(path)[0:N]

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

            inverseFrequency[word] = round(np.log(N / n_i), 2)

        InverseFrequencyDictionary[fileName] = inverseFrequency

    return InverseFrequencyDictionary  # retourne un dictionnaire contenant pour chaque mot de chaque txt, log(N/ni) -> N = nbr document (15)



def createTermDocumentMatrix(folderName, p, N):  # N = number of file taken in account
    inverseFreq = inverseDocumentFrequency(folderName)
    path = "./" + folderName + "/*.txt"
    filesNames = glob.glob(path)[0:N]
    termDocumentMatrix = []
    termDocumentMatrixWeight = []

    for files in filesNames:
        weightDictionary = dict()
        termFreq, DictFreq = termFrequency(files)
        for word in termFreq:
            weightDictionary[word] = termFreq[word] * inverseFreq[files][word]

            if weightDictionary[word] > 1:
                weightDictionary[word] = 1

            sortedWeightDictionary = dict(sorted(weightDictionary.items(), key=lambda item: item[1], reverse=True))

        sortedWeight = sorted(weightDictionary.values(), reverse=True)[0:p]
        pWeights = list(sortedWeightDictionary)[0:p]
        termDocumentMatrix.append(pWeights)
        termDocumentMatrixWeight.append(sortedWeight)

    termDocumentMatrix = np.asarray(termDocumentMatrix)
    termDocumentMatrixWeight = np.asarray(termDocumentMatrixWeight)
    T = list(termDocumentMatrix.flatten())
    Tweight = list(termDocumentMatrixWeight.flatten())

    return termDocumentMatrix, T, termDocumentMatrixWeight, Tweight  # Return termDocumentMatrix, T the same matrix but flaten and Corresponding weight matrix and the flatten version



### Functions for probabilistic model

def computeIrellevantInitProba(terme, T_m, N):
    return T_m.count(terme) / N


def generateQueryProba(size):  # Generate a binary query
    return [random.randint(0, 1) for i in range(size)]


def similarityProba(query, RelevanceProba, IRelevantProba, M, T_p):
    p = len(M[0])
    similarities = []
    for i in range(len(M)):
        partQuery = query[i * p:i * p + p]
        relevance = RelevanceProba[i * p:i * p + p]
        irelevance = IRelevantProba[i * p:i * p + p]
        t = T_p[i * p:i * p + p]
        values = [partQuery[j] * t[j] * (
                    abs(np.log(relevance[j] / (1 - relevance[j]))) + np.log((1 - irelevance[j]) / irelevance[j])) for j
                  in range(p)]
        somme = sum(values)
        similarities.append([i, somme])

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities  # Return list with file Number and similariti value


def updateOne(V, initRelevanceProba, initNonRelevantProba, N, M, T_p, query, r):
    totalWords = []
    for i in V:
        totalWords.append(M[i[0]])

    totalWords = np.asarray(totalWords)
    totalWords = totalWords.flatten()

    for i in range(len(T_m)):
        V_i = list(totalWords).count(T_m[i])
        initRelevanceProba[i] = (V_i + 0.5) / (r + 1)
        n_i = T_m.count(T_m[i])
        initNonRelevantProba[i] = (n_i - V_i + 0.5) / (N - r + 1)

    similarities = similarityProba(query, initRelevanceProba, initNonRelevantProba, M, T_p)
    V = similarities[0:r]

    return similarities


#Adjust parameters and generate a query to obtain the corresponding list of files
p = 5
N = 15
M,T_m,P,T_p = createTermDocumentMatrix("nasa",p,N)


initRelevanceProba = [0.5 for i in range(len(T_p))]
initNonRelevantProba = [computeIrellevantInitProba(terme,T_m,N) for terme in T_m]

query = generateQueryProba(len(T_p))

similarities = similarityProba(query,initRelevanceProba,initNonRelevantProba,M,T_p)



r = 7
V = similarities[0:r]

result = updateOne(V,initRelevanceProba,initNonRelevantProba,N,M,T_p,query,r)

print("First elements of the query: ",query[0:10])
print("Result:")
for element in result:
    print("File ",element[0]," with score ",element[1])



### Latent semantic indexing model


# Computation of M
M_semantic = P.transpose()
print("M: ")
print(M_semantic)
print(M_semantic.shape)
print(" ")

S,delta,D_t= np.linalg.svd(M_semantic,full_matrices=False)
print("Reconstruction verification")
print(np.allclose(M_semantic, np.dot(S * delta, D_t)))

print("S :")
print(S.shape)

print("D :")
print(D_t.shape)

print("Delta: ")
print(delta.shape)


## Take the l largest singular value and reconstruct M

l = 3

delta_l = delta[0:l]
S_l = S[:,0:l]
D_t_l = D_t[0:l,:]
print("Delta",delta_l.shape)
print("S_l",S_l.shape)
print("D_l",D_t_l.shape)

#reconstruction

M_l = np.dot(S_l * delta_l, D_t_l)
print("M_l: ",M_l.shape)
print(M_l)




M_corr = np.dot(M_l.T,M_l)
print("M_corr: ",M_corr.shape)
print(M_corr)

indexes = []
for element in sorted(M_corr[0], reverse=True):
    indexes.append([list(M_corr[0]).index(element), element])

print(len(indexes))
print("Result:")
for element in indexes:
    print("File ", element[0], " with score ", element[1])
