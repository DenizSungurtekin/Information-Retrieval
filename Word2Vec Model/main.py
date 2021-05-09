#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import glob
from nltk.tokenize import RegexpTokenizer
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
#porter = nltk.stem.PorterStemmer()
import time


# In[ ]:


# Useful function to creat our dataset

def read(fileName):  # Function reading a txt file and giving as output the number of occurence of each word (Only word)
    f = open(fileName,encoding="utf-8")
    m = f.read()
    return m

def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))
    training_data = []
    sentences = corpus.split(".")
    
    sentences = sentences[5:] #put off five first sentence which is note in the book
    nbSentences = len(sentences)
    
    #Take only a part of sentences otherwise computation time is too much long
    bound = int(nbSentences/10) # -> 785 sentences otherwise we have more than 7000 sentences
    sentences = sentences[:bound]
    #sentences = sentences[:10] #smaller dataset
    
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence if word not in stopwords.words('french') + stopwords.words('english')]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data


def prepare_data_for_training(sentences,window_size): #, w2v):
    data = {}
    X_train = []
    y_train = []
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1

    V = len(data)

    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i

    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]

            for j in range(i - window_size, i + window_size):
                if i != j and j >= 0 and j < len(sentence):
                    context[vocab[sentence[j]]] += 1
                    
                    
            X_train.append(center_word)
            y_train.append(context)         

    return X_train, y_train,V,data


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[ ]:


times = []
losses = []
likelihoods = []
##Small parameter: 10sentences,alpha= 0.005,lamba = 0.001,embed = 1000,epoch = 70 -> convergence more epoch learning too slow
##big parameter: 10sentences,alpha= 0.002 (if bigger, divergence),lamba = 0.00001,embed = 10,epoch = 30 -> convergence more epoch learning too slow
# Model object
class word2vec(object):
    def __init__(self):
        self.embedding_size = 10
        self.alpha_coeff = 0.002
        self.word_index = {}
        self.window_size = 2
        self.words = []
        self.X_train = []
        self.y_train = []
        self.tag = []


    def initialize(self, V, data):
        self.V = V
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.embedding_size))#Can have an impact
        self.W_hidden = np.random.uniform(-0.8, 0.8, (self.embedding_size, self.V))

        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    def forward_compute(self, X):
        self.h = np.dot(self.W.T, X).reshape(self.embedding_size, 1)
        self.u = np.dot(self.W_hidden.T, self.h)
        self.y = softmax(self.u)
        return self.y

    def gradient_descend(self, x, t):
        e = self.y - np.asarray(t).reshape(self.V, 1)
        dW_hidden = np.dot(self.h, e.T)
        X = np.array(x).reshape(self.V, 1)
        dLdW = np.dot(X, np.dot(self.W_hidden, e).T)
        self.W_hidden = self.W_hidden - self.alpha_coeff * dW_hidden
        self.W = self.W - self.alpha_coeff * dLdW

    def train(self, epochs):

        # ----------   SkipGram model ----------

        coeff_lambda = 0.00001
        # Computing likelihood take much more time so we decided to directly compute the loss function and analyse its evolution
        # which is equivalent. We minimize the negative log likelihood which is the loss function.
        # Because we can rewrite loss function with two sum to reduce operations.
        
        # for x in range(1, epochs):
        #     start_time = time.time()
        #     self.loss = 0
        #     self.likelihood = 0
            
        #     for j in range(len(self.X_train)):
        #         self.forward_compute(self.X_train[j])
        #         self.gradient_descend(self.X_train[j], self.y_train[j])
        #         C = 0
        #         for m in range(self.V):
        #             if (self.y_train[j][m]):
                          
        #                 self.loss += -1 * self.u[m][0] 
        #                 C += 1

        #         self.loss += C * np.log(np.sum(np.exp(self.u)))
                
        #         #self.likelihood = coeff_lambda * self.likelihood  #negative sampling
        #         #self.loss = -np.log(self.likelihood) #negative sampling
            
        #     # Computations values
        #     self.loss = self.loss * coeff_lambda 
        #     self.likelihood = np.exp((-self.loss) )
        #     print("epoch ", x, " loss = ", self.loss, " likelihood: ", self.likelihood)
        #     self.alpha_coeff *= 1 / ((1 + self.alpha_coeff * x))
        #     times.append((time.time() - start_time))
        #     losses.append(self.loss)
        #     likelihoods.append(self.likelihood)
            
        # ----------   SkipGram model ----------

        # ----------  Negative Sampling Model --------

        for x in range(1, epochs):
            start_time = time.time()
            self.loss = 0
            self.likelihood = 0
            
            for j in range(len(self.X_train)):
                self.forward_compute(self.X_train[j])
                self.gradient_descend(self.X_train[j], self.y_train[j])

                for m in range(self.V):
                  if (self.y_train[j][m] == 1):
                    self.likelihood += self.tag[j]*np.log(sigmoid(self.u[m][0])) + (1-self.tag[j])*np.log(1 - sigmoid(self.u[m][0]))
                
            # Computations values
            self.likelihood = -1 * coeff_lambda * self.likelihood
            self.loss = -np.log(self.likelihood)
            print("epoch ", x, " loss = ", self.loss, " likelihood: ", self.likelihood)
            self.alpha_coeff *= 1 / ((1 + self.alpha_coeff * x))
            times.append((time.time() - start_time))
            losses.append(self.loss)
            likelihoods.append(self.likelihood)



        # ----------  END Negative Sampling Model --------
            

    def predict(self, word, number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.forward_compute(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i

            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if (len(top_context_words) >= number_of_predictions):
                    break

            return top_context_words
        else:
            print("Word not found in dicitonary")


# In[ ]:



# Main
corpus = read("VingMilleLieues.txt")
window_size = 2
training_data = preprocessing(corpus)


# In[ ]:


def create_dataset(training_data,window_size,Negative_sampling,n):
  X_train = []
  y_train = []

  Vocabulary = []

  if Negative_sampling:
    print("Negative sampling done here")
    print("Number of samplings :", n)

  for sentence in training_data:

    for word in sentence:
        if word not in Vocabulary:
          Vocabulary.append(word)

    for target in (sentence[window_size:len(sentence)-2]):
        

      index = sentence.index(target)
      X_train.append((target,sentence[index-1]))
      X_train.append((target,sentence[index-2]))
      X_train.append((target,sentence[index+1]))
      X_train.append((target,sentence[index+2]))
      y_train.append(1)
      y_train.append(1)
      y_train.append(1)
      y_train.append(1)

      if Negative_sampling:
        for i in range(n):
          ind = random.randint(index,index+10)
          ind = ind % len(sentence)
          X_train.append((target,sentence[ind]))
          y_train.append(0)

         
  return X_train,y_train,Vocabulary


# In[ ]:


def hot_vector(data,Vocabulary):

    X_coded = []
    y_coded = []

    for tup in X:
      target = tup[0]
      context = tup[1]

      # print("target : ",target)
      # print("context : ",context)

      a = [0 for i in range(len(Vocabulary))]
      b = [0 for i in range(len(Vocabulary))]

      a[Vocabulary.index(target)] = 1
      b[Vocabulary.index(context)] = 1

      X_coded.append(a)
      y_coded.append(b)

    return X_coded,y_coded


# In[ ]:


X,tag,Vocabulary = create_dataset(training_data,2,True,2)
print(X[0:12])
print(tag[0:12])
print(len(Vocabulary))
print(Vocabulary[0:10])

print("")
print("***************")
print("")



X_coded , y_coded = hot_vector(X,Vocabulary)


# In[ ]:


w2v = word2vec()

w2v.X_train = X_coded
w2v.y_train = y_coded
w2v.tag = tag
w2v.initialize(len(Vocabulary), Vocabulary)

epochs = 31
w2v.train(epochs)


# In[ ]:


#Plot graphs
x = [i for i in range(epochs-1)]

plt.plot(x,losses)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss Evolutions")
plt.show()

plt.plot(x,times)
plt.xlabel("epochs")
plt.ylabel("times")
plt.title("Times Evolutions")
plt.show()

plt.plot(x,likelihoods)
plt.xlabel("epochs")
plt.ylabel("likelihood")
plt.title("Likelihood Evolutions")
plt.show()


# In[ ]:


#Result for 10 very frequent words

def removeOne(liste):
    for element in liste:
        if len(element)<2 or element == '»' or element == '«':
            liste.remove(element)
    return liste
    

# print("mer: ",w2v.predict("mer", 5))
# print("land: ",w2v.predict("land", 5))
# print("capitaine: ",w2v.predict("capitaine", 5))
# print("monsieur: ",w2v.predict("monsieur", 5))
# print("conseil: ",w2v.predict("conseil", 5))

# print("être: ",w2v.predict("être", 5))
# print("dit: ",w2v.predict("dit",5))
# print("deux: ",w2v.predict("deux", 5))
# print("dont: ",w2v.predict("dont", 5))
# print("si: ",w2v.predict("si", 5))


print("mer: ",removeOne(w2v.predict("mer", 8))[1:])
print("capitaine: ",removeOne(w2v.predict("capitaine", 5)))
print("monsieur: ",removeOne(w2v.predict("monsieur", 9)))
print("conseil: ",removeOne(w2v.predict("conseil", 8))[1:])

print("être: ",removeOne(w2v.predict("être", 5)))
print("dit: ",removeOne(w2v.predict("dit", 8))[1:])
print("deux: ",removeOne(w2v.predict("deux", 7)))
print("dont: ",removeOne(w2v.predict("dont", 8))[1:])
print("si: ",removeOne(w2v.predict("si", 8)))


# In[ ]:


## Negatif Sampling

# Skip-gram Negative Sampling (SGNS) helps to speed up training time and improve quality
# of resulting word vectors. This is done by training the network to only modify a small percentage 
# of the weights rather than all of them. Recall in our example above, we update the weights 
# for every other word and this can take a very long time if the vocab size is large. With SGNS, 
# we only need to update the weights for the target word and a small number (e.g. 5 to 20) of random ‘negative’ words.

