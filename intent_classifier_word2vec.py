import gensim
import pandas as pd
import nltk as nl
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
import numpy as np



def comp_mat(path):
    dbFilepandas = pd.read_csv(path).apply(lambda x: x.astype(str).str.lower())
    train = []
    STOP_WORDS = nl.corpus.stopwords.words()
    categories = ['book restaraunt', 'get weather', 'outlier']

    train1 = []
    for sentences in dbFilepandas[dbFilepandas.columns].values:
        train.extend(sentences)

    train1.extend([w for w in train if not w in categories])
    print(train1)
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(train1)):
        train1[i] = tokenizer.tokenize(train1[i])
    for sentence in train1:
        for word in sentence:
            if word in STOP_WORDS:
                sentence.remove(word)
    Y_dataset =[]
    for i in dbFilepandas.Intention:
        Y_dataset.append(i)

    matrix = [train1, Y_dataset]
    return matrix


def comp_mat_list(lst):
    STOP_WORDS = nl.corpus.stopwords.words()

    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(lst)):
        lst[i] = tokenizer.tokenize(lst[i])
    for sentence in lst:
        for word in sentence:
            if word in STOP_WORDS:
                sentence.remove(word)
    return lst


def build_model(path):
    train = comp_mat(path)[0]
    model = gensim.models.Word2Vec(train, size=100, min_count=1, workers=4)
    model.train(train, total_examples=len(train), epochs=40)
    return model


def convert_sentences(path, path2):
    sentences = comp_mat(path)[0]
    model = build_model(path2)
    sent_list = []
    a = []
    vocab = model.wv
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] in vocab:
                a.append(model.wv[sentences[i][j]])
            else:
                a.append(0)
        sent_list.append(a)
        a= []

    #finding sum
    sum = np.full((1, 100), 0.0)
    le = 0
    senmat = []
    for i in range(len(sent_list)):
        for j in range(len(sent_list[i])):
            sum = sum + np.asarray(sent_list[i][j])
            le = le+1
        sum = sum / le
        le = 0
        senmat.append(sum[0])
        sum = np.full((1, 100), 0.0)
    #done!
    senmat = np.asarray(senmat)
    senmat = np.nan_to_num(senmat)
    return senmat


def convert_list_sentences(lst, path2):
    sentences = comp_mat_list(lst)
    model = build_model(path2)
    sent_list = []
    a = []
    vocab = model.wv
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] in vocab:
                a.append(model.wv[sentences[i][j]])
            else:
                a.append(0)
        sent_list.append(a)
        a= []
    print(a)

    #finding sum
    sum = np.full((1, 100), 0.0)
    le = 0
    senmat = []
    for i in range(len(sent_list)):
        for j in range(len(sent_list[i])):
            sum = sum + np.asarray(sent_list[i][j])
            le = le+1
        sum = sum / le
        le = 0
        senmat.append(sum[0])
        sum = np.full((1, 100), 0.0)
    #done!
    return senmat


def train(path):
    X = convert_sentences(path, path)


    Y_dataset = comp_mat(path)[1]
    Y = Y_dataset[:len(X)]
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(X[:1000], Y[:1000])
    print('Training is Successful!')


def test(path, path2):
    test_mat = convert_sentences(path, path2)[:3000]
    X = convert_sentences(path2, path2)[3001:]
    Y_dataset = comp_mat(path2)[1]
    Y = Y_dataset[3001:]
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(X, Y)
    Y_dataset_test = comp_mat(path)[1]
    Y_test = np.asarray(Y_dataset_test[:3000])
    predict = clf.predict(test_mat)
    predict = predict.tolist()
    predict = np.asarray(predict)
    print('ANSWER: ', predict)
    print('Dataset: ', Y_test)
    score = clf.score(test_mat, Y_test)
    print(score)


def test_str(list, path2):

    test_mat = convert_list_sentences(list, path2)
    X = convert_sentences(path2, path2)
    Y_dataset = comp_mat(path2)[1]
    Y = Y_dataset[:len(X)]
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000)
    clf.fit(X, Y)
    predict = clf.predict(test_mat)
    predict = predict.tolist()
    predict = np.asarray(predict)
    print('ANSWER: ', predict)



#test('C:\\Users\\123\Desktop\\Full Data.csv','C:\\Users\\123\Desktop\\Full Data.csv' )

#test_str(['is it gonna be rainy tomorrow?', 'i wish i could find something to eat', 'hello! nice to meet you!'],'C:\\Users\\123\Desktop\\Full Data.csv' )











