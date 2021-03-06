#==============================================================================
#  Classificacao KNN no Conjunto de Dados IRIS
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Iris em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../data/iris.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,-1].values

#------------------------------------------------------------------------------
#  Separar 100 amostras para treinamento e 50 amostras para teste
#------------------------------------------------------------------------------

# Jeito ERRADO de fazer:
    
#X_treino = X[:100,:]
#X_teste  = X[100:,:]

#y_treino = y[:100]
#y_teste  = y[100:]

# Jeito CERTO de fazer (randomizado):

from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=50,
    random_state=42
    )

#------------------------------------------------------------------------------
#  Importar a classe KNeighborsClassifier do pacote sklearn.neighbors
#------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

#------------------------------------------------------------------------------
#  Criar o classificador
#------------------------------------------------------------------------------

classificador = KNeighborsClassifier(
    n_neighbors = 10,
    weights     = 'uniform',
    n_jobs      = -1   
    )

#------------------------------------------------------------------------------
#  Treinar o classificador
#------------------------------------------------------------------------------

classificador.fit(X_treino,y_treino)

#------------------------------------------------------------------------------
#  Testar o classificador
#------------------------------------------------------------------------------

y_resposta = classificador.predict(X_teste)

#------------------------------------------------------------------------------
#  Mostrar a matriz de confusão e a medida de acuracia
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score

print ('Matriz de Confusao:')

print (confusion_matrix(y_teste,y_resposta))

print ( 'Acuracia:',100*accuracy_score(y_teste,y_resposta), '%')
