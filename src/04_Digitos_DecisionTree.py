#==============================================================================
#  Classificacao KNN no Conjunto de Dados DIGITS
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Digits em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../data/digits.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:,1:-1].values
y = dataframe.iloc[:,-1].values

#------------------------------------------------------------------------------
#  Visualizar alguns digitos
#------------------------------------------------------------------------------

# import matplotlib.pyplot as plt

# for i in range(0,10):
#     plt.figure(figsize=(40,240))
#     d_plot = plt.subplot(1, 10, i+1)
#     d_plot.set_title("y = %.2f" % y[i])
 
#     d_plot.imshow(X[i,:].reshape(8,8),
#                   #interpolation='spline16',
#                   interpolation='nearest',
#                   cmap='binary',
#                   vmin=0 , vmax=16)
#     #plt.text(-8, 3, "y = %.2f" % y[i])

#     d_plot.set_xticks(())
#     d_plot.set_yticks(())
 
# plt.show()


#------------------------------------------------------------------------------
#  Dividir as amostras em conjunto de treino e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=500,
    random_state=42
    )

#------------------------------------------------------------------------------
#  Importar a classe KNeighborsClassifier do pacote sklearn.neighbors
#------------------------------------------------------------------------------

#from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier


#------------------------------------------------------------------------------
#  Criar o classificador
#------------------------------------------------------------------------------

# classificador = KNeighborsClassifier(
#     n_neighbors =  100,
#     weights     = 'uniform',
#     n_jobs      = -1   
#     )

classificador = DecisionTreeClassifier(
    criterion='entropy',
    max_features=64,
    max_depth=10
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

#------------------------------------------------------------------------------
#  Determinar experimentalmente o melhor valor de K
#------------------------------------------------------------------------------

for k in range(1,50,1):
    
    classificador = DecisionTreeClassifier(
        criterion='entropy',
        max_features=64,
        max_depth=k,
        min_impurity_decrease = 0.00001
        )

    classificador.fit(X_treino,y_treino)

    y_resposta = classificador.predict(X_teste)

    acuracia = 100*accuracy_score(y_teste,y_resposta)
    
    print ( 'k = %02d ,  acc = %4.1f' % ( k , acuracia ))


    