#------------------------------------------------------------------------------
#  Importar o conjunto de dados Digits em um dataframe do pandas,
#------------------------------------------------------------------------------
import pandas as pd
dataframe = pd.read_excel('../data/Digits.xlsx')
#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo,
#------------------------------------------------------------------------------
X = dataframe.iloc[:,1:-1].values
y = dataframe.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=500,
    random_state=42
    )
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(
        n_estimators = 150,
        criterion = 'entropy',
        max_depth = 11,
        min_samples_split = 2,
        max_features = 'auto',
        min_impurity_decrease = 0,
        bootstrap = False,
        oob_score = False,
        random_state = 42
        )

classificador.fit(X_treino,y_treino)
y_resposta = classificador.predict(X_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

accuracia=100*accuracy_score(y_teste,y_resposta)
print ('accuracy = %4.1f' % accuracia)
#------------------------------------------------------------------------------
#  Mostrar a matriz de confusão e a medida de acuracia,
#------------------------------------------------------------------------------
print ('Matriz de Confusao:')
print (confusion_matrix(y_teste,y_resposta))
print ( 'Acuracia:',100*accuracy_score(y_teste,y_resposta), '%')
