#==============================================================================
#  Classificacao KNN no Conjunto de Dados IRIS
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Iris em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../data/iris.xlsx')

#------------------------------------------------------------------------------
#  Separar em dataframes distintos os atributos e o alvo 
#    - os atributos são todas as colunas menos a última
#    - o alvo é a última coluna 
#------------------------------------------------------------------------------

attributes = dataframe.iloc[:,:-1]
target     = dataframe.iloc[:,-1]

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = attributes.values
y = target.values

