#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados BOSTON (problema de regressão)
#==============================================================================

from scipy.stats import pearsonr

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Boston em um dataframe do pandas
#------------------------------------------------------------------------------
    
import pandas as pd
dataframe = pd.read_excel('../data/Boston.xlsx')
dataframe = dataframe.iloc[:,1:]

#------------------------------------------------------------------------------
#  Verificar os nomes das colunas disponíveis
#------------------------------------------------------------------------------

columnNames = dataframe.columns

print("Nomes das colunas:\n")
for col in columnNames:
    print(col)

#------------------------------------------------------------------------------
#  Imprimir o grafico de dispersao do alvo em relacao ao atributo 'LSTAT'
#------------------------------------------------------------------------------

variavel = 'CRIM'

dataframe.plot.scatter(x=variavel,y='target')
        
print ( "pearson coef = " , pearsonr(dataframe[variavel],dataframe['target']))

#------------------------------------------------------------------------------
#  Imprimir os gráficos de dispersão do alvo em relação a acda atributo
#------------------------------------------------------------------------------

for col in columnNames:
    dataframe.plot.scatter(x=col,y='target')

for col in columnNames:
    print ( "pearson(%7s)" % col , ") = %6.3f , %6.3e" % pearsonr(dataframe[col],dataframe['target']))

    
    
    
    