#Importação dos pacotes e bibliotecas

import numpy as np
import pandas as pd

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# import datetime

# import plotly.express as px
from dash import Dash, dcc, html, Input, Output
# from dash.dash import no_update
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import talib
# import ta
import plotly.figure_factory as ff

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import json
from sklearn.datasets import make_classification

import dash_bootstrap_components as dbc

# from flask_caching import Cache


#Definição dos estilos do dasboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__
            # , external_stylesheets=external_stylesheets
           # , external_stylesheets=[dbc.themes.CERULEAN]
           )

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })

#Leitura de um dataset contendo o código (ticker) e o nome do ativo
dataset_filename = 'dataset/Brazillian_Ticker_Codes_IBOV.csv'
df_Ticker_Names = pd.read_csv(dataset_filename, sep=";")
df_Ticker_Names['Ticker_CompanyNames'] = df_Ticker_Names['Ticker'] + " - " + df_Ticker_Names['Company_Name']

lista_ativos = []
for a in df_Ticker_Names['Ticker_CompanyNames'].drop_duplicates():
    lista_ativos.append(a)

# lista_variaveis = ['Volatilidade', 'RSI', 'MM50', 'MM200', 'Doji Star', 'Hagingman', 'ADX']

#Disposição dos elementos na tela
app.layout = html.Div([
    
    dcc.ConfirmDialog(
        id='confirm-year',
        message='Atenção, escolha uma data de treinamento superior a data de download dos dados!',
    ),
    
    dcc.ConfirmDialog(
        id='confirm-variables',
        message='Atenção, selecione ao menos uma variável!',
    ),
    
    dcc.ConfirmDialog(
        id='confirm-hiper',
        message='Atenção, algum parâmetro inválido para o modelo selecionado!',
    ),
    
    
    #Título
    html.Div(
    html.H2('Aplicação de ML para Assessores de Investimento'),
    style={'textAlign': 'center'}
    ),
    
    # html.Hr(),
    
    #Descrição
    html.Div(
        '''
        Ferramenta para aplicação de modelos de ML em dados de ativos da bolsa brasileira.
        '''),
    
    html.Hr(),
    # html.Br(),
    
    html.Div([
    
    #Seleção do ativo através de um menu dropdown
    html.Label("Selecione o código da ação/ativo:"),
    # dcc.Input(id="input-stock", type="text", value='MGLU3.SA', style={'width': '50vh'}),
    
    dcc.Dropdown(lista_ativos
                 ,id="input-stock"
                 , value='BBAS3 - Banco do Brasil S.A.'
                  , persistence = True
                 , style={'width': '85vh'}),
    
    # html.Br(),
    html.Br(),
    
    #Seleção da data (ano) em que o ativo será considerado
    html.Label("Selecione desde que ano os dados serão baixados:"),
    dcc.Dropdown([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
                  , value = 2015
                  , id='input-year'
                  , style={'width': '50vh'}
                  , persistence = True),
    
    html.Br(),
    
    #Seleção da data (ano) em que o ativo será considerado
    html.Label("Selecione até que ano os dados serão treinados:"),
    dcc.Dropdown([2020, 2021, 2022]
                  , value = 2020
                  , id='input-year-train'
                  , style={'width': '50vh'}
                  , persistence = True),
    
    
    # dcc.Dropdown(id='input-year'
    #              # , value = 2019
    #              , style={'width': '50vh'}),
        
    ]
        , style={'display': 'inline-block', 'vertical-align': 'top',}
        ),
    
    
    html.Div([  
    
    html.Label("Selecione as variáveis do modelo:"),
    # dcc.Dropdown(lista_variaveis
    #              ,id="input-variables"
    #              , value='RSI'
    #              , persistence = True
    #              , multi = True
    #              , style={'width': '90vh'}),
    
    dcc.Dropdown(
        options = [
            {'label': 'Volatilidade', 'value': 'Volatilidade'},
            {'label': 'Relative Strength Index', 'value': 'RSI'},
            {'label': 'Simple Moving Average 50 days', 'value': 'MM50'},
            {'label': 'Simple Moving Average 200 days', 'value': 'MM200'},
            {'label': 'Doji Star', 'value': 'Doji Star'},
            {'label': 'Hagingman', 'value': 'Hagingman'},
            {'label': 'Average Directional Movement Index', 'value': 'ADX'},
            
            {'label': 'Kaufman Adaptive Moving Average 50 days', 'value': 'KAMA50'},
            {'label': 'Kaufman Adaptive Moving Average 200 days', 'value': 'KAMA200'},
            {'label': 'Weighted Moving Average 50 days', 'value': 'WMA50'},
            {'label': 'Weighted Moving Average 200 days', 'value': 'WMA50'},
            
            {'label': 'Average Directional Movement Index Rating', 'value': 'ADXR'},
            {'label': 'Rate of change', 'value': 'ROC'},
            {'label': 'Hilbert Transform - Dominant Cycle Period', 'value': 'HT_DCPERIOD'},
            {'label': 'Three Inside Up/Down', 'value': 'CDL3INSIDE'},
            {'label': 'Rickshaw Man', 'value': 'CDLRICKSHAWMAN'}
            
            ]
                 , id="input-variables"
                 , value='RSI'
                 , persistence = True
                 , multi = True
                 , style={'width': '90vh'}),
    
    html.Br(),
    html.Label("Selecione o modelo:"),
    dcc.Dropdown(
            options = [
                {'label': 'Logistic Regression', 'value': 'lr'},
                {'label': 'Decision Tree', 'value': 'dt'},
                {'label': 'Random Forest', 'value': 'rf'},
                {'label': 'Rede Neural', 'value': 'mlp'}
                ],
            value='lr',
            id="input-model"
        ),
    
    html.Br(),
    html.Label("Hiperparâmetros:"),
    html.Br(),
    # html.Button('Atualizar', id='btn-hiper', n_clicks=0,
    #             style = {'background-color': '#6671FA'})
    dbc.Button("Clique para atualizar", id='btn-hiper', color="primary", className="me-1"),
    
    
    
    ]
        , style={'display': 'inline-block', 'vertical-align': 'top', "margin-left": "100px"}
        ),
    
    html.Hr(),
        
    html.Center(
    dcc.Graph(id='indicator-graphic')
    ),
    
    
    html.Center(
    dcc.Graph(id='indicator-metrics')
      # ,style = {'textAlign': 'right'}
     )
    
#     ,style={'width': '48%', 'display': 'inline-block'}
])


@app.callback(
    Output('confirm-year', 'displayed'),
    Input('input-year', 'value'),
    Input('input-year-train', 'value')
    )
              
def display_confirm(year, end_train):
    if end_train <= year:
        return True
    return False

@app.callback(
    Output('confirm-variables', 'displayed'),
    Input('input-variables', 'value')
    )
              
def display_confirm_var(list_variables):
    if len(list_variables) == 0:
        return True
    return False

@app.callback(
    Output('confirm-hiper', 'displayed'),
    Input('input-model', 'value'),
    Input('btn-hiper', 'n_clicks')
    )
              
def display_confirm_hiper(model_choiced, btn_h):
    f = open('parameters.json')
    data = json.load(f)
    X, y = make_classification(n_samples=5, n_features=4,
                        n_informative=2, n_redundant=0,
                        random_state=0, shuffle=False)
    
    if model_choiced == 'lr':
        try:
            model = LogisticRegression(
                C = data['models']['logistic_regression'][0]['C']
                ,class_weight = data['models']['logistic_regression'][0]['class_weight']
                ,dual = data['models']['logistic_regression'][0]['dual']
                ,fit_intercept = data['models']['logistic_regression'][0]['fit_intercept']
                ,intercept_scaling = data['models']['logistic_regression'][0]['intercept_scaling']
                ,l1_ratio = data['models']['logistic_regression'][0]['l1_ratio']
                ,max_iter = data['models']['logistic_regression'][0]['max_iter']
                ,multi_class = data['models']['logistic_regression'][0]['multi_class']
                ,n_jobs = data['models']['logistic_regression'][0]['n_jobs']
                ,penalty = data['models']['logistic_regression'][0]['penalty']
                ,random_state = data['models']['logistic_regression'][0]['random_state']
                ,solver = data['models']['logistic_regression'][0]['solver']
                ,tol = data['models']['logistic_regression'][0]['tol']
                ,verbose = data['models']['logistic_regression'][0]['verbose']
                ,warm_start = data['models']['logistic_regression'][0]['warm_start']
            ).fit(X, y)
            return False
        except:
            return True
            
    elif model_choiced == 'dt':
        try:
            model = DecisionTreeClassifier(
                ccp_alpha = data['models']['decision_tree'][0]['ccp_alpha']
                ,class_weight = data['models']['decision_tree'][0]['class_weight']
                ,criterion = data['models']['decision_tree'][0]['criterion']
                ,max_depth = data['models']['decision_tree'][0]['max_depth']
                ,max_features = data['models']['decision_tree'][0]['max_features']
                ,max_leaf_nodes = data['models']['decision_tree'][0]['max_leaf_nodes']
                ,min_impurity_decrease = data['models']['decision_tree'][0]['min_impurity_decrease']
                ,min_samples_leaf = data['models']['decision_tree'][0]['min_samples_leaf']
                ,min_samples_split = data['models']['decision_tree'][0]['min_samples_split']
                ,min_weight_fraction_leaf = data['models']['decision_tree'][0]['min_weight_fraction_leaf']
                ,random_state = data['models']['decision_tree'][0]['random_state']
                ,splitter = data['models']['decision_tree'][0]['splitter']
            ).fit(X, y)
            return False
        except:
            return True
    elif model_choiced == 'rf':
        try:
            model = RandomForestClassifier(
                bootstrap = data['models']['random_forest'][0]['bootstrap']
                ,ccp_alpha = data['models']['random_forest'][0]['ccp_alpha']
                ,class_weight = data['models']['random_forest'][0]['class_weight']
                ,criterion = data['models']['random_forest'][0]['criterion']
                ,max_depth = data['models']['random_forest'][0]['max_depth']
                ,max_features = data['models']['random_forest'][0]['max_features']
                ,max_leaf_nodes = data['models']['random_forest'][0]['max_leaf_nodes']
                ,max_samples = data['models']['random_forest'][0]['max_samples']
                ,min_impurity_decrease = data['models']['random_forest'][0]['min_impurity_decrease']
                ,min_samples_leaf = data['models']['random_forest'][0]['min_samples_leaf']
                ,min_samples_split = data['models']['random_forest'][0]['min_samples_split']
                ,min_weight_fraction_leaf = data['models']['random_forest'][0]['min_weight_fraction_leaf']
                ,n_estimators = data['models']['random_forest'][0]['n_estimators']
                ,n_jobs = data['models']['random_forest'][0]['n_jobs']
                ,oob_score = data['models']['random_forest'][0]['oob_score']
                ,random_state = data['models']['random_forest'][0]['random_state']
                ,verbose = data['models']['random_forest'][0]['verbose']
                ,warm_start = data['models']['random_forest'][0]['warm_start']
            ).fit(X, y)
            return False
        except:
            return True
    elif model_choiced == 'mlp':
        try:
            model = MLPClassifier(
                alpha = data['models']['mlp_classifier'][0]['alpha']
                ,batch_size = data['models']['mlp_classifier'][0]['batch_size']
                ,beta_1 = data['models']['mlp_classifier'][0]['beta_1']
                ,beta_2 = data['models']['mlp_classifier'][0]['beta_2']
                ,early_stopping = data['models']['mlp_classifier'][0]['early_stopping']
                ,epsilon = data['models']['mlp_classifier'][0]['epsilon']
                ,hidden_layer_sizes = tuple(data['models']['mlp_classifier'][0]['hidden_layer_sizes'])
                ,learning_rate = data['models']['mlp_classifier'][0]['learning_rate']
                ,learning_rate_init = data['models']['mlp_classifier'][0]['learning_rate_init']
                ,max_fun = data['models']['mlp_classifier'][0]['max_fun']
                ,max_iter = data['models']['mlp_classifier'][0]['max_iter']
                ,momentum = data['models']['mlp_classifier'][0]['momentum']
                ,n_iter_no_change = data['models']['mlp_classifier'][0]['n_iter_no_change']
                ,nesterovs_momentum = data['models']['mlp_classifier'][0]['nesterovs_momentum']
                ,power_t = data['models']['mlp_classifier'][0]['power_t']
                ,random_state = data['models']['mlp_classifier'][0]['random_state']
                ,shuffle = data['models']['mlp_classifier'][0]['shuffle']
                ,solver = data['models']['mlp_classifier'][0]['solver']
                ,tol = data['models']['mlp_classifier'][0]['tol']
                ,validation_fraction = data['models']['mlp_classifier'][0]['validation_fraction']
                ,verbose = data['models']['mlp_classifier'][0]['verbose']
                ,warm_start = data['models']['mlp_classifier'][0]['warm_start']
            ).fit(X, y)
            return False
        except:
            return True
    
#Callback que relaciona os elementos de entrada, com a saída (gráfico)
@app.callback(
    Output('indicator-graphic', 'figure'),
    Output('indicator-metrics', 'figure'),
    Input('input-stock', 'value'), 
    Input('input-year', 'value'),
    Input('input-variables', 'value'),
    Input('input-model', 'value'),
    Input('input-year-train', 'value'),
    Input('btn-hiper', 'n_clicks')
    )

def update_fig(stock, year, list_variables, model_choiced, end_train, btn_h): 
    
    #Baixa os dados do Ibovespa
    stock_complete = stock
    stock = df_Ticker_Names[df_Ticker_Names['Ticker_CompanyNames'] == stock]['Ticker'].item()
    
    df_0 = yf.download(f'{stock}.SA' 
                        # , start = f'{year}-01-01'
    #                  , end='2022-12-23'
                    )
    
    #Filtra os dados de acordo com o ano selecionado pelo usuário
    df_0 = df_0.loc[df_0.index > f'{year}-12-31']
    
    df1 = df_0.copy()
    df1["Retornos"] = df1["Adj Close"].pct_change(1)
    df1.dropna(axis = 0, inplace = True) 
    df1["Alvo"] = np.where(df1["Retornos"].shift(-1) > 0, 1, 0)
    
    df1["Volatilidade"] = df1["Retornos"].rolling(20).std()*np.sqrt(252)*100
    # df1['RSI'] = ta.momentum.RSIIndicator(df1['Adj Close'], window=14).rsi()
    df1['RSI'] = talib.RSI(df1["Adj Close"], timeperiod= 14)
    # df1['MM50'] = ta.trend.SMAIndicator(df1['Adj Close'], window=50).sma_indicator()
    df1['MM50'] = talib.SMA(df1["Adj Close"], timeperiod= 50)
    # df1['MM200'] = ta.trend.SMAIndicator(df1['Adj Close'], window=200).sma_indicator()
    df1['MM200'] = talib.SMA(df1["Adj Close"], timeperiod= 200)
    df1["Doji Star"] = talib.CDLDOJISTAR(df1["Open"], df1["High"], df1["Low"], df1["Close"])
    df1["Hagingman"] = talib.CDLHANGINGMAN(df1["Open"], df1["High"], df1["Low"], df1["Close"])
    df1["Doji Star"] = np.where(df1["Doji Star"] == 100, 1
                                , np.where(df1["Doji Star"] == -100, -1, 0))
    df1["Hagingman"] = np.where(df1["Hagingman"] == 100, 1
                                , np.where(df1["Hagingman"] == -100, -1, 0))
    df1['ADX'] = talib.ADX(df1['High'], df1['Low'], df1['Close'], timeperiod=14)
    
    
    df1['KAMA50'] = talib.KAMA(df1["Adj Close"], timeperiod= 50)
    df1['KAMA200'] = talib.KAMA(df1["Adj Close"], timeperiod= 200)
    df1['WMA50'] = talib.WMA(df1["Adj Close"], timeperiod= 50)
    df1['WMA200'] = talib.WMA(df1["Adj Close"], timeperiod= 200)
    
    df1['ADXR'] = talib.ADXR(df1['High'], df1['Low'], df1['Close'], timeperiod=14)
    df1['ROC'] = talib.ROC(df1["Adj Close"], timeperiod= 14)
    df1['HT_DCPERIOD'] = talib.HT_DCPERIOD(df1["Adj Close"])
    
    df1["CDL3INSIDE"] = talib.CDL3INSIDE(df1["Open"], df1["High"], df1["Low"], df1["Close"])
    
    df1["CDL3INSIDE"] = np.where(df1["CDL3INSIDE"] == 100, 1
                                , np.where(df1["CDL3INSIDE"] == -100, -1, 0))
    
    df1["CDLRICKSHAWMAN"] = talib.CDLRICKSHAWMAN(df1["Open"], df1["High"], df1["Low"], df1["Close"])
    
    df1["CDLRICKSHAWMAN"] = np.where(df1["CDLRICKSHAWMAN"] == 100, 1
                                , np.where(df1["CDLRICKSHAWMAN"] == -100, -1, 0))
    
    
    df1.dropna(axis = 0, inplace = True) 
    
    list_variables.append('Alvo')
    df2 = df1[list_variables].copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df2.drop('Alvo', axis=1).values)
    df3 = pd.DataFrame(scaled_features, index=df2.drop('Alvo', axis=1).index, columns=df2.drop('Alvo', axis=1).columns)
    df4 = df3.merge(df2[['Alvo']], how='left', left_on=df3.index, right_on=df2[['Alvo']].index).set_index('key_0')
    
    if end_train <= year:
        end_train = year + 1
    
    # end_train = '2022'
    
    df_train = df4.loc[df4.index <= f'{end_train}-01-01']
    df_test = df4.loc[df4.index > f'{end_train}-01-01']
    
    x_train = df_train.drop(["Alvo"], axis = 1)
    y_train = df_train["Alvo"]
    
    x_test = df_test.drop(["Alvo"], axis = 1)
    y_test = df_test["Alvo"]
    
    f = open('parameters.json')
    data = json.load(f)

    
    # model = DecisionTreeClassifier().fit(x_train, y_train)

    if model_choiced == 'lr':
        model = LogisticRegression(
            C = data['models']['logistic_regression'][0]['C']
            ,class_weight = data['models']['logistic_regression'][0]['class_weight']
            ,dual = data['models']['logistic_regression'][0]['dual']
            ,fit_intercept = data['models']['logistic_regression'][0]['fit_intercept']
            ,intercept_scaling = data['models']['logistic_regression'][0]['intercept_scaling']
            ,l1_ratio = data['models']['logistic_regression'][0]['l1_ratio']
            ,max_iter = data['models']['logistic_regression'][0]['max_iter']
            ,multi_class = data['models']['logistic_regression'][0]['multi_class']
            ,n_jobs = data['models']['logistic_regression'][0]['n_jobs']
            ,penalty = data['models']['logistic_regression'][0]['penalty']
            ,random_state = data['models']['logistic_regression'][0]['random_state']
            ,solver = data['models']['logistic_regression'][0]['solver']
            ,tol = data['models']['logistic_regression'][0]['tol']
            ,verbose = data['models']['logistic_regression'][0]['verbose']
            ,warm_start = data['models']['logistic_regression'][0]['warm_start']
        ).fit(x_train, y_train)
    elif model_choiced == 'dt':
        model = DecisionTreeClassifier(
            ccp_alpha = data['models']['decision_tree'][0]['ccp_alpha']
            ,class_weight = data['models']['decision_tree'][0]['class_weight']
            ,criterion = data['models']['decision_tree'][0]['criterion']
            ,max_depth = data['models']['decision_tree'][0]['max_depth']
            ,max_features = data['models']['decision_tree'][0]['max_features']
            ,max_leaf_nodes = data['models']['decision_tree'][0]['max_leaf_nodes']
            ,min_impurity_decrease = data['models']['decision_tree'][0]['min_impurity_decrease']
            ,min_samples_leaf = data['models']['decision_tree'][0]['min_samples_leaf']
            ,min_samples_split = data['models']['decision_tree'][0]['min_samples_split']
            ,min_weight_fraction_leaf = data['models']['decision_tree'][0]['min_weight_fraction_leaf']
            ,random_state = data['models']['decision_tree'][0]['random_state']
            ,splitter = data['models']['decision_tree'][0]['splitter']
        ).fit(x_train, y_train)
    elif model_choiced == 'rf':
        model = RandomForestClassifier(
            bootstrap = data['models']['random_forest'][0]['bootstrap']
            ,ccp_alpha = data['models']['random_forest'][0]['ccp_alpha']
            ,class_weight = data['models']['random_forest'][0]['class_weight']
            ,criterion = data['models']['random_forest'][0]['criterion']
            ,max_depth = data['models']['random_forest'][0]['max_depth']
            ,max_features = data['models']['random_forest'][0]['max_features']
            ,max_leaf_nodes = data['models']['random_forest'][0]['max_leaf_nodes']
            ,max_samples = data['models']['random_forest'][0]['max_samples']
            ,min_impurity_decrease = data['models']['random_forest'][0]['min_impurity_decrease']
            ,min_samples_leaf = data['models']['random_forest'][0]['min_samples_leaf']
            ,min_samples_split = data['models']['random_forest'][0]['min_samples_split']
            ,min_weight_fraction_leaf = data['models']['random_forest'][0]['min_weight_fraction_leaf']
            ,n_estimators = data['models']['random_forest'][0]['n_estimators']
            ,n_jobs = data['models']['random_forest'][0]['n_jobs']
            ,oob_score = data['models']['random_forest'][0]['oob_score']
            ,random_state = data['models']['random_forest'][0]['random_state']
            ,verbose = data['models']['random_forest'][0]['verbose']
            ,warm_start = data['models']['random_forest'][0]['warm_start']
        ).fit(x_train, y_train)
    elif model_choiced == 'mlp':
        model = MLPClassifier(
            alpha = data['models']['mlp_classifier'][0]['alpha']
            ,batch_size = data['models']['mlp_classifier'][0]['batch_size']
            ,beta_1 = data['models']['mlp_classifier'][0]['beta_1']
            ,beta_2 = data['models']['mlp_classifier'][0]['beta_2']
            ,early_stopping = data['models']['mlp_classifier'][0]['early_stopping']
            ,epsilon = data['models']['mlp_classifier'][0]['epsilon']
            ,hidden_layer_sizes = tuple(data['models']['mlp_classifier'][0]['hidden_layer_sizes'])
            ,learning_rate = data['models']['mlp_classifier'][0]['learning_rate']
            ,learning_rate_init = data['models']['mlp_classifier'][0]['learning_rate_init']
            ,max_fun = data['models']['mlp_classifier'][0]['max_fun']
            ,max_iter = data['models']['mlp_classifier'][0]['max_iter']
            ,momentum = data['models']['mlp_classifier'][0]['momentum']
            ,n_iter_no_change = data['models']['mlp_classifier'][0]['n_iter_no_change']
            ,nesterovs_momentum = data['models']['mlp_classifier'][0]['nesterovs_momentum']
            ,power_t = data['models']['mlp_classifier'][0]['power_t']
            ,random_state = data['models']['mlp_classifier'][0]['random_state']
            ,shuffle = data['models']['mlp_classifier'][0]['shuffle']
            ,solver = data['models']['mlp_classifier'][0]['solver']
            ,tol = data['models']['mlp_classifier'][0]['tol']
            ,validation_fraction = data['models']['mlp_classifier'][0]['validation_fraction']
            ,verbose = data['models']['mlp_classifier'][0]['verbose']
            ,warm_start = data['models']['mlp_classifier'][0]['warm_start']
        ).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    recall = np.round(recall_score(y_test, y_pred), 2)
    precision = np.round(precision_score(y_test, y_pred), 2)
    f1 = np.round(f1_score(y_test, y_pred), 2)
    acc = np.round(accuracy_score(y_test, y_pred), 2)
    
    if y_pred[-1] > 0:
        status = 'Compre!'
    else:
        status = 'Não Compre!'
    
    # np.round(model.predict_proba(x_test.tail(1))[0][1], 2)
    
    
    result_mat = [
    ["Recall", "Precision", 'f1-score', 'Acuraccy', 'Recomendação'],
    [recall, precision, f1, acc, status]
    ]

    swt_table = ff.create_table(result_mat)
    swt_table['data'][0].colorscale=[[0, '#2fa4e7'],[1, '#ffffff']]
    swt_table['layout']['height'] = 100
    swt_table['layout']['width'] = 930
    swt_table['layout']['margin']['t'] = 10
    swt_table['layout']['margin']['b'] = 10
        

    #Plota a figura
    fig = go.Figure(data = go.Scatter(x = df_0.index, y = df_0['Adj Close']))

    fig.update_layout( 
                      title_text = f'Gráfico de Preços - {stock_complete}'
    #                   , font_color = 'blue'
    #                   , title_font_color = 'black'
                      , xaxis_title = 'Data'
                      , yaxis_title = 'Preço [R$]'
                      , title_x = 0.5
    #                   , font = dict(size = 15, color = 'black')
                    )
    
    # fig.show()
    fig.update_traces(line_color='#2fa4e7')
    
    return fig, swt_table

if __name__ == '__main__':
    app.run_server(
        # debug=True
           dev_tools_hot_reload=False
         )


    
    