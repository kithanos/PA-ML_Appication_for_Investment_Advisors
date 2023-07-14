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

# import dash_bootstrap_components as dbc

# from flask_caching import Cache


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__
           # , external_stylesheets=external_stylesheets
           # , external_stylesheets=[dbc.themes.CERULEAN]
           )

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })

dataset_filename = 'dataset/Brazillian_Ticker_Codes_IBOV.csv'
df_Ticker_Names = pd.read_csv(dataset_filename, sep=";")
df_Ticker_Names['Ticker_CompanyNames'] = df_Ticker_Names['Ticker'] + " - " + df_Ticker_Names['Company_Name']

lista_ativos = []
for a in df_Ticker_Names['Ticker_CompanyNames'].drop_duplicates():
    lista_ativos.append(a)

app.layout = html.Div([
    
    html.Div(
    html.H2('Aplicação de ML para Assessores de Investimento'),
    style={'textAlign': 'center'}
    ),
    
    # html.Hr(),
    
    html.Div(
        '''
        Ferramenta para aplicação de modelos de ML em dados de ativos da bolsa brasileira.
        '''),
    
    # html.Hr(),
    html.Br(),
    
    html.Div([
        
    html.Label("Digite o código da ação/ativo:"),
    # dcc.Input(id="input-stock", type="text", value='MGLU3.SA', style={'width': '50vh'}),
    
    dcc.Dropdown(lista_ativos
                 ,id="input-stock"
                 , value='BBAS3 - Banco do Brasil S.A.'
                  , persistence = True
                 , style={'width': '85vh'}),
    
    html.Br(),
    html.Br(),
    
    html.Label("Digite desde que ano os dados serão considerados:"),
    dcc.Dropdown([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
                  , value = 2020
                  , id='input-year'
                  , style={'width': '50vh'}
                  , persistence = True),
    
    
    # dcc.Dropdown(id='input-year'
    #              # , value = 2019
    #              , style={'width': '50vh'}),
        
    ], 
    # style={'width': '30%', 'display': 'inline-block'}
    # style = {'padding': 10, 'flex': 1}
    ),
    
    # html.Hr(), 
        
    dcc.Graph(id='indicator-graphic')
    
#     ,style={'width': '48%', 'display': 'inline-block'}
])

        

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('input-stock', 'value'), 
    Input('input-year', 'value'),
    # Input('input-w', 'value')
    )

def update_fig(stock, year): 
    
    stock = df_Ticker_Names[df_Ticker_Names['Ticker_CompanyNames'] == stock]['Ticker'].item()
    
    df_0 = yf.download(f'{stock}.SA' 
                        # , start = f'{year}-01-01'
    #                  , end='2022-12-23'
                    )
    
    df_0 = df_0.loc[df_0.index > f'{year}-01-01']



    fig = go.Figure(data = go.Scatter(x = df_0.index, y = df_0['Adj Close']))

    fig.update_layout( 
                      title_text = f'Gráfico de Preços - {stock}'
    #                   , font_color = 'blue'
    #                   , title_font_color = 'black'
                      , xaxis_title = 'Data'
                      , yaxis_title = 'Preço [R$]'
                      , title_x = 0.5
    #                   , font = dict(size = 15, color = 'black')
                    )
    
    # fig.show()
    
    return fig

if __name__ == '__main__':
    app.run_server(
        # debug=True
           dev_tools_hot_reload=False
         )


    
    