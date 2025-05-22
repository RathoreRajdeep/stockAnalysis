from dash import Dash, dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np



app = Dash(__name__, 
           external_stylesheets=[
               "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
               "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
           ],
           title="Stock Analysis Dashboard")

server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("./NSE-TATA.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("saved_lstm_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:]
valid['Predictions']=closing_price



df= pd.read_csv("./stock_data.csv")

app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1(
            "📊 Stock Price Analysis Dashboard",
            className="text-center my-4 fw-bold",
            style={'color': '#1a1a1a', 'fontSize': '2.5rem'}
        ),
        html.Div([
            html.P("📌 Facing difficulty? ", style={"display": "inline", 'color': '#555'}),
            html.A(
                "Click here to learn how to use ↓",
                href="#how-to-use",
                className="text-decoration-none fw-bold",
                style={'color': '#007bff'}
            )
        ], className="text-center mb-4")
    ], style={
        'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
    }),

    # Tabs Section
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(
            label='NSE-TATAGLOBAL Stock Data',
            style={'padding': '15px', 'fontWeight': 'bold', 'fontSize': '1.1rem'},
            selected_style={
                'backgroundColor': '#007bff',
                'color': 'white',
                'padding': '15px',
                'fontWeight': 'bold',
                'fontSize': '1.1rem'
            },
            children=[
                html.Div([
                    html.H2(
                        "Actual Closing Price",
                        style={'textAlign': 'center', 'color': '#333', 'marginTop': '20px'}
                    ),
                    dcc.Graph(
                        id="Actual Data",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=train.index,
                                    y=valid["Close"],
                                    mode='markers',
                                    marker={'color': '#007bff', 'size': 8},
                                    name='Actual'
                                )
                            ],
                            "layout": go.Layout(
                                title='',
                                xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
                                yaxis={'title': 'Closing Rate', 'gridcolor': '#e9ecef'},
                                plot_bgcolor='#ffffff',
                                paper_bgcolor='#ffffff',
                                font={'color': '#333'},
                                margin={'t': 20}
                            )
                        },
                        style={'borderRadius': '8px', 'overflow': 'hidden'}
                    ),
                    html.H2(
                        "LSTM Predicted Closing Price",
                        style={'textAlign': 'center', 'color': '#333', 'marginTop': '30px'}
                    ),
                    dcc.Graph(
                        id="Predicted Data",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=valid.index,
                                    y=valid["Predictions"],
                                    mode='markers',
                                    marker={'color': '#ff6b6b', 'size': 8},
                                    name='Predicted'
                                )
                            ],
                            "layout": go.Layout(
                                title='',
                                xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
                                yaxis={'title': 'Closing Rate', 'gridcolor': '#e9ecef'},
                                plot_bgcolor='#ffffff',
                                paper_bgcolor='#ffffff',
                                font={'color': '#333'},
                                margin={'t': 20}
                            )
                        },
                        style={'borderRadius': '8px', 'overflow': 'hidden'}
                    )
                ], className="container", style={'padding': '20px'})
            ]
        ),
        dcc.Tab(
            label='Tech Companies Stock Data',
            style={'padding': '15px', 'fontWeight': 'bold', 'fontSize': '1.1rem'},
            selected_style={
                'backgroundColor': '#007bff',
                'color': 'white',
                'padding': '15px',
                'fontWeight': 'bold',
                'fontSize': '1.1rem'
            },
            children=[
                html.Div([
                    html.H1(
                        "Stocks High vs Lows",
                        style={'textAlign': 'center', 'color': '#333', 'marginTop': '20px'}
                    ),
                    dcc.Dropdown(
                        id='my-dropdown',
                        options=[
                            {'label': 'Tesla', 'value': 'TSLA'},
                            {'label': 'Apple', 'value': 'AAPL'},
                            {'label': 'Facebook', 'value': 'FB'},
                            {'label': 'Microsoft', 'value': 'MSFT'}
                        ],
                        multi=True,
                        value=['FB'],
                        style={
                            "width": "60%",
                            "margin": "0 auto 20px auto",
                            "borderRadius": "5px",
                            "border": "1px solid #ced4da"
                        }
                    ),
                    dcc.Graph(id='highlow', style={'borderRadius': '8px', 'overflow': 'hidden'}),
                    html.H1(
                        "Stocks Market Volume",
                        style={'textAlign': 'center', 'color': '#333', 'marginTop': '30px'}
                    ),
                    dcc.Dropdown(
                        id='my-dropdown2',
                        options=[
                            {'label': 'Tesla', 'value': 'TSLA'},
                            {'label': 'Apple', 'value': 'AAPL'},
                            {'label': 'Facebook', 'value': 'FB'},
                            {'label': 'Microsoft', 'value': 'MSFT'}
                        ],
                        multi=True,
                        value=['FB'],
                        style={
                            "width": "60%",
                            "margin": "0 auto 20px auto",
                            "borderRadius": "5px",
                            "border": "1px solid #ced4da"
                        }
                    ),
                    dcc.Graph(id='volume', style={'borderRadius': '8px', 'overflow': 'hidden'})
                ], className="container", style={'padding': '20px'})
            ]
        )
    ], style={
        'backgroundColor': '#ffffff',
        'borderRadius': '8px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
        'margin': '20px 0'
    }),

    # How to Use Section
    html.Div(id="how-to-use", children=[
        html.Hr(style={'borderColor': '#e9ecef'}),
        html.Div([
            html.H5(
                "🧠 How to Use",
                className="text-primary mb-3",
                style={'fontWeight': 'bold'}
            ),
            html.Ul([
                html.Li("Navigate between tabs to explore different datasets.", style={'color': '#555'}),
                html.Li("In the first tab, view LSTM-based predictions for Tata Global.", style={'color': '#555'}),
                html.Li("In the second tab, select one or more stocks to view High/Low trends and Market Volume.", style={'color': '#555'}),
                html.Li("Use the date range slider below the charts to zoom into specific time periods.", style={'color': '#555'}),
            ], className="mb-4")
        ], className="container bg-light p-4 rounded shadow-sm")
    ], style={'margin': '20px 0'}),

    # Updated Contact Section
    html.Hr(style={'borderColor': '#e9ecef'}),
    html.Div([
        html.H5(
            "📬 Contact Information",
            className="text-primary mb-3",
            style={'fontWeight': 'bold', 'textAlign': 'center'}
        ),
        html.P(
            "For inquiries, feedback, or collaboration opportunities, feel free to reach out through the following channels:",
            className="text-center mb-4",
            style={'color': '#555'}
        ),
        html.Div([
            html.A(
                html.I(className="fab fa-linkedin fa-2x me-3"),
                href="https://www.linkedin.com/in/rajdeep-singh-rathore",
                target="_blank",
                title="Connect on LinkedIn",
                style={'color': '#0e76a8'}
            ),
            html.A(
                html.I(className="fab fa-github fa-2x me-3"),
                href="https://github.com/RathoreRajdeep",
                target="_blank",
                title="View GitHub Projects",
                style={'color': '#333'}
            ),
            html.A(
                html.I(className="fas fa-envelope fa-2x me-3"),
                href="mailto:rajdeeprathore92@gmail.com",
                title="Send an Email",
                style={'color': '#D44638'}
            ),
        ], className="text-center mb-4"),
        html.P(
            "Built with passion for data-driven insights | © 2025 Rajdeep",
            className="text-center mt-3 fw-bold",
            style={'color': '#555'}
        )
    ], style={
        'margin': '30px 0',
        'padding': '20px',
        'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
        'borderRadius': '10px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
    })
], style={
    'backgroundColor': '#f1f3f5',
    'minHeight': '100vh',
    'padding': '20px',
    'fontFamily': '"Roboto", sans-serif'
})




@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run(debug=True)