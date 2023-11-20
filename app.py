import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import psycopg2
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table
import numpy as np
from sklearn.linear_model import LinearRegression
from dash.dependencies import Input, Output


def get_connection():
    return psycopg2.connect('postgresql://gabybula19:cBF07iAOwhaZ@ep-crimson-scene-69952535.us-east-2.aws.neon.tech/finanzas')

def fetch_from_table(table, cols="*", conditions="", group_by="", order="", limit=""):
    conn = get_connection()
    query = f'select {cols} from {table}'
    if conditions != "":
        query += f' where {conditions}'
    if group_by != "":
        query += f' group by {group_by}'
    if order != "":
        query += f' order by {order}'
    if limit != "":
        query += f' limit {limit}'
    query += ';'
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()

def get_dataframe(col_names, **kwargs):
    results = fetch_from_table(**kwargs)
    df = pd.DataFrame(
        [[i for i in row] for row in results], 
        columns=col_names,
    ) 
    return df

def obtener_estado_de_resultados_df():
    return get_dataframe(
        col_names=["Año", "Ingresos", "Gastos"],
        table="ingresos i, egresos e",
        cols="i.FECHA, SUM(i.RECAUDO_ACUMULADO), SUM(e.PAGOS)",
        conditions="i.FECHA = e.FECHA",
        group_by="i.FECHA",
        order="i.FECHA desc",
    )

def obtener_balance_general_df():
    return get_dataframe(
        col_names=["Año", "Activos", "Pasivos"],
        table="ingresos i, egresos e",
        cols="i.FECHA, SUM(i.PTO_DEFINITIVO), SUM(e.APROPIACION_DEFINITIVA)",
        conditions="i.FECHA = e.FECHA",
        group_by="i.FECHA",
        order="i.FECHA desc",
    )

def obtener_flujo_de_efectivo_grafico():
    df = get_dataframe(
        col_names=["Año", "Entradas", "Salidas"],
        table="ingresos i, egresos e",
        cols="i.FECHA, SUM(i.RECAUDO_ACUMULADO), SUM(e.PAGOS)",
        conditions="i.FECHA = e.FECHA",
        group_by="i.FECHA",
        order="i.FECHA desc",
    )
    
    flujo_fig = px.line(
        df, 
        x='Año', 
        y=['Entradas', 'Salidas'], 
        title='Flujo de Efectivo a lo Largo del Tiempo'
    )
    flujo_fig.update_layout(title_font=dict(family='system-ui', size=20))
    return flujo_fig

def generate_table(dataframe):
    return dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in dataframe.columns],
    )
        
def calcular_promedio_movil(df, columna, ventana=3):
    return df[columna].rolling(window=ventana, min_periods=1).mean()

def realizar_proyeccion(df, columna, total_length):
    modelo = LinearRegression()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[columna].dropna().values
    modelo.fit(X[:len(y)], y) 

    # Realizar predicciones para el rango extendido
    X_future = np.arange(total_length).reshape(-1, 1)
    return modelo.predict(X_future)

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#### GRÁFICAS
## PESTAÑA 1

#1ra grafica pestana#1
df = get_dataframe(
    col_names=['Año','RECAUDO_ACUMULADO','PTO_INICIAL'],
    table='ingresos',
    cols='FECHA, sum(RECAUDO_ACUMULADO), sum(PTO_INICIAL)',
    group_by='FECHA',
    order='FECHA asc',
)
df['Ingresos'] = df['RECAUDO_ACUMULADO'] + df['PTO_INICIAL']

ingresos_fig = px.line(df, x='Año', y='Ingresos', title='Tendencia de Ingresos a Largo Plazo')
ingresos_fig.update_layout(title_font=dict(family='system-ui', size=20))
ingresos_fig.update_traces(line=dict(color='rgb(255, 99, 71)'))
ingresos_fig.update_xaxes(type='category')

ingresos_egresos_anuales = pd.DataFrame(columns=['Año', 'Monto', 'Tipo'])
ingresos_egresos_anuales['Año'] = df['Año'].copy()
ingresos_egresos_anuales['Monto'] = df['Ingresos'].copy()
ingresos_egresos_anuales['Tipo'] = 'Ingresos'

#2da grafica pestaña#1
df = get_dataframe(
    col_names=['Año', 'Egresos'],
    table='egresos',
    cols='FECHA, sum(PAGOS)',
    group_by='FECHA',
    order='FECHA asc',
)

egresos_fig = px.line(df, x='Año', y='Egresos', title='Tendencia de Egresos a Largo Plazo', line_shape='linear')
egresos_fig.update_layout(title_font=dict(family='system-ui', size=20))
egresos_fig.update_traces(line=dict(color='rgb(220, 20, 60)'))
egresos_fig.update_xaxes(type='category')

#3ra grafica pestaña#1
df['Tipo'] = 'Egresos'
df.rename(columns={'Egresos':'Monto'}, inplace=True)
ingresos_egresos_anuales = pd.concat([ingresos_egresos_anuales, df])

ing_eg_fig = px.bar(
    ingresos_egresos_anuales,
    x='Año',
    y='Monto',
    title='Evolución de Ingresos y Egresos por Año',
    color='Tipo',
    color_discrete_map={'Ingresos': '#FF6347', 'Egresos': '#DC143C '}, 
    barmode='group',
    orientation='v'
)
ing_eg_fig.update_xaxes(type='category')
ing_eg_fig.update_layout(title_font=dict(family='system-ui', size=20))

## PESTAÑA 2

#1ra grafica pestaña#2
df = get_dataframe(
    col_names=['Tipo de Rubro', 'Frecuencia'],
    table='ingresos',
    cols='upper(DESCRIPCION), count(*) as frec',
    group_by='DESCRIPCION',
    order='frec desc',
    limit=10,
)
df.dropna(inplace=True)
rubroin_fig = px.bar(
    df,  
    x='Tipo de Rubro', 
    y='Frecuencia',
    color='Tipo de Rubro',
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    title='Tipos de Rubro por Ingresos',
)
rubroin_fig.update_layout(title_font=dict(family='system-ui', size=20))

#2da grafica pestaña#2
df = get_dataframe(
    col_names=['Tipo de Rubro', 'Frecuencia'],
    table='egresos',
    cols='upper(DESCRIPCION), count(*) as frec',
    group_by='DESCRIPCION',
    order='frec desc',
    limit=10,
)
rubroeg_fig = px.bar(
    df,
    x='Tipo de Rubro', 
    y='Frecuencia',
    color='Tipo de Rubro',
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    title='Tipos de Rubro por Egresos',
)
rubroeg_fig.update_layout(title_font=dict(family='system-ui', size=20))

## PESTAÑA 3

#1ra grafica pestaña#3
df = get_dataframe(
    col_names=['DESCRIPCIÓN', 'RECAUDO ACUMULADO'],
    table='ingresos',
    cols='upper(DESCRIPCION), sum(RECAUDO_ACUMULADO) as rec',
    group_by='DESCRIPCION',
    order='rec desc',
    limit=5,
)
porcentajein_fig = px.pie(
    df, 
    values='RECAUDO ACUMULADO',
    names='DESCRIPCIÓN', 
    title=f'Proporción de Ingresos por Rubro (Top 5)',
    color_discrete_sequence=px.colors.sequential.Plasma_r
)
porcentajein_fig.update_layout(title_font=dict(family='system-ui', size=20))

#2da grafica pestaña#3
df = get_dataframe(
    col_names=['DESCRIPCIÓN', 'PAGOS'],
    table='egresos',
    cols='upper(DESCRIPCION), sum(PAGOS) as p',
    group_by='DESCRIPCION',
    order='p desc',
    limit=5,
)
porcentajeeg_fig = px.pie(
    df, 
    values='PAGOS', 
    names='DESCRIPCIÓN',
    title=f'Proporción de Egresos por Rubro (Top 5)',
    color_discrete_sequence=px.colors.sequential.Plasma_r
)
porcentajeeg_fig.update_layout(title_font=dict(family='system-ui', size=20))

#1ra grafica pestaña#4
df = get_dataframe(
    col_names=['Rubro', 'Frecuencia'],
    table='egresos',
    cols='upper(DESCRIPCION), count(*) as frec',
    group_by='DESCRIPCION',
    order='frec desc',
    limit=10,
)
gastos_fig = px.pie(
    df, 
    values='Frecuencia', 
    names='Rubro', 
    title='Distribución de Gastos por Rubro',
    color_discrete_sequence=px.colors.sequential.Plasma_r
)
gastos_fig.update_layout(title_font=dict(family='system-ui', size=20))


#### LAYOUT

ganancias_style = {'background-color': '#FF6347', 'padding': '10px', 'border-radius': '5px', 'color': 'white', 'font-family': 'Lucida Sans Unicode'}
perdidas_style = {'background-color': '#DC143C', 'padding': '10px', 'border-radius': '5px', 'color': 'white', 'font-family': 'Lucida Sans Unicode'}

#Define the layout of the first tab
tab_1_content = html.Div([
    dcc.Interval(id='update-data-interval', interval=60000, n_intervals=0),  # Interval to simulate data update
    
    dbc.Row([
        dbc.Col([
            
            html.H4("Ingresos (COP)", style={'text-align': 'center'}),
            html.H5('{:,}'.format(ingresos_egresos_anuales[ingresos_egresos_anuales['Tipo']=='Ingresos']['Monto'].sum()), style={'font-size': '24px', 'text-align': 'center'}),
        ], width=6, style=ganancias_style),
        
        dbc.Col([
            html.H4("Egresos (COP)", style={'text-align': 'center'}),
            html.H5('{:,}'.format(ingresos_egresos_anuales[ingresos_egresos_anuales['Tipo']=='Egresos']['Monto'].sum()), style={'font-size': '24px', 'text-align': 'center'}),
        ], width=6, style=perdidas_style),
    ]),
     dcc.Graph(
        id='primeragraf-pestaña-grafico',
        figure=ingresos_fig,  
    ),  
     dcc.Graph(
        id='segundagraf-pestaña-grafico',
        figure=egresos_fig, 
    ),
      dcc.Graph(
        id='terceragraf-pestaña-grafico',
        figure=ing_eg_fig, 
    ),
])

tab_2_content = html.Div([
    dcc.Graph(
        id='primeragraf-pestaña2-grafico',
        figure=rubroin_fig,  
    ),
    dcc.Graph(
        id='segundagraf-pestaña2-grafico',
        figure=rubroeg_fig,  
    ),
])

tab_3_content = html.Div([
    dcc.Graph(
        id='primeragraf-pestaña3-grafico',
        figure=porcentajein_fig,  
    ),
    dcc.Graph(
        id='segundagraf-pestaña3-grafico',
        figure=porcentajeeg_fig,  
    ),
])

tab_4_content = html.Div([
    dcc.Graph(
        id='primeragraf-pestaña4-grafico',
        figure=gastos_fig,  
    ),

])

tab_informes_content = html.Div([
    html.Div([
        html.H3("Informes Financieros"),
        html.H4("Estado de Resultados"),
        generate_table(obtener_estado_de_resultados_df()),
        html.H4("Balance General"),
        generate_table(obtener_balance_general_df()),
        html.H4("Flujo de Efectivo"),
        dcc.Graph(figure=obtener_flujo_de_efectivo_grafico())
    ], style={"width": "90%"})
], style={"display":"flex", "justify-content":"center"})

tab_analisis_proyecciones_content = html.Div([ 
    html.Div([
        html.H3("Análisis y Proyecciones Financieras"),
        html.Div([
            html.Label("Seleccione el número de años a visualizar (1-6):"),
            dcc.Input(id='num_years_input', type='number', min=1, max=6, value=1, step=1),
        ]),
        dcc.Graph(id='tendencia_proyeccion_graph'),
    ], style={"width": "90%"})
], style={"display":"flex", "justify-content":"center"})

@dash_app.callback(
    Output('tendencia_proyeccion_graph', 'figure'),
    [Input('num_years_input', 'value')]
)
def update_graph(num_years):

    if num_years is None or num_years < 1:
        num_years = 1

    # Datos actualizados
    df = obtener_estado_de_resultados_df()
    
    current_year = df["Año"].max()
    
    future_years = range(current_year + 1, current_year + num_years + 1)
    
    df_future = pd.DataFrame({
        'Año': future_years,
        'Ingresos': [np.nan] * num_years,  # Asume valores NaN para proyecciones futuras
        'Gastos': [np.nan] * num_years
    })

    df_extended = pd.concat([df, df_future], ignore_index=True)

    # Calcula las tendencias y proyecciones utilizando la columna 'Year'
    df_extended['Ingresos_Tendencia'] = calcular_promedio_movil(df_extended, 'Ingresos')
    df_extended['Egresos_Tendencia'] = calcular_promedio_movil(df_extended, 'Gastos')
    df_extended['Proyeccion_Ingresos'] = realizar_proyeccion(df, 'Ingresos', len(df_extended))
    df_extended['Proyeccion_Egresos'] = realizar_proyeccion(df, 'Gastos', len(df_extended))
    
    
    # Crea la figura de Plotly usando 'Año' como eje x
    figure = px.line(
        df_extended, 
        x='Año', 
        y=['Ingresos_Tendencia', 'Egresos_Tendencia', 'Proyeccion_Ingresos', 'Proyeccion_Egresos'], 
        title='Tendencia y Proyección de Ingresos y Gastos'
    )

    # Ajusta las etiquetas del eje x para mostrar solo los años (hice una batalla contra gpt porque nada de lo que ponia servia
    # y gpt tampoco pudo xD)
    figure.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=df_extended['Año'],
        ticktext=df_extended['Año'].astype(str)
    )

    return figure


title_style = {
    'font-family': 'Lucida Sans', 
    'font-size': '36px',  
    'color': 'black',
    'text-align': 'center'}

dash_app.layout = html.Div([
    html.H1('DASHBOARD', 
            style={'font-family': 'Lucida Sans', 
                   'font-size': '36px',  
                   'color': 'black',
                   'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Estado', children=tab_1_content),
        dcc.Tab(label='Tipos de Rubros', children=tab_2_content),
        dcc.Tab(label='Proporción de Rubros', children=tab_3_content),
        dcc.Tab(label='Distribución Gastos', children=tab_4_content),
        dcc.Tab(label='Informes Financieros', children=tab_informes_content),
        dcc.Tab(label='Análisis y Proyecciones', children=tab_analisis_proyecciones_content),
    ]),
])

if __name__ == '__main__':
    dash_app.run_server(debug=False)
