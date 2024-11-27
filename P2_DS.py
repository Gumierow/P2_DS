# Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind
import streamlit as st

# Função principal do Streamlit
def main():
    st.title("Análise de Acidentes Fatais na Austrália (1989-2021)")
    st.sidebar.title("Configurações de Análise")

    # URL do arquivo raw no GitHub
    url = 'https://raw.githubusercontent.com/Gumierow/P2_DS/refs/heads/main/Crash_Data.csv'
    
    # Carregar o dataset
    data = pd.read_csv(url)
    st.write("### Pré-visualização do Dataset")
    st.dataframe(data.head())

    # Limpeza de dados
    data_cleaned = clean_data(data)
    st.write("### Dados Limpos")
    st.dataframe(data_cleaned.head())

    # Estatísticas descritivas
    st.header("Estatísticas Descritivas")
    descriptive_stats(data_cleaned)

    # Análise Temporal
    st.header("Análise Temporal")
    temporal_analysis(data_cleaned)

    # Estatística Inferencial
    st.header("Estatística Inferencial")
    inferential_statistics(data_cleaned)

    # Análise específica: Acidentes entre jovens homens e mulheres no volante
    st.header("Análise por Gênero (Jovens)")
    gender_analysis(data_cleaned)

# Função para limpar os dados
def clean_data(data):
    data_cleaned = data.copy()
    # Remover colunas irrelevantes
    columns_to_drop = [
        'Speed Limit', 'National Remoteness Areas', 'SA4 Name 2016',
        'National Road Type', 'Bus Involvement', 'Heavy Rigid Truck Involvement'
    ]
    data_cleaned.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # Conversões de tipos
    data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M', errors='coerce')
    data_cleaned['Month'] = pd.Categorical(data_cleaned['Month'], categories=range(1, 13))
    data_cleaned['Year'] = data_cleaned['Year'].astype(int)
    data_cleaned['Christmas Period'] = data_cleaned['Christmas Period'].map({'Yes': 1, 'No': 0})
    data_cleaned['Easter Period'] = data_cleaned['Easter Period'].map({'Yes': 1, 'No': 0})
    return data_cleaned

# Função para estatísticas descritivas
def descriptive_stats(data):
    st.write("### Medidas Resumo")
    stats = data.describe(include='all').T
    st.dataframe(stats)
    
    # Histogramas de idade usando Regra de Sturges
    st.write("### Distribuição de Idade (Histograma)")
    bins = int(1 + 3.322 * np.log10(len(data)))
    fig, ax = plt.subplots()
    ax.hist(data['Age'], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribuição de Idade')
    ax.set_xlabel('Idade')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

# Função para análise temporal
def temporal_analysis(data):
    # Frequência por mês
    monthly_accidents = data['Month'].value_counts().sort_index()
    st.write("### Frequência de Acidentes por Mês")
    fig = px.bar(x=monthly_accidents.index, y=monthly_accidents.values, labels={'x': 'Mês', 'y': 'Acidentes'}, title="Frequência Mensal de Acidentes")
    st.plotly_chart(fig)

    # Frequência por dia da semana
    weekly_accidents = data['Dayweek'].value_counts()
    st.write("### Frequência de Acidentes por Dia da Semana")
    fig = px.bar(x=weekly_accidents.index, y=weekly_accidents.values, labels={'x': 'Dia da Semana', 'y': 'Acidentes'}, title="Acidentes por Dia da Semana")
    st.plotly_chart(fig)

    # Frequência por períodos do dia
    time_of_day_accidents = data['Time of day'].value_counts()
    st.write("### Acidentes por Período do Dia")
    fig = px.bar(x=time_of_day_accidents.index, y=time_of_day_accidents.values, labels={'x': 'Período do Dia', 'y': 'Acidentes'}, title="Acidentes por Período do Dia")
    st.plotly_chart(fig)

    # Tendência Anual
    yearly_accidents = data['Year'].value_counts().sort_index()
    st.write("### Tendência Anual de Acidentes")
    fig = px.line(x=yearly_accidents.index, y=yearly_accidents.values, labels={'x': 'Ano', 'y': 'Acidentes'}, title="Tendência Anual de Acidentes")
    st.plotly_chart(fig)

# Função para estatística inferencial
def inferential_statistics(data):
    # Teste T para feriados vs não feriados
    st.write("### Teste T de Student")
    christmas_accidents = data[data['Christmas Period'] == 1]['Age']
    non_christmas_accidents = data[data['Christmas Period'] == 0]['Age']
    t_stat, p_val = ttest_ind(christmas_accidents, non_christmas_accidents, nan_policy='omit')
    st.write(f"Estatística t: {t_stat:.3f}, Valor p: {p_val:.3f}")
    if p_val < 0.05:
        st.success("Diferença estatisticamente significativa entre feriados de Natal e períodos normais.")
    else:
        st.info("Nenhuma diferença estatisticamente significativa encontrada.")

# Função para análise por gênero (jovens)
def gender_analysis(data):
    # Filtrar jovens (idade entre 17 e 25) e registros com Gender não nulo
    data_young = data[(data['Age Group'] == '17_to_25') & (data['Gender'].notnull())]
    if data_young.empty:
        st.warning("Não há dados suficientes para análise de jovens por gênero.")
        return
    
    # Contagem por gênero
    gender_counts = data_young['Gender'].value_counts()
    st.write("### Distribuição por Gênero entre Jovens")
    fig = px.bar(x=gender_counts.index, y=gender_counts.values, labels={'x': 'Gênero', 'y': 'Número de Acidentes'}, title="Acidentes por Gênero (17-25 anos)")
    st.plotly_chart(fig)

    # Taxa de acidentes por gênero
    total_accidents = len(data_young)
    gender_rates = (gender_counts / total_accidents) * 100
    st.write("### Taxa de Acidentes por Gênero")
    st.dataframe(gender_rates)

# Executar o aplicativo
if __name__ == '__main__':
    main()
