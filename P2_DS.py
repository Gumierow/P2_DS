import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind
import streamlit as st

# Função principal do Streamlit
def main():
    st.title("Análise de dados sobre acidentes fatais na Austrália entre 1989 e 2021.")
    st.subheader("Esta análise se baseia em dados de acidentes fatais ocorridos na Austrália entre 1989 e 2021. O objetivo é explorar tendências e padrões, com foco nas variáveis mais significativas relacionadas a idade, gênero, período do dia e feriados.")

    # Carregar o dataset do GitHub
    dataset_url = "https://raw.githubusercontent.com/Gumierow/P2_DS/refs/heads/main/Crash_Data.csv"
    data = pd.read_csv(dataset_url)
    
    # Limpeza de dados
    data_cleaned = clean_data(data)

    # Barra lateral de navegação
    st.sidebar.title("Configurações de Análise")
    options = ["Introdução", "Estatística Descritiva", "Estatística Inferencial", "Conclusão"]
    selected_option = st.sidebar.radio("Escolha a análise", options)

    # Seção de Introdução
    if selected_option == "Introdução":
        st.header("Introdução")
        st.subheader("Pré-visualização e Limpeza dos dados")
        st.write("As colunas presentes no dataset são: 'Year', 'Month', 'Day', 'Dayweek', 'Time', 'Age', 'Gender', 'Crash Severity', 'Crash Type', 'Accident Description', 'Time of day', 'Christmas Period', 'Easter Period'.")
        st.dataframe(data.head())
        st.write(f"As colunas removidas durante a limpeza foram: 'Speed Limit', 'National Remoteness Areas', 'SA4 Name 2016', 'National Road Type', 'Bus Involvement', 'Heavy Rigid Truck Involvement'.")
        st.dataframe(data_cleaned.head())

    # Seção de Estatística Descritiva
    elif selected_option == "Estatística Descritiva":
        st.header("Estatísticas Descritivas")
        st.subheader("Total de acidentes por ano")
        total_accidents_per_year(data_cleaned)

        st.subheader("Distribuição de frequência dos acidentes por mês de 2010 a 2021")
        monthly_accidents_distribution(data_cleaned)

        st.subheader("Distribuição de frequência dos acidentes por idade de 2010 a 2021")
        age_distribution(data_cleaned)

        st.subheader("Comparativo entre jovens homens e jovens mulheres entre 2010 e 2021")
        gender_comparison(data_cleaned)

        st.subheader("Boxplot de Gênero e Idade dos Envolvidos")
        gender_age_boxplot(data_cleaned)

        st.subheader("Acidentes por dia da semana e período")
        accidents_by_day_and_period(data_cleaned)

    # Seção de Estatística Inferencial
    elif selected_option == "Estatística Inferencial":
        st.header("Estatística Inferencial")
        st.subheader("A influência de datas comemorativas no número de acidentes")
        inferential_statistics(data_cleaned)

    # Seção de Conclusão
    elif selected_option == "Conclusão":
        st.header("Conclusão da Análise")
        st.write("""
            Concluímos que o Natal tem um grande impacto na taxa de acidentes, sendo um período festivo que leva as pessoas a sair de casa e viajar, o que aumenta a probabilidade de acidentes. 
            Além disso, observamos que a maioria dos acidentes fatais são causados por motoristas jovens, com uma disparidade notável entre os gêneros: os homens causam 3,32 vezes mais acidentes fatais do que as mulheres.
        """)

# Função para limpar os dados
def clean_data(data):
    data_cleaned = data.copy()
    # Remover colunas irrelevantes
    columns_to_drop = ['Speed Limit', 'National Remoteness Areas', 'SA4 Name 2016', 'National Road Type', 'Bus Involvement', 'Heavy Rigid Truck Involvement']
    data_cleaned.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # Conversões de tipos
    data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M', errors='coerce')
    data_cleaned['Month'] = pd.Categorical(data_cleaned['Month'], categories=range(1, 13))
    data_cleaned['Year'] = data_cleaned['Year'].astype(int)
    data_cleaned['Christmas Period'] = data_cleaned['Christmas Period'].map({'Yes': 1, 'No': 0})
    data_cleaned['Easter Period'] = data_cleaned['Easter Period'].map({'Yes': 1, 'No': 0})
    return data_cleaned

# Total de acidentes por ano
def total_accidents_per_year(data):
    yearly_accidents = data['Year'].value_counts().sort_index()
    fig = px.line(x=yearly_accidents.index, y=yearly_accidents.values, labels={'x': 'Ano', 'y': 'Acidentes'}, title="Total de Acidentes por Ano")
    st.plotly_chart(fig)

# Calcular a taxa de diminuição de acidentes
def calculate_decrease_rate(data):
    yearly_accidents = data['Year'].value_counts().sort_index()
    return (yearly_accidents.iloc[0] - yearly_accidents.iloc[-1]) / yearly_accidents.iloc[0] * 100

# Distribuição de acidentes por mês (frequência relativa)
def monthly_accidents_distribution(data):
    data_filtered = data[data['Year'] >= 2010]
    monthly_accidents = data_filtered['Month'].value_counts(normalize=True).sort_index() * 100  # Frequência relativa
    fig = px.bar(x=monthly_accidents.index, y=monthly_accidents.values, labels={'x': 'Mês', 'y': 'Frequência Relativa (%)'}, title="Frequência Relativa de Acidentes por Mês")
    st.plotly_chart(fig)

# Distribuição de acidentes por idade
def age_distribution(data):
    data_filtered = data[data['Year'] >= 2010]
    bins = int(1 + 3.322 * np.log10(len(data_filtered)))  # Regra de Sturges
    fig, ax = plt.subplots()
    ax.hist(data_filtered['Age'], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribuição de Idade dos Envolvidos')
    ax.set_xlabel('Idade')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

# Comparativo entre jovens homens e mulheres
def gender_comparison(data):
    data_young = data[(data['Age Group'] == '17_to_25') & (data['Gender'].notnull())]
    gender_counts = data_young['Gender'].value_counts()
    st.write(f"Homens dessa faixa etária têm {gender_counts['Male'] / gender_counts['Female']:.2f} vezes mais acidentes do que as mulheres.")
    fig = px.bar(x=gender_counts.index, y=gender_counts.values, labels={'x': 'Gênero', 'y': 'Número de Acidentes'}, title="Acidentes por Gênero entre Jovens")
    st.plotly_chart(fig)

# Boxplot de Gênero e Idade dos Envolvidos
def gender_age_boxplot(data):
    st.header("Boxplot de Gênero e Idade dos Envolvidos")
    data_filtered = data[data['Gender'].notnull()]
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Gender', y='Age', data=data_filtered, palette='Set2')
    plt.title("Distribuição de Idade por Gênero dos Envolvidos em Acidentes")
    plt.xlabel("Gênero")
    plt.ylabel("Idade")
    st.pyplot(plt)

# Acidentes por dia da semana e período
def accidents_by_day_and_period(data):
    data_filtered = data[data['Year'] >= 2011]
    accidents_day_period = data_filtered.groupby(['Dayweek', 'Time of day']).size().unstack().fillna(0)
    diurnal_to_nocturnal_ratio = (
        accidents_day_period['Day'].sum() / accidents_day_period['Night'].sum()
    )
    st.write(f"A proporção de acidentes diurnos para noturnos é de {diurnal_to_nocturnal_ratio:.2f}.")
    accidents_day_period.plot(kind='bar', stacked=False, color=['yellow', 'blue'])
    plt.title("Acidentes por Dia da Semana e Período")
    plt.xlabel("Dia da Semana")
    plt.ylabel("Quantidade de Acidentes")
    st.pyplot(plt)

# Teste T para influências de datas comemorativas
def inferential_statistics(data):
    # Teste T para o Natal
    christmas_accidents = data[data['Christmas Period'] == 1]['Age']
    non_christmas_accidents = data[data['Christmas Period'] == 0]['Age']
    t_stat_christmas, p_val_christmas = ttest_ind(christmas_accidents, non_christmas_accidents, nan_policy='omit')
    st.write(f"Estatística t para o Natal: {t_stat_christmas:.2f}, valor-p: {p_val_christmas:.4f}")
    if p_val_christmas < 0.05:
        st.write("A diferença de acidentes durante o Natal é estatisticamente significativa.")
    else:
        st.write("A diferença de acidentes durante o Natal não é estatisticamente significativa.")
    
    # Teste T para a Páscoa
    easter_accidents = data[data['Easter Period'] == 1]['Age']
    non_easter_accidents = data[data['Easter Period'] == 0]['Age']
    t_stat_easter, p_val_easter = ttest_ind(easter_accidents, non_easter_accidents, nan_policy='omit')
    st.write(f"Estatística t para a Páscoa: {t_stat_easter:.2f}, valor-p: {p_val_easter:.4f}")
    if p_val_easter < 0.05:
        st.write("A diferença de acidentes durante a Páscoa é estatisticamente significativa.")
    else:
        st.write("A diferença de acidentes durante a Páscoa não é estatisticamente significativa.")

# Executar a função principal
if __name__ == "__main__":
    main()
