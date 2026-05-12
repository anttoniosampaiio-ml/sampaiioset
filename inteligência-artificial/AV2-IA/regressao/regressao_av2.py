"""
Projeto de Regressão - AV2 Inteligência Artificial

Dataset: Communities and Crime
Modelos: Regressão Linear, Ridge Regression e Lasso Regression

Este arquivo executa um pipeline completo de regressão:
- leitura e exploração inicial do dataset;
- tratamento de valores ausentes e inconsistências;
- análise de atributos relevantes;
- treinamento dos modelos Linear, Ridge e Lasso;
- avaliação com MAE, MSE, RMSE e R²;
- visualização da reta de regressão simples;
- validação cruzada;
- Grid Search para Ridge e Lasso;
- comparação final dos resultados.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

DATASET_PATH = Path("communities.csv")
OUTPUT_DIR = Path("regressao")
TARGET = "ViolentCrimesPerPop"


# -------------------------------------------------------
# 1. Carregamento do dataset
# -------------------------------------------------------

def carregar_dados(caminho=DATASET_PATH):
    """
    Carrega o dataset Communities and Crime a partir de um arquivo CSV.

    Caso o arquivo esteja em outra pasta, basta alterar a constante DATASET_PATH.
    O dataset possui atributos socioeconômicos das comunidades e a variável alvo
    ViolentCrimesPerPop, que representa crimes violentos por população.
    """
    if not caminho.exists():
        caminho_alternativo = Path("datasets") / caminho.name
        if caminho_alternativo.exists():
            caminho = caminho_alternativo
        else:
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}. "
                "Coloque o communities.csv na mesma pasta do código ou dentro da pasta datasets."
            )

    df = pd.read_csv(caminho)
    return df


# -------------------------------------------------------
# 2. Exploração inicial
# -------------------------------------------------------

def explorar_dados(df):
    """
    Mostra uma visão inicial do dataset, incluindo quantidade de linhas e colunas,
    primeiras linhas, tipos de dados e estatísticas descritivas.
    Essa etapa ajuda a compreender a estrutura dos dados antes da modelagem.
    """
    print("--- Exploração Inicial ---")
    print(f"\nShape: {df.shape}")

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nEstatísticas descritivas:")
    print(df.describe().T.round(4))

    print("\nInformações gerais:")
    df.info()


# -------------------------------------------------------
# 3. Valores ausentes e inconsistências
# -------------------------------------------------------

def verificar_missing(df):
    """
    Verifica a quantidade e o percentual de valores ausentes por coluna.
    No Communities and Crime, algumas colunas possuem muitos valores faltantes,
    principalmente atributos relacionados a dados policiais.
    """
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100

    tabela_missing = pd.DataFrame({
        "missing": missing,
        "percentual": missing_percent.round(2)
    }).sort_values("missing", ascending=False)

    print("\n--- Valores Ausentes ---")
    print(tabela_missing[tabela_missing["missing"] > 0].head(30))

    return tabela_missing


def plotar_missing(tabela_missing):
    """
    Plota as 20 colunas com maior percentual de valores ausentes.
    O gráfico auxilia na decisão de remover colunas com muitos dados faltando.
    """
    top_missing = tabela_missing[tabela_missing["missing"] > 0].head(20)

    if top_missing.empty:
        print("Nenhum valor ausente encontrado para plotar.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_missing.reset_index(),
        x="percentual",
        y="index",
        color="steelblue",
        ax=ax
    )
    ax.set_title("Top 20 colunas com valores ausentes")
    ax.set_xlabel("Percentual de valores ausentes (%)")
    ax.set_ylabel("Colunas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values_regressao.png", dpi=150)
    plt.close()

    print("salvo: missing_values_regressao.png")


def preparar_base(df, limite_missing=0.50):
    """
    Realiza a limpeza inicial do dataset.

    Etapas:
    1. Substitui possíveis símbolos '?' por NaN.
    2. Remove colunas identificadoras que não ajudam na regressão.
    3. Converte colunas para valores numéricos quando possível.
    4. Remove colunas com mais de 50% de valores ausentes.
    5. Mantém os valores ausentes restantes para serem tratados dentro do Pipeline.

    A imputação é feita no Pipeline para evitar data leakage, ou seja, para evitar
    que informações do conjunto de teste sejam usadas durante o treinamento.
    """
    df = df.copy()

    df.replace("?", np.nan, inplace=True)

    colunas_identificadoras = [
        "state",
        "county",
        "community",
        "communityname",
        "fold"
    ]

    colunas_para_remover = [col for col in colunas_identificadoras if col in df.columns]
    df.drop(columns=colunas_para_remover, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    limite = int(len(df) * limite_missing)
    colunas_muito_missing = df.columns[df.isna().sum() > limite].tolist()

    if TARGET in colunas_muito_missing:
        colunas_muito_missing.remove(TARGET)

    df.drop(columns=colunas_muito_missing, inplace=True)

    print("\n--- Pré-processamento inicial ---")
    print(f"Colunas identificadoras removidas: {colunas_para_remover}")
    print(f"Colunas removidas por excesso de valores ausentes: {len(colunas_muito_missing)}")
    print(f"Shape após limpeza inicial: {df.shape}")

    return df, colunas_muito_missing


# -------------------------------------------------------
# 4. Análise da variável alvo e atributos relevantes
# -------------------------------------------------------

def plotar_distribuicao_target(df):
    """
    Plota a distribuição da variável alvo ViolentCrimesPerPop.
    Como é uma variável numérica, usa-se histograma com curva de densidade.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=TARGET, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribuição da variável alvo - ViolentCrimesPerPop")
    ax.set_xlabel("Crimes violentos por população")
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribuicao_target_regressao.png", dpi=150)
    plt.close()

    print("salvo: distribuicao_target_regressao.png")


def obter_base_para_analise(df):
    """
    Cria uma cópia do dataset com valores ausentes preenchidos pela mediana.
    Essa base é usada apenas para análise exploratória e correlação.
    Para o treinamento dos modelos, a imputação acontece dentro do Pipeline.
    """
    df_analise = df.copy()
    medianas = df_analise.median(numeric_only=True)
    df_analise = df_analise.fillna(medianas)
    return df_analise


def analisar_correlacoes(df):
    """
    Calcula a correlação de Pearson entre cada atributo e a variável alvo.
    Retorna uma tabela ordenada pela correlação absoluta, destacando os atributos
    com maior relação com ViolentCrimesPerPop.
    """
    df_analise = obter_base_para_analise(df)

    correlacoes = df_analise.corr(numeric_only=True)[TARGET].drop(TARGET)
    tabela_corr = pd.DataFrame({
        "correlacao": correlacoes,
        "correlacao_abs": correlacoes.abs()
    }).sort_values("correlacao_abs", ascending=False)

    print("\n--- Top 15 atributos mais correlacionados com a variável alvo ---")
    print(tabela_corr.head(15).round(4))

    return tabela_corr


def plotar_top_correlacoes(tabela_corr, top_n=15):
    """
    Plota os atributos com maior correlação absoluta com a variável alvo.
    Esse gráfico ajuda a justificar a escolha dos atributos relevantes no relatório.
    """
    top_corr = tabela_corr.head(top_n).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_corr,
        x="correlacao",
        y="index",
        color="steelblue",
        ax=ax
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Top {top_n} correlações com ViolentCrimesPerPop")
    ax.set_xlabel("Correlação de Pearson")
    ax.set_ylabel("Atributos")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlacoes_regressao.png", dpi=150)
    plt.close()

    print("salvo: top_correlacoes_regressao.png")


def plotar_heatmap_correlacao(df, tabela_corr, top_n=10):
    """
    Plota um heatmap com a variável alvo e os atributos mais correlacionados.
    O objetivo é visualizar a relação entre as variáveis escolhidas.
    """
    df_analise = obter_base_para_analise(df)
    top_features = tabela_corr.head(top_n).index.tolist()
    cols = top_features + [TARGET]

    corr = df_analise[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Heatmap de correlação - principais atributos")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_correlacao_regressao.png", dpi=150)
    plt.close()

    print("salvo: heatmap_correlacao_regressao.png")


def selecionar_features(df, tabela_corr, top_n=15):
    """
    Seleciona os atributos mais relevantes para o modelo usando a maior correlação
    absoluta com a variável alvo. O número de atributos selecionados pode ser
    ajustado pelo parâmetro top_n.
    """
    selected_features = tabela_corr.head(top_n).index.tolist()

    X = df[selected_features]
    y = df[TARGET]

    print(f"\nFeatures selecionadas ({top_n}):")
    print(selected_features)

    return X, y, selected_features


# -------------------------------------------------------
# 5. Preparação para modelagem
# -------------------------------------------------------

def dividir_dados(X, y):
    """
    Divide os dados em treino e teste.

    Utiliza 80% dos dados para treino e 20% para teste.
    O random_state garante que a divisão seja reproduzível.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n--- Divisão dos dados ---")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    return X_train, X_test, y_train, y_test


def construir_pipeline(modelo):
    """
    Constrói um Pipeline para regressão.

    Etapas:
    - SimpleImputer: preenche valores ausentes usando a mediana do treino.
    - StandardScaler: padroniza os atributos.
    - Modelo: LinearRegression, Ridge ou Lasso.

    O uso de Pipeline evita data leakage porque o tratamento é ajustado apenas
    nos dados de treino.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", modelo)
    ])

    return pipe


# -------------------------------------------------------
# 6. Treinamento e avaliação dos modelos
# -------------------------------------------------------

def calcular_metricas(y_true, y_pred):
    """
    Calcula as principais métricas de regressão exigidas no trabalho:
    MAE, MSE, RMSE e R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def treinar_modelos(X_train, X_test, y_train, y_test):
    """
    Treina três modelos de regressão:
    - Regressão Linear padrão;
    - Ridge Regression;
    - Lasso Regression.

    Retorna os modelos treinados, as predições e a tabela de métricas.
    """
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000)
    }

    modelos_treinados = {}
    predicoes = {}
    resultados = []

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(modelo)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        modelos_treinados[nome] = pipe
        predicoes[nome] = y_pred

        metricas = calcular_metricas(y_test, y_pred)
        resultados.append({
            "modelo": nome,
            "MAE": metricas["MAE"],
            "MSE": metricas["MSE"],
            "RMSE": metricas["RMSE"],
            "R2": metricas["R2"]
        })

    df_resultados = pd.DataFrame(resultados).set_index("modelo")

    print("\n--- Resultados no conjunto de teste ---")
    print(df_resultados.round(4))

    df_resultados.to_csv(OUTPUT_DIR / "resultados_teste_regressao.csv")

    return modelos_treinados, predicoes, df_resultados


def plotar_comparacao_modelos(df_resultados):
    """
    Gera gráficos comparando os modelos por RMSE e R².
    Quanto menor o RMSE, melhor. Quanto maior o R², melhor.
    """
    df_plot = df_resultados.reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_plot, x="modelo", y="RMSE", color="steelblue", ax=ax)
    ax.set_title("Comparação dos modelos - RMSE")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparacao_rmse_regressao.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_plot, x="modelo", y="R2", color="steelblue", ax=ax)
    ax.set_title("Comparação dos modelos - R²")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("R²")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparacao_r2_regressao.png", dpi=150)
    plt.close()

    print("salvo: comparacao_rmse_regressao.png")
    print("salvo: comparacao_r2_regressao.png")


# -------------------------------------------------------
# 7. Regressão linear simples com um atributo
# -------------------------------------------------------

def escolher_atributo_regressao_simples(tabela_corr):
    """
    Escolhe automaticamente o atributo com maior correlação absoluta com a variável alvo.
    Esse atributo será usado para visualizar a reta de regressão simples.
    """
    atributo = tabela_corr.index[0]
    correlacao = tabela_corr.iloc[0]["correlacao"]

    print("\n--- Atributo escolhido para regressão simples ---")
    print(f"Atributo: {atributo}")
    print(f"Correlação com {TARGET}: {correlacao:.4f}")

    return atributo


def plotar_reta_regressao_simples(df, atributo):
    """
    Treina uma regressão linear simples usando apenas um atributo e plota:
    - pontos reais do dataset;
    - reta de regressão estimada.

    Esse gráfico atende ao critério de visualização da reta de regressão com
    um único atributo relevante.
    """
    dados = df[[atributo, TARGET]].copy()
    dados = dados.fillna(dados.median(numeric_only=True))

    X = dados[[atributo]]
    y = dados[TARGET]

    modelo = LinearRegression()
    modelo.fit(X, y)

    x_linha = np.linspace(X[atributo].min(), X[atributo].max(), 200)
    y_linha = modelo.predict(pd.DataFrame({atributo: x_linha}))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dados, x=atributo, y=TARGET, alpha=0.5, ax=ax)
    ax.plot(x_linha, y_linha, linewidth=2)
    ax.set_title(f"Regressão Linear Simples: {atributo} x {TARGET}")
    ax.set_xlabel(atributo)
    ax.set_ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reta_regressao_simples.png", dpi=150)
    plt.close()

    print("salvo: reta_regressao_simples.png")

    print("\nEquação da reta:")
    print(f"{TARGET} = {modelo.intercept_:.4f} + ({modelo.coef_[0]:.4f} * {atributo})")

    return modelo


# -------------------------------------------------------
# 8. Validação cruzada
# -------------------------------------------------------

def validacao_cruzada(X_train, y_train):
    """
    Aplica validação cruzada 5-fold para avaliar os modelos de forma mais robusta.

    São calculadas as métricas:
    - MAE;
    - MSE;
    - RMSE;
    - R².

    As métricas negativas retornadas pelo scikit-learn são convertidas para valores
    positivos quando representam erro.
    """
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000)
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2"
    }

    resultados_cv = []

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(modelo)

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=1
        )

        mae = -scores["test_MAE"].mean()
        mse = -scores["test_MSE"].mean()
        rmse = np.sqrt(mse)
        r2 = scores["test_R2"].mean()

        resultados_cv.append({
            "modelo": nome,
            "MAE_CV": mae,
            "MSE_CV": mse,
            "RMSE_CV": rmse,
            "R2_CV": r2
        })

    df_cv = pd.DataFrame(resultados_cv).set_index("modelo")

    print("\n--- Validação Cruzada 5-fold ---")
    print(df_cv.round(4))

    df_cv.to_csv(OUTPUT_DIR / "resultados_validacao_cruzada.csv")

    return df_cv, kfold


# -------------------------------------------------------
# 9. Grid Search
# -------------------------------------------------------

def grid_search(X_train, X_test, y_train, y_test, kfold):
    """
    Aplica Grid Search para Ridge e Lasso, buscando o melhor valor de alpha.

    O alpha controla a força da regularização:
    - valores baixos reduzem pouco os coeficientes;
    - valores altos aumentam a penalização e deixam o modelo mais simples.

    O melhor modelo é escolhido com base no maior R² médio na validação cruzada.
    """
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    modelos = {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=20000)
    }

    resultados_grid = {}
    linhas = []

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(modelo)

        param_grid = {
            "modelo__alpha": alphas
        }

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=kfold,
            scoring="r2",
            n_jobs=1
        )

        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)

        resultados_grid[nome] = gs

        linhas.append({
            "modelo": nome,
            "melhor_alpha": gs.best_params_["modelo__alpha"],
            "melhor_R2_CV": gs.best_score_,
            "MAE_teste": metricas["MAE"],
            "MSE_teste": metricas["MSE"],
            "RMSE_teste": metricas["RMSE"],
            "R2_teste": metricas["R2"]
        })

    df_grid = pd.DataFrame(linhas).set_index("modelo")

    print("\n--- Grid Search Ridge e Lasso ---")
    print(df_grid.round(4))

    df_grid.to_csv(OUTPUT_DIR / "resultados_grid_search.csv")

    return resultados_grid, df_grid


# -------------------------------------------------------
# 10. Comparação final
# -------------------------------------------------------

def comparar_resultados(df_teste, df_cv, df_grid):
    """
    Junta os resultados de treino/teste, validação cruzada e Grid Search.
    Essa tabela facilita a comparação final entre os modelos.
    """
    comparacao = df_teste.copy()
    comparacao = comparacao.join(df_cv, how="left")

    print("\n--- Comparação geral: teste x validação cruzada ---")
    print(comparacao.round(4))

    print("\n--- Resultados otimizados por Grid Search ---")
    print(df_grid.round(4))

    comparacao.to_csv(OUTPUT_DIR / "comparacao_geral_regressao.csv")

    return comparacao


def conclusao_automatica(df_teste, df_grid):
    """
    Imprime uma conclusão simples com base nos resultados obtidos.
    O melhor modelo no teste é escolhido pelo maior R² e menor RMSE.
    """
    melhor_r2 = df_teste["R2"].idxmax()
    melhor_rmse = df_teste["RMSE"].idxmin()

    print("\n--- Conclusão crítica inicial ---")
    print(f"Melhor modelo pelo R² no teste: {melhor_r2}")
    print(f"Melhor modelo pelo RMSE no teste: {melhor_rmse}")

    if melhor_r2 == melhor_rmse:
        print(
            f"O modelo {melhor_r2} apresentou o melhor equilíbrio entre explicação "
            "da variância da variável alvo e menor erro médio de previsão."
        )
    else:
        print(
            "Os resultados mostram diferença entre o melhor modelo pelo R² e pelo RMSE. "
            "Nesse caso, a escolha final deve considerar tanto a capacidade explicativa "
            "quanto o tamanho do erro de previsão."
        )

    if not df_grid.empty:
        melhor_grid = df_grid["melhor_R2_CV"].idxmax()
        print(
            f"No Grid Search, o melhor desempenho médio em validação cruzada foi obtido por {melhor_grid}."
        )


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    """Executa o pipeline completo de regressão da AV2."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = carregar_dados()
    explorar_dados(df)

    tabela_missing = verificar_missing(df)
    plotar_missing(tabela_missing)

    df_limpo, colunas_removidas = preparar_base(df)

    plotar_distribuicao_target(df_limpo)

    tabela_corr = analisar_correlacoes(df_limpo)
    plotar_top_correlacoes(tabela_corr, top_n=15)
    plotar_heatmap_correlacao(df_limpo, tabela_corr, top_n=10)

    X, y, selected_features = selecionar_features(df_limpo, tabela_corr, top_n=15)

    X_train, X_test, y_train, y_test = dividir_dados(X, y)

    modelos_treinados, predicoes, df_teste = treinar_modelos(
        X_train,
        X_test,
        y_train,
        y_test
    )

    plotar_comparacao_modelos(df_teste)

    atributo_simples = escolher_atributo_regressao_simples(tabela_corr)
    plotar_reta_regressao_simples(df_limpo, atributo_simples)

    df_cv, kfold = validacao_cruzada(X_train, y_train)

    resultados_grid, df_grid = grid_search(
        X_train,
        X_test,
        y_train,
        y_test,
        kfold
    )

    comparar_resultados(df_teste, df_cv, df_grid)
    conclusao_automatica(df_teste, df_grid)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
