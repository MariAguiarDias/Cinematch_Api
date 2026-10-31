import os
import pickle
import time
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from knn import rodar_knn, avaliar_knn
from matriz_fatorizada import rodar_svd, avaliar_svd
from rede_neural import rodar_rede_neural, avaliar_rede_neural

ARQUIVO_FILMES = "movies_small.csv"  # catálogo de filmes usado na execução
ARQUIVO_AVALIACOES = "ratings_small.csv"  # avaliações usadas na execução
CACHE_DADOS = "dados_processados.pkl"  # arquivo de cache dos CSVs

MIN_AVALIACOES = 5 # mínimo de notas exigido do usuário, sem limite máximo
NOTA_MIN, NOTA_MAX = 0.5, 5.0  # limites da escala de notas do MovieLens
CORE_GENERO_MIN_FRACAO = 0.6  # fração dos filmes do usuário em que um gênero precisa aparecer para entrar no perfil core
MODO_FILTRO_GENERO = "core" # "core", "intersecao" ou None, detalhados em filtros.py
PROPORCAO_TESTE = 0.2  # parte da base separada para medir as métricas
SEMENTE = 42  # semente que torna a divisão treino/teste reproduzível
SEPARADOR = "\n"+ ("-" * 100) + "\n"  # linha que separa as etapas na saída do terminal
CALCULAR_METRICAS = False  # True mede RMSE e MAE dos algoritmos, o que deixa a execução alguns segundos mais lenta


# Identifica a versão atual dos CSVs pelo tamanho e pela data de modificação
def assinatura_arquivos():
    return {
        nome: (os.path.getsize(nome), os.path.getmtime(nome))
        for nome in (ARQUIVO_FILMES, ARQUIVO_AVALIACOES)
    }


# Carrega filmes e avaliações, reaproveitando o cache enquanto os CSVs não mudarem
def carregamento_dados():
    """
    Carrega os dados dos filmes e das avaliações.
    return: Lista de DataFrames (movies, ratings)
    """

    assinatura = assinatura_arquivos()  # versão atual dos CSVs

    if os.path.exists(CACHE_DADOS):
        # Um cache gerado em outro computador ou com outra versão do pandas pode não abrir, e nesse caso é refeito
        try:
            with open(CACHE_DADOS, "rb") as f:
                dados = pickle.load(f)  # conteúdo salvo no cache
        except Exception:
            dados = {}  # cache ilegível, tratado como desatualizado

        # O cache só é reaproveitado se tiver sido gerado a partir dos mesmos CSVs
        if dados.get("assinatura") == assinatura:
            print("\nCarregando ratings e movies do cache local...")
            return dados["movies"], dados["ratings"]

        print("\nCache desatualizado ou ilegível, recarregando os CSVs...")
    else:
        print()  # separa a leitura dos CSVs da linha de início

    print("Lendo arquivos CSV originais...")
    movies = pd.read_csv(ARQUIVO_FILMES)  # catálogo com movieId, title e genres
    ratings = pd.read_csv(ARQUIVO_AVALIACOES)  # avaliações com userId, movieId, rating e timestamp

    with open(CACHE_DADOS, "wb") as f:
        pickle.dump({"movies": movies, "ratings": ratings, "assinatura": assinatura}, f)
    print("Dados salvos em cache com sucesso!")

    return movies, ratings


# Recusa avaliações que os algoritmos não conseguiriam usar, explicando o motivo
def validar_avaliacoes(novas_avaliacoes, movies):

    if len(novas_avaliacoes) < MIN_AVALIACOES:
        raise ValueError(
            f"É preciso avaliar pelo menos {MIN_AVALIACOES} filmes "
            f"({len(novas_avaliacoes)} recebidas)."
        )

    ids = [mid for mid, _ in novas_avaliacoes]  # movieIds informados pelo usuário

    # Notas repetidas para o mesmo filme seriam somadas na matriz do KNN
    repetidos = sorted(mid for mid, qtd in Counter(ids).items() if qtd > 1)
    if repetidos:
        raise ValueError(f"Filmes avaliados mais de uma vez: {repetidos}")

    inexistentes = sorted(set(ids) - set(movies["movieId"]))  # IDs que não estão no catálogo
    if inexistentes:
        raise ValueError(f"movieIds que não existem em {ARQUIVO_FILMES}: {inexistentes}")

    # Multiplicar por 2 transforma os passos de 0.5 em números inteiros
    fora_da_escala = [  
        # pares com nota fora da faixa ou fora dos passos de 0.5
        (mid, nota) for mid, nota in novas_avaliacoes
        if not NOTA_MIN <= nota <= NOTA_MAX or (nota * 2) % 1 != 0
    ]
    if fora_da_escala:
        raise ValueError(
            f"Notas fora da escala de {NOTA_MIN} a {NOTA_MAX} em passos de 0.5: {fora_da_escala}"
        )


# Valida as avaliações e as acrescenta à base como um novo usuário
def inserir_novo_usuario(ratings, movies, novas_avaliacoes):
    """
    Insere um novo usuário no dataset de ratings, com suas avaliações.
    TODO: inserir no arquivo ratings.csv também.
    ratings: DataFrame de ratings atual
    novas_avaliacoes: lista de tuplas (movieId, rating)
    return: tupla (ratings com o novo usuário, userId do novo usuário)
    """

    # Validar antes de inserir impede que IDs inexistentes sejam descartados em silêncio no merge
    validar_avaliacoes(novas_avaliacoes, movies)

    novo_usuario_id = int(ratings["userId"].max()) + 1  # próximo ID livre da base
    timestamp = int(pd.Timestamp.now().timestamp())  # momento da inserção, no mesmo formato do MovieLens

    # Avaliações do novo usuário no formato do ratings
    novos_registros = pd.DataFrame({
        "userId": [novo_usuario_id] * len(novas_avaliacoes),
        "movieId": [mid for mid, _ in novas_avaliacoes],
        "rating": [float(r) for _, r in novas_avaliacoes],
        "timestamp": [timestamp] * len(novas_avaliacoes)
    })

    return pd.concat([ratings, novos_registros], ignore_index=True), novo_usuario_id


# Pré-calcula o histórico e o perfil de gêneros do usuário, compartilhados por todos os algoritmos
def montar_calculos(movies, ratings, user_id):

    # Pega os filmes já avaliados pelo usuário-alvo
    user_ratings = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")  # avaliações do usuário com título e gêneros
    watched_ids = set(user_ratings["movieId"])  # filmes que o usuário já avaliou
    
    # filmes avaliados pelo usuário, da maior para a menor nota
    historico_usuario = (
        user_ratings[["title", "genres", "rating"]]
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )

    # Cria o perfil de gêneros do usuário, ou seja, a união de todos os gêneros dos filmes que ele avaliou
    perfil_generos_usuario = set()  # todos os gêneros do histórico do usuário
    for genres_str in user_ratings["genres"]:
        if genres_str != "(no genres listed)":
            perfil_generos_usuario.update(genres_str.split("|"))

    # Cria o perfil "core", com gêneros que aparecem em pelo menos CORE_GENERO_MIN_FRACAO dos filmes avaliados (captura a COMBINAÇÃO que define o gosto do usuário)
    contagem_generos = Counter()  # em quantos filmes do usuário cada gênero aparece
    for genres_str in user_ratings["genres"]:
        if genres_str != "(no genres listed)":
            contagem_generos.update(set(genres_str.split("|")))

    total_filmes_usuario = len(user_ratings)  # quantidade de filmes avaliados pelo usuário

    # Gêneros que definem o gosto do usuário
    perfil_core_usuario = {
        genero
        for genero, qtd in contagem_generos.items()
        if qtd / total_filmes_usuario >= CORE_GENERO_MIN_FRACAO
    }

    # Conjunto de gêneros de cada filme
    genero_por_filme = {
        # movieId para o conjunto de gêneros do filme
        int(mid): set(g.split("|")) if g != "(no genres listed)" else set()
        for mid, g in zip(movies["movieId"], movies["genres"])
    }

    return {
        "watched_ids": watched_ids,
        "perfil_generos_usuario": perfil_generos_usuario,
        "perfil_core_usuario": perfil_core_usuario,
        "genero_por_filme": genero_por_filme,
        "modo_filtro_genero": MODO_FILTRO_GENERO,
        "historico_usuario": historico_usuario,
    }


# Separa a base uma única vez para que todos os algoritmos sejam avaliados nas mesmas avaliações
def dividir_treino_teste(ratings):

    # Constrói o trainset e o testset (20% dos dados para teste)
    return train_test_split(ratings, test_size=PROPORCAO_TESTE, random_state=SEMENTE)


# Converte um valor em texto para exibição, sem alterar o dado original
def formatar_valor(valor, casas):
    if isinstance(valor, float):
        return f"{valor:.{casas}f}"
    if isinstance(valor, (list, set, tuple)):
        return ", ".join(sorted(str(item) for item in valor))
    return str(valor)


# Converte uma coluna em texto, com todos os números no mesmo número de casas decimais (de 1 a 4)
def formatar_coluna(coluna):
    casas = max([1] + [len(f"{v:.4f}".rstrip("0").split(".")[1]) for v in coluna if isinstance(v, float)])  # menor quantidade de casas que mostra todos os valores
    return coluna.map(lambda valor: formatar_valor(valor, casas))


# Monta uma tabela em texto com todas as colunas alinhadas à esquerda, usada só na exibição do terminal
def formatar_tabela(df):
    textos = df.apply(formatar_coluna)  # cópia da tabela com cada valor já em texto
    larguras = {c: max([len(str(c))] + textos[c].str.len().tolist()) for c in textos.columns}  # largura de cada coluna
    linhas = ["  ".join(str(c).ljust(larguras[c]) for c in textos.columns)]  # começa pelo cabeçalho
    linhas += [
        "  ".join(valor.ljust(larguras[c]) for c, valor in zip(textos.columns, linha))
        for linha in textos.itertuples(index=False)
    ]
    return "\n".join(linha.rstrip() for linha in linhas)


if __name__ == "__main__":

    print("\nInício do carregamento dos dados")
    start = time.time()  # início da medição de tempo de cada etapa

    movies, ratings_base = carregamento_dados()  # catálogo e avaliações originais, sem o usuário fictício

    # Horror
    avaliacoes_usuario = [
        # Pares (movieId, nota) de um usuário que gosta de terror
        (2459, 5.0),
        (148671, 5.0),
        (168250, 4.5),
        (168252, 5.0),
        (162082, 4.0),
    ]

    # Avaliações inválidas encerram a execução com uma mensagem clara em vez de um traceback
    try:
        ratings, USER_ID = inserir_novo_usuario(ratings_base, movies, avaliacoes_usuario)  # base com o usuário fictício e o ID dele
    except ValueError as erro:
        raise SystemExit(f"Avaliações inválidas. {erro}")

    calculos = montar_calculos(movies, ratings, USER_ID)  # histórico e perfil de gêneros usados pelos dois algoritmos
    
    print(SEPARADOR)
    print(f"===== Filmes avaliados pelo usuário {USER_ID} =====")
    print(formatar_tabela(calculos["historico_usuario"]))

    print(f"\nPerfil de gêneros do usuário {USER_ID}: {sorted(calculos['perfil_generos_usuario'])}")
    print(f"Perfil de gêneros do usuário {USER_ID} (core, >= {CORE_GENERO_MIN_FRACAO:.0%} dos filmes): {sorted(calculos['perfil_core_usuario'])}")
    print(f"Modo do filtro de gênero: {MODO_FILTRO_GENERO}")

    end = time.time()  # fim da medição de tempo da etapa atual
    print(f"Fim do carregamento dos dados ({end - start:.4f} segundos)")

    tempos_recomendacao = {}  # segundos que cada algoritmo levou para gerar o top 5

    print(SEPARADOR)
    print("Começo KNN...")
    start = time.time()

    recomendacoes_knn = rodar_knn(movies, ratings, USER_ID, calculos)  # top 5 do KNN

    print(f"\n===== Top 5 filmes recomendados para o usuário {USER_ID} (KNN) =====")
    print(formatar_tabela(recomendacoes_knn[["title", "genres", "nota_estimada", "qtd_vizinhos", "generos_em_comum"]]))

    end = time.time()
    tempos_recomendacao["KNN"] = end - start
    print(f"\nFim do KNN... ({end - start:.4f} segundos)")
    print(SEPARADOR)
    print("Começo Matriz Fatorizada...")
    start = time.time()

    recomendacoes_svd = rodar_svd(movies, ratings, USER_ID, calculos)  # top 5 do SVD

    print(f"\n===== Top 5 filmes recomendados para o usuário {USER_ID} (SVD) =====")
    print(formatar_tabela(recomendacoes_svd[["title", "genres", "nota_estimada", "generos_em_comum"]]))

    end = time.time()
    tempos_recomendacao["SVD"] = end - start
    print(f"\nFim do Matriz Fatorizada... ({end - start:.4f} segundos)")
    print(SEPARADOR)
    print("Começo Rede Neural...")
    start = time.time()

    recomendacoes_rede = rodar_rede_neural(movies, ratings, USER_ID, calculos)  # top 5 da rede neural

    print(f"\n===== Top 5 filmes recomendados para o usuário {USER_ID} (Rede Neural) =====")
    print(formatar_tabela(recomendacoes_rede[["title", "genres", "nota_estimada", "generos_em_comum"]]))

    end = time.time()
    tempos_recomendacao["Rede Neural"] = end - start
    print(f"\nFim da Rede Neural... ({end - start:.4f} segundos)")
    print(SEPARADOR)

    resumo = pd.DataFrame({"tempo_recomendacao_s": tempos_recomendacao})  # uma linha por algoritmo, com o tempo do top 5

    # A avaliação treina cada algoritmo de novo, então só roda quando as métricas forem necessárias
    if CALCULAR_METRICAS:

        # A avaliação usa a base original, sem o usuário fictício, para que as métricas não mudem a cada perfil testado
        treino_df, teste_df = dividir_treino_teste(ratings_base)  # 80% para treino e 20% para teste

        metricas = {}  # RMSE, MAE, fração prevista e tempo de cada algoritmo
        for nome, avaliar in [("KNN", avaliar_knn), ("SVD", avaliar_svd), ("Rede Neural", avaliar_rede_neural)]:
            start = time.time()
            metricas[nome] = avaliar(treino_df, teste_df)
            metricas[nome]["tempo_avaliacao_s"] = time.time() - start

        resumo = resumo.join(pd.DataFrame(metricas).T)  # acrescenta RMSE, MAE e tempo de avaliação ao resumo

    print("===== Resumo da execução =====")
    print(formatar_tabela(resumo.rename_axis("algoritmo").reset_index()))

    if CALCULAR_METRICAS:
        print(f"\nRMSE e MAE medidos em {len(teste_df)} avaliações de teste")
