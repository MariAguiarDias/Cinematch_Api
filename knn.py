import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from filtros import aplicar_filtro_genero

N_RECOMENDACOES = 5  # quantidade de filmes na lista final
K_VIZINHOS = 20 # quantos usuários parecidos foram considerados
SUPORTE_MINIMO = 3 # o filme precisa ter sido avaliado por pelo menos N vizinhos

# Monta a matriz esparsa usuário x filme e guarda os IDs reais de cada linha e coluna
def montar_matriz(ratings):

    # Converte a coluna de IDs de usuários para o tipo "category" do Pandas
    user_ids = ratings["userId"].astype("category")  # O Pandas mapeia cada ID real para um número inteiro sequencial (começando do 0)
    movie_ids = ratings["movieId"].astype("category")  # mesmo mapeamento, aplicado aos IDs de filmes

    # Extrai e guarda a lista ordenada dos IDs reais
    user_index = user_ids.cat.categories  # A posição numérica na matriz corresponde ao valor real guardado aqui
    movie_index = movie_ids.cat.categories  # IDs reais dos filmes, na ordem das colunas da matriz

    # Cria uma matriz com as notas
    matriz = csr_matrix(
        # Linhas são usuários, colunas são filmes e zero significa sem avaliação
        (
            ratings["rating"],
            (user_ids.cat.codes, movie_ids.cat.codes),
        ),
        shape=(len(user_index), len(movie_index)),
    )

    return matriz, user_index, movie_index


# Configura o modelo de vizinhança e o ajusta sobre a matriz de notas
def treinar_knn(matriz):

    # Configura o modelo de vizinhos mais próximos. Sem centralizar as notas pela média, a escala de cada usuário ainda pesa um pouco
    knn = NearestNeighbors(
        # Usa a "similaridade de cosseno" para calcular a distância entre os vetores
        metric="cosine",  # Isso mede o ângulo entre os perfis de notas, focando nos gostos em comum
        algorithm="brute",  # compara com todos os usuários, sem índice aproximado
    )

    knn.fit(matriz)

    return knn


# Busca os vizinhos mais parecidos com um usuário, sem ele mesmo e sem os de similaridade nula
def buscar_vizinhos(knn, matriz, user_pos):

    # Pega os K_VIZINHOS vizinhos mais próximos do usuário-alvo, com distâncias e posições do mais próximo ao mais distante
    distancias, indices = knn.kneighbors(
        matriz[user_pos], n_neighbors=K_VIZINHOS + 1
    )

    # Remove o próprio usuário-alvo da lista de vizinhos
    mascara = indices[0] != user_pos  # True para todos, menos para o próprio usuário

    # Menores distâncias significam maior similaridade
    similaridades = 1 - distancias[0][mascara]  # similaridade de cada vizinho, de 0 a 1
    vizinhos = indices[0][mascara]  # posição de cada vizinho na matriz

    # Vizinhos sem nenhuma similaridade não dizem nada sobre o gosto do usuário
    positivos = similaridades > 0  # True para os vizinhos que serão mantidos

    return vizinhos[positivos], similaridades[positivos]


# Roda o algoritmo KNN para recomendar filmes para o usuário USER_ID
def rodar_knn(movies, ratings, USER_ID, calculos):

    watched_ids = calculos['watched_ids']  # filmes que o usuário já avaliou
    perfil_generos_usuario = calculos['perfil_generos_usuario']  # todos os gêneros do histórico do usuário
    genero_por_filme = calculos['genero_por_filme']  # conjunto de gêneros de cada filme

    matriz_avaliacoes, user_index, movie_index = montar_matriz(ratings)  # matriz de notas e IDs reais de linhas e colunas

    # Pega a posição do usuário-alvo dentro da matriz
    user_pos = np.where(user_index == USER_ID)[0][0]  # linha do usuário-alvo na matriz

    knn = treinar_knn(matriz_avaliacoes)  # modelo ajustado com todas as avaliações
    vizinhos, similaridades = buscar_vizinhos(knn, matriz_avaliacoes, user_pos)  # vizinhos úteis e o peso de cada um

    # Agrega as notas dos vizinhos, ponderadas pela similaridade, para todos os filmes que o usuário-alvo ainda não viu
    scores = {}  # soma de similaridade * nota de cada filme
    soma_similaridades = {}  # soma das similaridades de quem avaliou cada filme
    contagem_vizinhos = {}  # Quantos vizinhos avaliaram cada filme (suporte)

    for idx_vizinho, similaridade in zip(vizinhos, similaridades):

        # Transforma as notas do vizinho atual de linha da matriz esparsa para um array plano
        notas_vizinho = matriz_avaliacoes[idx_vizinho].toarray().ravel()  # nota do vizinho para cada filme, com zero onde não avaliou

        # Pega os índices dos filmes que este vizinho realmente avaliou (valores diferentes de zero)
        filmes_avaliados_pos = notas_vizinho.nonzero()[0]  # colunas dos filmes avaliados pelo vizinho

        # Itera sobre cada filme avaliado pelo vizinho atual
        for pos_filme in filmes_avaliados_pos:
            movie_id = movie_index[pos_filme]  # ID real do filme da coluna atual

            # Descarta filmes que o usuário-alvo já assistiu
            if movie_id in watched_ids:
                continue

            nota = notas_vizinho[pos_filme]  # nota que o vizinho deu ao filme

            # Acumula a nota ponderada pela similaridade do vizinho (similaridade * nota)
            scores[movie_id] = scores.get(movie_id, 0) + similaridade * nota

            # Acumula a soma das similaridades dos vizinhos que avaliaram este filme (para a divisão da média ponderada)
            soma_similaridades[movie_id] = soma_similaridades.get(movie_id, 0) + similaridade

            # Incrementa o contador de quantos vizinhos avaliaram este filme específico (usado no filtro de suporte mínimo)
            contagem_vizinhos[movie_id] = contagem_vizinhos.get(movie_id, 0) + 1

    # Nota estimada de cada filme candidato
    pontuacao_final = {
        # Média ponderada: soma(similaridade * nota) / soma(similaridade)
        movie_id: scores[movie_id] / soma_similaridades[movie_id]
        for movie_id in scores
        # Só considera filmes com suporte mínimo (evita recomendar algo que só 1 vizinho avaliou)
        if contagem_vizinhos[movie_id] >= SUPORTE_MINIMO
    }

    # Filtra os candidatos pelo perfil de gênero do usuário
    pontuacao_final = aplicar_filtro_genero(pontuacao_final, calculos)  # Se o candidato tiver o perfil core do usuário, ele é mantido

    # Seleciona os top N_RECOMENDACOES filmes recomendados
    top_ids = sorted(pontuacao_final, key=pontuacao_final.get, reverse=True)[:N_RECOMENDACOES]  # IDs dos filmes com maior nota estimada

    # Um perfil core muito restrito pode deixar menos candidatos do que o necessário
    if len(top_ids) < N_RECOMENDACOES:
        print(f"Aviso: o KNN encontrou apenas {len(top_ids)} candidatos após os filtros.")

    # Monta o DataFrame final de recomendações para retornar
    recomendados = movies[movies["movieId"].isin(top_ids)].copy()  # dados dos filmes escolhidos
    recomendados["nota_estimada"] = recomendados["movieId"].map(pontuacao_final)
    recomendados["qtd_vizinhos"] = recomendados["movieId"].map(contagem_vizinhos)
    recomendados["generos_em_comum"] = recomendados["movieId"].apply(
        lambda mid: sorted(genero_por_filme.get(int(mid), set()) & perfil_generos_usuario)
    )
    recomendados = recomendados.sort_values("nota_estimada", ascending=False).reset_index(drop=True)  # lista da maior para a menor nota estimada

    return recomendados


# Mede RMSE e MAE do KNN no teste, usando a média do usuário quando nenhum vizinho avaliou o filme
def avaliar_knn(treino, teste):

    matriz, user_index, movie_index = montar_matriz(treino)  # matriz montada só com o treino
    knn = treinar_knn(matriz)  # modelo ajustado só com o treino

    # Médias usadas como previsão de reserva quando os vizinhos não conseguem prever
    media_global = treino["rating"].mean()  # média de todas as notas do treino
    media_usuario = treino.groupby("userId")["rating"].mean()  # média de cada usuário no treino

    # Dicionários evitam buscas lineares nos índices a cada usuário do teste
    pos_usuario = {uid: pos for pos, uid in enumerate(user_index)}  # ID real do usuário para a linha da matriz
    pos_filme = {mid: pos for pos, mid in enumerate(movie_index)}  # ID real do filme para a coluna da matriz

    erros = []  # diferenças entre previsão e nota real, por usuário
    previstos_por_vizinhos = 0  # quantos pares do teste foram previstos pelos vizinhos

    for user_id, grupo in teste.groupby("userId"):
        reais = grupo["rating"].to_numpy()  # notas reais do usuário no teste
        previsoes = np.full(len(grupo), media_usuario.get(user_id, media_global))  # começa com a previsão de reserva em todos os filmes

        if user_id in pos_usuario:
            vizinhos, similaridades = buscar_vizinhos(knn, matriz, pos_usuario[user_id])  # vizinhos úteis do usuário no treino

            # Filmes ausentes do treino recebem -1 e mantêm a previsão de reserva
            colunas = np.array([pos_filme.get(mid, -1) for mid in grupo["movieId"]])  # coluna de cada filme do teste na matriz
            conhecidos = np.where(colunas >= 0)[0]  # posições dos filmes que existem no treino

            if len(vizinhos) and len(conhecidos):
                notas = matriz[vizinhos][:, colunas[conhecidos]].toarray()  # notas dos vizinhos para os filmes do teste

                # Mesma média ponderada da recomendação, calculada de uma vez para todos os filmes do usuário
                numerador = similaridades @ notas  # soma de similaridade * nota de cada filme
                denominador = similaridades @ (notas > 0)  # soma das similaridades de quem avaliou cada filme

                com_vizinho = denominador > 0  # True onde ao menos um vizinho avaliou o filme
                previsoes[conhecidos[com_vizinho]] = numerador[com_vizinho] / denominador[com_vizinho]
                previstos_por_vizinhos += int(com_vizinho.sum())

        erros.append(previsoes - reais)

    erros = np.concatenate(erros)  # todos os erros do teste em um único array

    return {
        "rmse": float(np.sqrt(np.mean(erros ** 2))),
        "mae": float(np.mean(np.abs(erros))),
        "fracao_prevista_pelo_modelo": previstos_por_vizinhos / len(erros),
    }
