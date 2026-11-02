import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from filtros import aplicar_filtro_genero

N_RECOMENDACOES = 5  # Quantidade de filmes na lista final
K_VIZINHOS = 20 # Quantos usuários parecidos foram considerados
SUPORTE_MINIMO = 3 # O filme precisa ter sido avaliado por pelo menos N vizinhos

_modelo_base = {}  # Matriz da base original e modelo já ajustado, guardados em memória entre as chamadas

# Monta a matriz esparsa usuário x filme e guarda os IDs reais de cada linha e coluna
def montar_matriz(ratings):

    # Converte a coluna de IDs de usuários para o tipo "category" do Pandas
    user_ids = ratings["userId"].astype("category") # O Pandas mapeia cada ID real para um número inteiro sequencial 
    movie_ids = ratings["movieId"].astype("category")  # Mesmo mapeamento, aplicado aos IDs de filmes

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

    # Configura o modelo de vizinhos mais próximos.
    knn = NearestNeighbors(
        # Usa a "similaridade de cosseno" para calcular a distância entre os vetores
        metric="cosine",  # Isso mede o ângulo entre os perfis de notas, focando nos gostos em comum
        algorithm="brute",  # Compara com todos os usuários, sem índice aproximado
    )

    knn.fit(matriz)

    return knn

# Monta a matriz da base original e ajusta o modelo uma única vez, reaproveitando nas chamadas seguintes
def preparar_knn(ratings_base):

    matriz, user_index, movie_index = montar_matriz(ratings_base)  # Matriz de notas da base, sem nenhum usuário novo

    _modelo_base.update(
        matriz=matriz,
        movie_index=movie_index,
        pos_filme={mid: pos for pos, mid in enumerate(movie_index)},  # ID real do filme para a coluna da matriz
        knn=treinar_knn(matriz),
    )

# Monta a linha do usuário novo com as mesmas colunas da matriz da base
def montar_vetor_usuario(notas_usuario):

    pos_filme = _modelo_base["pos_filme"]  # Coluna de cada filme na matriz da base

    # Filmes que ninguém da base avaliou não têm coluna e não alteram a escolha dos vizinhos
    conhecidas = notas_usuario[notas_usuario["movieId"].isin(pos_filme)]  # Notas de filmes presentes na matriz
    colunas = conhecidas["movieId"].map(pos_filme).to_numpy()  # Coluna de cada nota

    return csr_matrix(
        (conhecidas["rating"].to_numpy(), (np.zeros(len(colunas), dtype=int), colunas)),
        shape=(1, len(pos_filme)),
    )

# Busca os vizinhos mais parecidos com um usuário, sem ele mesmo e sem os de similaridade nula
def buscar_vizinhos(knn, vetor_usuario, user_pos=None):

    # Um usuário que já está na matriz aparece como vizinho de si mesmo, por isso é pedido um vizinho a mais
    n_vizinhos = K_VIZINHOS + (user_pos is not None)  # Quantidade de vizinhos pedida ao modelo

    # Pega os vizinhos mais próximos do usuário-alvo, com distâncias e posições do mais próximo ao mais distante
    distancias, indices = knn.kneighbors(vetor_usuario, n_neighbors=n_vizinhos)

    # Remove o próprio usuário-alvo da lista de vizinhos
    mascara = indices[0] != user_pos

    # Menores distâncias significam maior similaridade
    similaridades = 1 - distancias[0][mascara]  # Similaridade de cada vizinho, de 0 a 1
    vizinhos = indices[0][mascara]  # Posição de cada vizinho na matriz

    # Vizinhos sem nenhuma similaridade não dizem nada sobre o gosto do usuário
    positivos = similaridades > 0

    return vizinhos[positivos], similaridades[positivos]

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
            vizinhos, similaridades = buscar_vizinhos(knn, matriz[pos_usuario[user_id]], pos_usuario[user_id])  # vizinhos úteis do usuário no treino

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

# Roda o algoritmo KNN para recomendar filmes para o usuário USER_ID
def rodar_knn(movies, ratings, USER_ID, calculos):

    watched_ids = calculos['watched_ids']  # filmes que o usuário já avaliou
    perfil_generos_usuario = calculos['perfil_generos_usuario']  # todos os gêneros do histórico do usuário
    genero_por_filme = calculos['genero_por_filme']  # conjunto de gêneros de cada filme

    # Se o main ainda não preparou o modelo, ele é montado agora com a base sem o usuário-alvo
    if not _modelo_base:
        preparar_knn(ratings[ratings["userId"] != USER_ID])

    matriz_avaliacoes = _modelo_base["matriz"]  # matriz de notas da base original
    movie_index = _modelo_base["movie_index"]  # IDs reais dos filmes, na ordem das colunas

    vetor_usuario = montar_vetor_usuario(ratings[ratings["userId"] == USER_ID])  # notas do usuário-alvo no formato de uma linha da matriz
    vizinhos, similaridades = buscar_vizinhos(_modelo_base["knn"], vetor_usuario)  # vizinhos úteis e o peso de cada um

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
