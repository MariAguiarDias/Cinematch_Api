from surprise import Dataset, Reader, SVD
from surprise import accuracy

from filtros import aplicar_filtro_genero

N_RECOMENDACOES = 5  # quantidade de filmes na lista final
ESCALA_NOTAS = (0.5, 5.0) # escala real das notas do MovieLens
PARAMETROS_SVD = dict(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)  # fatores latentes, aprendizado, regularização, épocas e semente


# Treina o SVD com todas as avaliações recebidas, sem separar teste
def treinar_svd(ratings):

    # Configura o leitor e carrega os dados para o formato do Surprise
    reader = Reader(rating_scale=ESCALA_NOTAS)  # informa ao Surprise a escala das notas
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)  # avaliações no formato do Surprise

    # Inicia e treina o modelo SVD
    model = SVD(**PARAMETROS_SVD)  # modelo de fatoração de matrizes
    model.fit(data.build_full_trainset())

    return model


# Mede RMSE e MAE do SVD no conjunto de teste recebido do main
def avaliar_svd(treino, teste):

    # Treina com o trainset e monta o testset, vindos da divisão 80/20 feita no main
    model = treinar_svd(treino)  # modelo treinado só com o treino
    testset = list(teste[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))  # avaliações de teste no formato (usuário, filme, nota)

    # Avalia o modelo com métricas RMSE e MAE no conjunto de teste
    predictions = model.test(testset)  # previsão do modelo para cada avaliação de teste

    # Fora do treino, o SVD prevê só com média global e vieses, sem usar os fatores latentes
    conhecidos = sum(1 for uid, iid, _ in testset if _esta_no_treino(model.trainset, uid, iid))  # pares do teste com usuário e filme presentes no treino

    return {
        "rmse": accuracy.rmse(predictions, verbose=False),
        "mae": accuracy.mae(predictions, verbose=False),
        "fracao_prevista_pelo_modelo": conhecidos / len(testset),
    }


# Indica se usuário e filme existem no treino, ou seja, se têm fatores latentes aprendidos
def _esta_no_treino(trainset, uid, iid):
    try:
        trainset.to_inner_uid(uid)
        trainset.to_inner_iid(iid)
        return True
    except ValueError:
        return False


# Roda a Fatoração de Matrizes (SVD) para recomendar filmes para o usuário USER_ID
def rodar_svd(movies, ratings, USER_ID, calculos):

    watched_ids = calculos['watched_ids']  # filmes que o usuário já avaliou
    genero_por_filme = calculos['genero_por_filme']  # conjunto de gêneros de cada filme
    perfil_generos_usuario = calculos['perfil_generos_usuario']  # todos os gêneros do histórico do usuário

    # Todas as notas do usuário-alvo entram no treino para que nenhuma parte do perfil seja descartada
    model = treinar_svd(ratings)  # modelo treinado com a base completa e o usuário-alvo

    # Filtra os filmes que o usuário já viu
    candidatos = movies.loc[~movies["movieId"].isin(watched_ids), "movieId"]  # IDs dos filmes ainda não vistos

    # Gera as notas estimadas
    pontuacao_final = {  # nota estimada de cada filme candidato
        movie_id: model.predict(USER_ID, movie_id).est
        for movie_id in candidatos
    }

    # Filtra os candidatos pelo perfil de gênero do usuário
    pontuacao_final = aplicar_filtro_genero(pontuacao_final, calculos)  # Se o candidato tiver o perfil core do usuário, ele é mantido

    # Seleciona os top N_RECOMENDACOES filmes recomendados
    top_ids = sorted(pontuacao_final, key=pontuacao_final.get, reverse=True)[:N_RECOMENDACOES]  # IDs dos filmes com maior nota estimada

    # Um perfil core muito restrito pode deixar menos candidatos do que o necessário
    if len(top_ids) < N_RECOMENDACOES:
        print(f"Aviso: o SVD encontrou apenas {len(top_ids)} candidatos após os filtros.")

    # Monta o DataFrame final de recomendações para retornar
    recomendados = movies[movies["movieId"].isin(top_ids)].copy()  # dados dos filmes escolhidos
    recomendados["nota_estimada"] = recomendados["movieId"].map(pontuacao_final)
    recomendados["generos_em_comum"] = recomendados["movieId"].apply(
        lambda mid: sorted(genero_por_filme.get(int(mid), set()) & perfil_generos_usuario)
    )
    recomendados = recomendados.sort_values("nota_estimada", ascending=False).reset_index(drop=True)  # lista da maior para a menor nota estimada

    return recomendados
