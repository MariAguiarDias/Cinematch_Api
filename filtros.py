import math

FRACAO_MIN_INTERSECAO = 0.5 # no modo "intersecao", fração mínima do perfil core que o filme precisa ter

# Mantém só os candidatos compatíveis com o perfil core do usuário, conforme o modo definido no main
def aplicar_filtro_genero(pontuacao_final, calculos):

    modo = calculos.get("modo_filtro_genero")  # "core", "intersecao" ou None
    perfil_core = calculos["perfil_core_usuario"]  # gêneros que definem o gosto do usuário
    genero_por_filme = calculos["genero_por_filme"]  # conjunto de gêneros de cada filme

    # Sem modo ou sem perfil core definido, todos os candidatos seguem adiante
    if modo is None or not perfil_core:
        return pontuacao_final

    # "core" exige todos os gêneros do perfil e "intersecao" exige só parte deles
    if modo == "core":
        minimo_em_comum = len(perfil_core)  # todos os gêneros do perfil core
    elif modo == "intersecao":
        minimo_em_comum = math.ceil(len(perfil_core) * FRACAO_MIN_INTERSECAO)  # pelo menos metade do perfil core (arredondado para cima)
    else:
        raise ValueError(f"Modo de filtro de gênero desconhecido: {modo}")

    return {
        movie_id: pontuacao
        for movie_id, pontuacao in pontuacao_final.items()
        if len(perfil_core & genero_por_filme.get(int(movie_id), set())) >= minimo_em_comum
    }
