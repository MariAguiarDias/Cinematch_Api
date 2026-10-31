import numpy as np
import torch
from torch import nn

from filtros import aplicar_filtro_genero

N_RECOMENDACOES = 5  # quantidade de filmes na lista final
ESCALA_NOTAS = (0.5, 5.0)  # escala real das notas do MovieLens
DIM_EMBEDDING = 16  # quantidade de números do vetor que representa cada usuário e cada filme
CAMADAS_OCULTAS = (32, 16)  # neurônios de cada camada densa, da primeira para a última
TAXA_DROPOUT = 0.2  # fração de neurônios desligados a cada passo do treino, para evitar decorar os dados
TAXA_APRENDIZADO = 0.003  # tamanho de cada ajuste dos pesos
REGULARIZACAO = 0.0001  # penalidade para pesos muito grandes
EPOCAS = 10  # quantas vezes a rede percorre todas as avaliações no treino
TAMANHO_LOTE = 1024  # avaliações usadas em cada ajuste dos pesos
SEMENTE = 42  # semente que torna o treino reproduzível


# Rede que combina os embeddings de usuário e filme em camadas densas para prever a nota
class ModeloNCF(nn.Module):

    # Cria os embeddings e as camadas densas da rede
    def __init__(self, n_usuarios, n_filmes):
        super().__init__()
        self.emb_usuario = nn.Embedding(n_usuarios, DIM_EMBEDDING)  # vetor aprendido de cada usuário
        self.emb_filme = nn.Embedding(n_filmes, DIM_EMBEDDING)  # vetor aprendido de cada filme

        camadas = []  # sequência de camadas densas, cada uma com ativação ReLU e dropout
        tamanho_entrada = 2 * DIM_EMBEDDING  # os dois embeddings concatenados

        for tamanho_saida in CAMADAS_OCULTAS:
            camadas += [nn.Linear(tamanho_entrada, tamanho_saida), nn.ReLU(), nn.Dropout(TAXA_DROPOUT)]
            tamanho_entrada = tamanho_saida
        camadas.append(nn.Linear(tamanho_entrada, 1))

        self.camadas_densas = nn.Sequential(*camadas)  # parte da rede que aprende a combinar usuário e filme

    # Calcula a nota prevista para cada par (usuário, filme) recebido
    def forward(self, usuarios, filmes):
        entrada = torch.cat([self.emb_usuario(usuarios), self.emb_filme(filmes)], dim=1)  # embeddings do usuário e do filme lado a lado
        saida = self.camadas_densas(entrada).squeeze(1)  # valor bruto produzido pela última camada

        # A sigmoide limita a saída entre 0 e 1, e a conta seguinte a leva para a escala de notas
        nota_min, nota_max = ESCALA_NOTAS  # limites da escala
        return nota_min + (nota_max - nota_min) * torch.sigmoid(saida)


# Converte os IDs reais em posições sequenciais, que são os índices usados pelos embeddings
def mapear_ids(ratings):
    pos_usuario = {uid: pos for pos, uid in enumerate(ratings["userId"].unique())}  # ID real do usuário para a posição no embedding
    pos_filme = {mid: pos for pos, mid in enumerate(ratings["movieId"].unique())}  # ID real do filme para a posição no embedding
    return pos_usuario, pos_filme


# Treina a rede com todas as avaliações recebidas e devolve o modelo junto com os mapas de IDs
def treinar_rede_neural(ratings):
    torch.manual_seed(SEMENTE)
    pos_usuario, pos_filme = mapear_ids(ratings)  # índices de usuários e filmes nos embeddings

    usuarios = torch.tensor(ratings["userId"].map(pos_usuario).to_numpy(), dtype=torch.long)  # posição do usuário de cada avaliação
    filmes = torch.tensor(ratings["movieId"].map(pos_filme).to_numpy(), dtype=torch.long)  # posição do filme de cada avaliação
    notas = torch.tensor(ratings["rating"].to_numpy(), dtype=torch.float32)  # nota real de cada avaliação

    modelo = ModeloNCF(len(pos_usuario), len(pos_filme))  # rede com pesos iniciais aleatórios
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=REGULARIZACAO)  # algoritmo que ajusta os pesos
    gerador = torch.Generator().manual_seed(SEMENTE)  # embaralhamento reproduzível das avaliações

    modelo.train()
    for _ in range(EPOCAS):
        ordem = torch.randperm(len(notas), generator=gerador)  # ordem aleatória das avaliações nesta época

        # Cada lote ajusta os pesos na direção que reduz o erro quadrático entre nota prevista e nota real
        for inicio in range(0, len(notas), TAMANHO_LOTE):
            lote = ordem[inicio:inicio + TAMANHO_LOTE]  # posições das avaliações deste lote
            otimizador.zero_grad()
            erro = nn.functional.mse_loss(modelo(usuarios[lote], filmes[lote]), notas[lote])  # erro médio do lote
            erro.backward()
            otimizador.step()

    modelo.eval()
    return modelo, pos_usuario, pos_filme


# Prevê as notas de vários pares (usuário, filme) de uma vez, sem calcular gradientes
def prever_notas(modelo, posicoes_usuarios, posicoes_filmes):
    with torch.no_grad():
        return modelo(
            torch.tensor(posicoes_usuarios, dtype=torch.long),
            torch.tensor(posicoes_filmes, dtype=torch.long),
        ).numpy()


# Roda a rede neural (NCF) para recomendar filmes para o usuário USER_ID
def rodar_rede_neural(movies, ratings, USER_ID, calculos):

    watched_ids = calculos['watched_ids']  # filmes que o usuário já avaliou
    genero_por_filme = calculos['genero_por_filme']  # conjunto de gêneros de cada filme
    perfil_generos_usuario = calculos['perfil_generos_usuario']  # todos os gêneros do histórico do usuário

    # Todas as notas do usuário-alvo entram no treino para que nenhuma parte do perfil seja descartada
    modelo, pos_usuario, pos_filme = treinar_rede_neural(ratings)  # rede treinada com a base completa e o usuário-alvo

    # Filmes sem nenhuma avaliação não têm embedding treinado, então ficam de fora da lista de candidatos
    candidatos = [mid for mid in movies["movieId"] if mid in pos_filme and mid not in watched_ids]  # IDs dos filmes ainda não vistos

    # Gera as notas estimadas
    notas_previstas = prever_notas(  
        # nota prevista para cada candidato, na mesma ordem
        modelo,
        [pos_usuario[USER_ID]] * len(candidatos),
        [pos_filme[mid] for mid in candidatos],
    )
    pontuacao_final = dict(zip(candidatos, notas_previstas.astype(float)))  # nota estimada de cada filme candidato

    # Filtra os candidatos pelo perfil de gênero do usuário
    pontuacao_final = aplicar_filtro_genero(pontuacao_final, calculos)  # Se o candidato tiver o perfil core do usuário, ele é mantido

    # Seleciona os top N_RECOMENDACOES filmes recomendados
    top_ids = sorted(pontuacao_final, key=pontuacao_final.get, reverse=True)[:N_RECOMENDACOES]  # IDs dos filmes com maior nota estimada

    # Um perfil core muito restrito pode deixar menos candidatos do que o necessário
    if len(top_ids) < N_RECOMENDACOES:
        print(f"Aviso: a rede neural encontrou apenas {len(top_ids)} candidatos após os filtros.")

    # Monta o DataFrame final de recomendações para retornar
    recomendados = movies[movies["movieId"].isin(top_ids)].copy()  # dados dos filmes escolhidos
    recomendados["nota_estimada"] = recomendados["movieId"].map(pontuacao_final)
    recomendados["generos_em_comum"] = recomendados["movieId"].apply(
        lambda mid: sorted(genero_por_filme.get(int(mid), set()) & perfil_generos_usuario)
    )
    recomendados = recomendados.sort_values("nota_estimada", ascending=False).reset_index(drop=True)  # lista da maior para a menor nota estimada

    return recomendados


# Mede RMSE e MAE da rede neural no teste, usando a média do usuário quando o par não existe no treino
def avaliar_rede_neural(treino, teste):

    modelo, pos_usuario, pos_filme = treinar_rede_neural(treino)  # rede treinada só com o treino

    # Médias usadas como previsão de reserva para usuários ou filmes ausentes do treino
    media_global = treino["rating"].mean()  # média de todas as notas do treino
    media_usuario = treino.groupby("userId")["rating"].mean()  # média de cada usuário no treino

    conhecidos = (teste["userId"].isin(pos_usuario) & teste["movieId"].isin(pos_filme)).to_numpy()  # True para pares com usuário e filme no treino
    previsoes = teste["userId"].map(media_usuario).fillna(media_global).to_numpy(copy=True)  # começa com a previsão de reserva em todos os pares

    previsoes[conhecidos] = prever_notas(
        modelo,
        teste.loc[conhecidos, "userId"].map(pos_usuario).to_numpy(),
        teste.loc[conhecidos, "movieId"].map(pos_filme).to_numpy(),
    )

    erros = previsoes - teste["rating"].to_numpy()  # diferença entre previsão e nota real em cada par

    return {
        "rmse": float(np.sqrt(np.mean(erros ** 2))),
        "mae": float(np.mean(np.abs(erros))),
        "fracao_prevista_pelo_modelo": float(conhecidos.mean()),
    }
