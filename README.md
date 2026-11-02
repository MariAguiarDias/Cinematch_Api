# CineMatch

Sistema de recomendação de filmes desenvolvido como TCC. A pessoa avalia filmes, recebe sugestões de **três algoritmos diferentes** e dá uma nota para cada um deles. Essas notas permitem comparar qual algoritmo agrada mais.

O projeto tem duas partes:

- **Back-end (Python/Flask):** recebe as avaliações, roda os três algoritmos e devolve as sugestões.
- **Front-end (HTML, CSS e JavaScript):** telas para buscar e avaliar filmes, comparar as sugestões e avaliar os algoritmos.

## Como funciona

1. **Avaliar 5 filmes.** A pessoa busca filmes que já viu e dá uma nota de 0,5 a 5, em passos de 0,5. O mínimo de cinco é obrigatório.
2. **Avaliar as sugestões.** O front envia as notas para a API, que devolve 5 filmes de cada algoritmo. A pessoa dá de 1 a 5 estrelas (em passos de 0,5) a cada lista.
3. **Concluir.** As três notas são enviadas e salvas em `avaliacoes_teste.csv`.

No back-end, as notas viram um **usuário novo** acrescentado à base. Os modelos são treinados só com a base original, então o usuário novo nunca altera o que eles aprenderam.

## Algoritmos

| Algoritmo | Arquivo | Como recomenda |
|---|---|---|
| **KNN** (filtragem colaborativa) | `knn.py` | Busca os 20 usuários mais parecidos (similaridade de cosseno) e calcula a nota estimada de cada filme como média das notas dos vizinhos, ponderada pela similaridade. Só considera filmes avaliados por pelo menos 3 vizinhos. |
| **SVD** (matriz fatorizada) | `matriz_fatorizada.py` | Treina um SVD do Surprise (20 fatores latentes, 20 épocas). O perfil do usuário novo (viés e fatores) é ajustado sobre o modelo pronto, sem retreinar. A nota estimada é: média global + viés do usuário + viés do filme + produto dos fatores. |
| **NCF** (rede neural) | `rede_neural.py` | Rede em PyTorch que combina embeddings de usuário e filme (16 dimensões) em camadas densas (32 e 16 neurônios, com dropout). O vetor do usuário novo é aprendido em 100 passos, com o resto da rede congelado. |

Os três devolvem os 5 filmes com maior nota estimada, com colunas extras como `nota_estimada` e `generos_em_comum`. O KNN também devolve `qtd_vizinhos`.

### Filtro de gênero

Depois de estimar as notas, `filtros.py` descarta candidatos que não combinam com o gosto do usuário. O **perfil core** é formado pelos gêneros presentes em pelo menos 60% dos filmes que ele avaliou. O modo é definido em `MODO_FILTRO_GENERO` (`main.py`):

- `"core"`: o filme precisa ter **todos** os gêneros do perfil core.
- `"intersecao"`: o filme precisa ter pelo menos **metade** dos gêneros do perfil core.
- `None`: sem filtro.

## Estrutura do projeto

```
├── main.py                   # API Flask e orquestração dos algoritmos
├── knn.py                    # Algoritmo KNN
├── matriz_fatorizada.py      # Algoritmo SVD
├── rede_neural.py            # Algoritmo NCF (rede neural)
├── filtros.py                # Filtro por perfil de gênero
├── movies_small.csv          # Catálogo de filmes (MovieLens)
├── ratings_small.csv         # Avaliações (MovieLens)
├── requirements.txt          # Dependências do back-end
├── avaliacoes_teste.csv      # Gerado: notas dadas aos algoritmos
│
├── index.html                # Início
├── resultados.html           # Busca de filmes
├── avaliados.html            # Filmes que a pessoa já avaliou
├── filme.html                # Detalhes e avaliação de um filme
├── sugestoes.html            # Comparação dos três algoritmos
├── css/style.css             # Estilos
└── js/                       # app.js, catalogo.js, recomendacoes.js, icons.js, escala.js
```

> O `main.py` importa `matriz_fatorizada`, então o arquivo do SVD precisa ter exatamente esse nome.

## Instalação

Requer Python 3.

```bash
python -m venv venv
source venv/bin/activate        # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

O `scikit-surprise` compila extensões em C. Se a instalação falhar, instale antes as ferramentas de compilação do seu sistema (no Windows, as *Build Tools* do Visual Studio).

## Como executar

### Back-end

```bash
python main.py
```

O servidor sobe na porta `8000` (ou na definida pela variável de ambiente `PORT`).

Para produção, com gunicorn:

```bash
gunicorn main:app --timeout 300
```

Os modelos são treinados na **primeira requisição** a `/recomendacoes` e ficam em memória. Por isso essa primeira resposta demora mais do que as seguintes. Se os CSVs mudarem, tudo é retreinado. O `--timeout` maior evita que o gunicorn derrube o processo durante esse treino.

### Front-end

Sirva a pasta do front por qualquer servidor estático, por exemplo:

```bash
python -m http.server 5500
```

Depois abra `http://localhost:5500`. A URL da API é configurada em `js/recomendacoes.js`, que define a função `comunicacaoApi`. O back-end já libera CORS.

## API

### `POST /recomendacoes`

Recebe as avaliações da pessoa e devolve os IDs sugeridos por cada algoritmo.

```json
{
  "avaliacoes": {
    "1":   { "nota": 4.5 },
    "296": { "nota": 5.0 },
    "318": { "nota": 3.5 },
    "356": { "nota": 4.0 },
    "593": { "nota": 2.5 }
  }
}
```

A chave é o `movieId`. Resposta:

```json
{
  "knn": [2571, 1196, 260, 1210, 2959],
  "svd": [858, 50, 1221, 912, 1213],
  "ncf": [527, 1089, 1136, 2028, 1193]
}
```

**Regras de validação:**

- No mínimo 5 filmes.
- Nenhum filme repetido.
- Todos os `movieId` precisam existir em `movies_small.csv`.
- Notas entre 0,5 e 5,0, em passos de 0,5.

### `POST /avaliacoes`

Recebe a nota que a pessoa deu a cada algoritmo e acrescenta uma linha em `avaliacoes_teste.csv` (colunas `knn`, `svd`, `ncf`).

```json
{ "avaliacoes": { "knn": 4.5, "svd": 3.5, "ncf": 4.0 } }
```

Respostas: `200` com `{"mensagem": "..."}` em caso de sucesso, `400` se nenhuma avaliação for enviada e `500` se houver falha ao salvar.

## Configuração

As constantes ficam no topo de cada arquivo.

| Constante | Arquivo | Padrão | O que controla |
|---|---|---|---|
| `MIN_AVALIACOES` | `main.py` | `5` | Mínimo de filmes avaliados pela pessoa |
| `CORE_GENERO_MIN_FRACAO` | `main.py` | `0.6` | Fração de filmes em que um gênero precisa aparecer para entrar no perfil core |
| `MODO_FILTRO_GENERO` | `main.py` | `"core"` | `"core"`, `"intersecao"` ou `None` |
| `CALCULAR_METRICAS` | `main.py` | `False` | Mede RMSE e MAE dos algoritmos a cada requisição |
| `PROPORCAO_TESTE` | `main.py` | `0.2` | Parte da base reservada para teste nas métricas |
| `N_RECOMENDACOES` | `knn.py`, `matriz_fatorizada.py`, `rede_neural.py` | `5` | Filmes na lista de cada algoritmo |
| `K_VIZINHOS`, `SUPORTE_MINIMO` | `knn.py` | `20`, `3` | Vizinhos considerados e mínimo de vizinhos por filme |
| `PARAMETROS_SVD` | `matriz_fatorizada.py` | ver arquivo | Fatores, aprendizado, regularização, épocas e semente |
| `EPOCAS`, `DIM_EMBEDDING`, `CAMADAS_OCULTAS` etc. | `rede_neural.py` | ver arquivo | Hiperparâmetros da rede |

## Métricas

Com `CALCULAR_METRICAS = True`, cada requisição também mede o desempenho dos três algoritmos. A base original é dividida em 80% para treino e 20% para teste, com semente fixa, e o terminal mostra uma tabela com:

- **RMSE** e **MAE** de cada algoritmo;
- **fração prevista pelo modelo**, a parte do teste em que usuário e filme existiam no treino (nos demais casos o algoritmo cai para uma média);
- tempo de recomendação e tempo de avaliação.

Isso treina cada algoritmo mais uma vez, então deixa a execução alguns segundos mais lenta.

## Dados

O projeto usa o conjunto **MovieLens Small**, do GroupLens: 100.836 avaliações de 610 usuários sobre 9.742 filmes. Escala de notas de 0,5 a 5, em passos de 0,5.

> F. Maxwell Harper e Joseph A. Konstan. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems, 2015.
