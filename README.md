# Ingestão e Busca Semântica com LangChain e Postgres

Implementação do desafio **Ingestão e Busca Semântica com LangChain e Postgres** do MBA em Engenharia de Software com IA da Full Cycle.

## Minha implementação

CLI em Python que responde perguntas usando só o conteúdo de um PDF: ingestão em Postgres com pgVector e busca semântica com Gemini. As decisões que valem registro:

- **Ingestão que pode rodar de novo.** Cada trecho recebe um ID determinístico (UUID v5 de coleção, arquivo e posição). Reexecutar o mesmo PDF atualiza os registros em vez de duplicá-los.
- **Um embedding por trecho.** A classe `IndividualGoogleEmbeddings` chama a API trecho a trecho, sem depender do endpoint em lote, e mostra o progresso a cada 10 trechos.
- **Tipos de tarefa separados.** Os trechos são embedados como `RETRIEVAL_DOCUMENT` e a pergunta como `RETRIEVAL_QUERY`, que é o par recomendado para busca.
- **Chunks de 1000 caracteres com overlap de 150**, guardando o índice de início e a página de origem. O contexto enviado ao modelo traz `[Página N]` antes de cada trecho.
- **Resposta presa ao contexto.** Os 10 trechos mais próximos (`k=10`) entram no prompt do enunciado; fora do contexto, a resposta é a frase fixa "Não tenho informações necessárias para responder sua pergunta."
- **Falha explícita.** Variáveis de ambiente obrigatórias são validadas antes de qualquer chamada, e a busca sem trechos ingeridos manda rodar a ingestão. O modelo de chat tem timeout de 60 s e 2 tentativas.
- **Banco só local.** O Postgres sobe em `127.0.0.1:55432`, e um serviço auxiliar do Compose cria a extensão `vector` e termina.

Código em `src/`: `ingest.py` (ingestão), `search.py` (busca e prompt) e `chat.py` (CLI).

## Configuração

Copie `.env.example` para `.env`. Preencha `GOOGLE_API_KEY` e
`POSTGRES_PASSWORD`. Use a mesma senha em `DATABASE_URL`:

```text
DATABASE_URL=postgresql+psycopg://postgres:SUA_SENHA@127.0.0.1:55432/rag
```

O PostgreSQL fica disponível em `127.0.0.1:55432`. O `.env` não entra no Git.
Mantenha `GOOGLE_EMBEDDING_MODEL` igual na ingestão e na busca. O modelo de
resposta vem de `GOOGLE_CHAT_MODEL`; o exemplo usa `gemini-3.6-flash`.

Cada pessoa deve usar sua própria chave. A cota gratuita do Gemini é aplicada
ao projeto Google dessa chave, não ao repositório. Se a API retornar `429` por
limite diário, aguarde a renovação da cota. Consulte os
[limites da API](https://ai.google.dev/gemini-api/docs/rate-limits).

## Ambiente Python

Validado com Python 3.12:

```
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução

Se você usa Colima no macOS, inicie-o antes do Compose:

```
colima start
```

1. Inicie o PostgreSQL:

```
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria a extensão `vector` e termina com código 0.
Confira se o PostgreSQL está saudável e se a extensão foi criada:

```
docker compose ps --all
docker compose exec -T postgres psql -U postgres -d rag -Atc "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

2. Ingira o PDF:

```
python src/ingest.py
```

A ingestão faz uma chamada de embedding por trecho. Reexecutar o mesmo PDF
atualiza os registros, sem duplicá-los.

3. Inicie o chat:

```
python src/chat.py
```

Pergunte sobre o PDF. Digite `sair` para encerrar. A cada pergunta, a busca
recupera até 10 trechos para compor o contexto da resposta.

<details>
<summary>Enunciado do desafio (MBA Full Cycle)</summary>

## Objetivo

Você deve entregar um software capaz de:

- Ingestão: Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.
- Busca: Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Exemplo no CLI

Faça sua pergunta:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Perguntas fora do contexto:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Tecnologias obrigatórias

- Linguagem: Python
- Framework: LangChain
- Banco de dados: PostgreSQL + pgVector
- Execução do banco de dados: Docker & Docker Compose (docker-compose fornecido no repositório de exemplo)

## Pacotes recomendados

- Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
- Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
- PDF: `from langchain_community.document_loaders import PyPDFLoader`
- Ingestão: `from langchain_postgres import PGVector`
- Busca: `similarity_search_with_score(query, k=10)`

## OpenAI

- Crie uma API Key da OpenAI.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial da OpenAI para ver os modelos disponíveis.

## Gemini

- Crie uma API Key da Google.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial do Google para ver os modelos disponíveis.

Os limites de requisições gratuitas dos modelos podem mudar com frequência. Para informações atualizadas, consulte a documentação oficial do Google.

## Escolha dos modelos

Este desafio não fixa modelos. Nomes e versões mudam com frequência e alguns são descontinuados, então faz parte do desafio consultar a documentação oficial do provedor que você escolher, ver quais modelos estão disponíveis no momento e selecionar os que atendem ao objetivo. Para o volume deste desafio, os modelos mais leves e baratos de cada provedor são suficientes.

Atenção: modelos de embedding diferentes geram vetores com dimensões diferentes. A tabela de vetores é criada na primeira ingestão, já com a dimensão do modelo que você escolheu. Se você trocar de modelo de embeddings depois disso, a ingestão passa a falhar por incompatibilidade de dimensão. Nesse caso é responsabilidade sua apagar a collection existente (ou o volume do banco) e refazer a ingestão do zero com o novo modelo.

## Requisitos

### 1. Ingestão do PDF

- O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150.
- Cada chunk deve ser convertido em embedding.
- Os vetores devem ser armazenados no banco de dados PostgreSQL com pgVector.

### 2. Consulta via CLI

Criar um script Python para simular um chat no terminal.

Passos ao receber uma pergunta:

- Vetorizar a pergunta.
- Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
- Montar o prompt e chamar a LLM.
- Retornar a resposta ao usuário.

Prompt a ser utilizado:

```
CONTEXTO:
{resultados concatenados do banco de dados}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Estrutura obrigatória do projeto

Faça um fork do repositório para utilizar a estrutura abaixo: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## Entregável

Repositório público no GitHub contendo todo o código-fonte e README com instruções claras de execução do projeto.

</details>
