# Rota Livre — IA (Generative AI for Engineering)

Componente de **Inteligência** da CitizenTech *Rota Livre* — Global Solution 2026.2 · FIAP · 4ES.

Uma LLM que ajuda **pessoas com mobilidade reduzida (cadeirantes, idosos,
pessoas com andador ou muletas)** a circular a pé por São Paulo **sem barreiras**:

1. **Rota acessível:** dado um trajeto (ex.: Metrô Sé → Theatro Municipal), descarta os caminhos com escada,
   travessia sem rampa, ladeira acima de 8,33% (NBR 9050) ou calçada com menos de 0,90 m livres, e recomenda o
   caminho acessível com **menos buracos e obstáculos** por 100 m, explicando por quê.
   *Base de calçadas simulada para o MVP.*
2. **Dúvidas com fonte oficial:** responde sobre buracos, calçadas, SP156 e indenização com base em 36 trechos de
   fontes oficiais (CTB, CF/88, Lei 15.442/2011, SP156), citando a fonte.
3. **Triagem de relatos:** transforma o relato livre do cidadão em JSON (tipo, gravidade, órgão responsável).

Em tudo, a IA responde em **JSON válido** e **recusa com "não consta"** o que a base não cobre. Os placares
provam que a IA com contexto **acerta mais e alucina menos** que o mesmo modelo sem contexto.

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Gomes Catanzaro | RM93445 |
| Lucas Gomes Alcantara | RM98766 |
| Henrique Canali Cuzato | RM551326 |

## Arquitetura

```
pergunta do cidadão
      │
      ▼
┌─────────────┐   top-3 trechos    ┌──────────────────┐
│ Busca TF-IDF │ ─────────────────▶ │  LLM local (3B)   │
│ (36 trechos  │   score < 0.15 →   │  llama.cpp Vulkan │──▶ JSON {resposta,
│  com fonte)  │   recusa direta    │  GPU AMD RX 6600  │         fonte_id,
└─────────────┘                    └──────────────────┘         confianca}
                 agente: reformula a busca, valida o JSON,
                 confere a citação e faz autocrítica antes de entregar
```

Três configurações comparadas no placar: **sem_contexto** (linha de base) · **rag** · **agente**
(especialização escolhida no lugar do LoRA — roda 100% local em GPU AMD).

## Estrutura do repositório

```
base/        base_conhecimento.json (36 trechos com fonte) + fontes.md
avaliacao/   casos_teste.json (27 casos com gabarito, incl. "não consta"),
             placar.md e resultados.json (gerados)
notebooks/   01 base e busca (recall@1/@3, limiar) · 02 configurações · 03 placar
src/         busca.py · llm.py · configuracoes.py · avaliar.py · api.py
RELATORIO.md relatório da entrega
```

## Como reproduzir (Windows, GPU AMD, sem CUDA/WSL)

### 1. LLM local (llama.cpp com Vulkan)

1. Baixe o zip `llama-<versão>-bin-win-vulkan-x64.zip` em
   <https://github.com/ggml-org/llama.cpp/releases> e extraia (ex.: `H:\llama.cpp`).
2. Baixe o modelo **Qwen2.5-3B-Instruct** quantizado (~2 GB):
   <https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF> →
   arquivo `Qwen2.5-3B-Instruct-Q4_K_M.gguf` (ex.: em `H:\modelos`).
3. Suba o servidor (API compatível com OpenAI na porta 8080):

```powershell
H:\llama.cpp\llama-server.exe -m H:\modelos\Qwen2.5-3B-Instruct-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

`-ngl 99` manda todas as camadas para a GPU (RX 6600 via Vulkan).
Teste em <http://localhost:8080> — há uma interface de chat embutida.

> **Porta 8080 ocupada?** (ex.: software de backup usando a porta) Suba com `--port 8081` e
> aponte o projeto para ela antes de rodar notebooks, scripts ou a API:
> `$env:LLM_URL = "http://localhost:8081/v1/chat/completions"`

### 2. Ambiente Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rodar

```powershell
# notebooks (execute na ordem 01 → 02 → 03 e salve com as saídas)
# ou direto pela linha de comando:
python -m src.busca "como reporto um buraco?"   # busca
python -m src.configuracoes "quem conserta a calçada?"  # 3 configurações
python -m src.avaliar                            # placar completo (alguns minutos)

# API usada pelo MVP
uvicorn src.api:app --port 8000
```

### 4. Tela de demonstração

Com o `llama-server` e a API no ar, abra **http://127.0.0.1:8000**:
Tela pensada para acessibilidade (letras grandes, alto contraste, botões grandes, navegação por teclado):
- **Rota acessível:** escolha "Saindo de" e "Chegando em". O resultado do agente aparece em destaque
  (caminho passo a passo, metros, buracos, obstáculos), seguido das **rotas descartadas para cadeira de rodas**
  com o motivo (escada, sem rampa, ladeira). Em "Como a IA chegou nisso?" ficam as 3 configurações lado a lado.
- **Tirar dúvida:** pergunta sobre calçadas e direitos, respondida com fonte oficial (ou "não consta").
- **Relatar problema:** um relato livre vira tipo, gravidade, órgão responsável e local.

### 5. API

```
GET  /health              → {"api": "ok", "llm_local": true}
POST /perguntar           → {"pergunta": "...", "modo": "agente"}
POST /triagem             → {"texto": "relato livre do cidadão"}
GET  /rotas/locais        → locais disponíveis na base de calçadas
POST /rota                → {"pergunta": "Sou cadeirante. Caminho acessível da Sé ao Theatro Municipal?", "modo": "agente"}
```

A resposta de `/perguntar` traz o JSON `{resposta, fonte_id, confianca}`, a fonte por extenso
e o log de passos do agente. Integração com o MVP: o app (repositório
`tapa-buraco-devops`) chama `/triagem` ao receber um relato e `/perguntar` no chat de dúvidas.

## Resultados

### Rota acessível (notebook 04, dados simulados)

Placar em [`avaliacao/rotas_placar.md`](avaliacao/rotas_placar.md) — 12 casos (8 trajetos da base + 4 fora da base).
Base: `base/calcadas_simuladas.json` (28 trechos de calçada, 12 locais, 18 rotas). Em 4 dos 6 trajetos o caminho
mais curto é **bloqueado** para cadeira de rodas, então a métrica mais importante é **"recomendou rota bloqueada"**
(quanto menor, melhor). O agente usa uma **ferramenta** que descarta as rotas bloqueadas e ordena as acessíveis
pelo índice; a LLM só explica. O RAG precisa fazer isso sozinho; sem contexto, o modelo não tem os dados.

### Dúvidas com fonte oficial (notebooks 01 a 03)

| Configuração | Acurácia | Alucinação ↓ | JSON válido | Citação correta | Tempo médio |
|---|---|---|---|---|---|
| sem_contexto | 26% | 20% | 100% | 0% | 2,6 s |
| rag | **74%** | **0%** | 100% | 84% | 2,9 s |
| agente | 70% | **0%** | 100% | **100%** | 4,8 s |

Busca: **recall@1 = 95,5% · recall@3 = 100%**. Análise completa e os 5 erros no [RELATORIO.md](RELATORIO.md).

## Segurança, explicabilidade e LGPD

- **Explicável**: toda resposta cita o trecho (`fonte_id`) e o log de passos do agente mostra
  busca, scores, validações e autocrítica.
- **Recusa**: perguntas fora da base recebem `"não consta"` — limiar de busca + regra no prompt.
- **LGPD**: a base só tem conteúdo público; a API não coleta dados pessoais; a triagem instrui
  o modelo a não inventar dados e o MVP não exige identificação do cidadão (minimização).
- **Local**: nenhum dado sai da máquina — o modelo roda offline.

## Links

- Repositório: https://github.com/GabrielCatanzaro/tapa-buraco-ia
- Vídeo: [PREENCHER]
- MVP e deploy (componente Produção): repositório `tapa-buraco-devops`
