# Rota Livre — IA (Generative AI for Engineering)

Componente de **Inteligência** da CitizenTech *Rota Livre* — Global Solution 2026.2 · FIAP · 4ES.

Uma **LLM local** (sem nuvem, sem CUDA) que responde dúvidas do cidadão sobre **buracos e
obstáculos em vias e calçadas de São Paulo** com base em fontes oficiais, cita a fonte,
responde em **JSON válido** e **recusa com "não consta"** o que a base não cobre — além de
fazer a **triagem** de relatos para o MVP. Provamos no placar que a IA com contexto
**acerta mais e alucina menos** que o mesmo modelo sem contexto.

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Gomes Catanzaro | RM93445 |
| Lucas Gomes Alcantara | RM98766 |
| Filipe Prado Menezes | RM98765 |

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

Com o `llama-server` e a API no ar, abra **http://127.0.0.1:8000**. A tela mostra, lado a lado, a resposta das
3 configurações (sem contexto, RAG e agente) para a mesma pergunta, com a fonte citada, a confiança e os passos
do agente, e faz a triagem de um relato livre em JSON.

### 5. API

```
GET  /health              → {"api": "ok", "llm_local": true}
POST /perguntar           → {"pergunta": "...", "modo": "agente"}
POST /triagem             → {"texto": "relato livre do cidadão"}
```

A resposta de `/perguntar` traz o JSON `{resposta, fonte_id, confianca}`, a fonte por extenso
e o log de passos do agente. Integração com o MVP: o app (repositório
`tapa-buraco-devops`) chama `/triagem` ao receber um relato e `/perguntar` no chat de dúvidas.

## Resultados

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
