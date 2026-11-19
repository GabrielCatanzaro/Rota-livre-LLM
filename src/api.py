"""API do assistente Rota Livre (FastAPI).

Expõe a IA para o MVP:
- POST /perguntar  → resposta fundamentada em JSON (modo: sem_contexto | rag | agente)
- POST /triagem    → transforma o relato livre do cidadão em um JSON estruturado
- GET  /health     → status da API e da LLM local
- POST /rota       → rota com menos buracos (modo: sem_contexto | rag | agente) — dados simulados
- GET  /rotas/locais → origens/destinos conhecidos na base de rotas
- GET  /           → tela de demonstração (perguntas, rotas e triagem)

Rodar:  uvicorn src.api:app --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import llm, rotas
from .busca import Busca
from .configuracoes import CONFIGURACOES

app = FastAPI(title="Rota Livre - IA", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

busca = Busca()


class Pergunta(BaseModel):
    pergunta: str
    modo: str = "agente"  # sem_contexto | rag | agente


class Relato(BaseModel):
    texto: str


@app.get("/", response_class=HTMLResponse)
def demo():
    """Tela de demonstração: pergunta nas 3 configurações lado a lado + triagem."""
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"api": "ok", "llm_local": llm.disponivel()}


@app.post("/perguntar")
def perguntar(p: Pergunta):
    fn = CONFIGURACOES.get(p.modo, CONFIGURACOES["agente"])
    r = fn(p.pergunta, busca)
    trecho = busca.por_id((r["json"] or {}).get("fonte_id") or "")
    return {
        "modo": r["config"],
        "resposta": r["json"],
        "fonte": trecho["fonte"] if trecho else None,
        "trechos_usados": r["trechos_usados"],
        "passos": r["passos"],
    }


TRIAGEM_SISTEMA = (
    "Você faz a triagem de relatos de problemas em vias e calçadas de São Paulo. "
    "Dado o relato do cidadão, devolva SOMENTE um JSON válido no formato: "
    '{"tipo": "buraco_via|buraco_corredor_onibus|calcada|obstaculo|iluminacao|outro", '
    '"gravidade": "baixa|media|alta", "descricao_curta": "...", '
    '"orgao_sugerido": "Subprefeitura|SPTrans|Responsável pelo imóvel|outro", '
    '"local_citado": "..." } '
    "Regras: corredor/faixa de ônibus -> SPTrans; calçada -> Responsável pelo imóvel; "
    "via comum -> Subprefeitura. gravidade alta = risco imediato de acidente. "
    "Se o relato não citar local, use \"local_citado\": null. Não invente dados."
)


@app.post("/triagem")
def triagem(r: Relato):
    saida = llm.conversar([
        {"role": "system", "content": TRIAGEM_SISTEMA},
        {"role": "user", "content": r.texto},
    ])
    obj = llm.extrair_json(saida)
    if obj is None:
        obj = {"tipo": "outro", "gravidade": "media",
               "descricao_curta": r.texto[:120], "orgao_sugerido": "Subprefeitura",
               "local_citado": None, "observacao": "triagem automática indisponível"}
    return {"triagem": obj}


class PedidoRota(BaseModel):
    pergunta: str
    modo: str = "agente"


@app.get("/rotas/locais")
def rotas_locais():
    return {"locais": [{"id": k, "nome": v["nome"]} for k, v in rotas.LOCAIS.items()],
            "aviso": rotas.DADOS["aviso"]}


@app.post("/rota")
def rota(p: PedidoRota):
    r = rotas.responder(p.pergunta, p.modo)
    return {"modo": r["config"], "resposta": r["json"], "rota": r["rota"],
            "ranking": r["ranking"], "passos": r["passos"]}
