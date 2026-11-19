"""As três configurações comparadas no placar.

1. sem_contexto  — o modelo responde sozinho, sem acesso à base (linha de base).
2. rag           — RAG simples: top-3 trechos no prompt + regra de recusa.
3. agente        — especialização por agente: busca com reformulação de
                   consulta, validação do JSON, verificação da citação e
                   autocrítica antes de entregar. (Escolhemos agente em vez de
                   LoRA: roda 100% local em GPU AMD, sem treino em nuvem.)

Todas devolvem um dicionário com o JSON final e um log de passos.
"""

import json

from . import llm
from .busca import Busca, LIMIAR

RECUSA = {"resposta": "não consta", "fonte_id": None, "confianca": 0.0}

SISTEMA = (
    "Você é o assistente do Rota Livre, um serviço sobre buracos e obstáculos "
    "em vias e calçadas do município de São Paulo. Responda SEMPRE em português "
    "e SOMENTE com um objeto JSON válido, sem texto fora do JSON, no formato: "
    '{"resposta": "...", "fonte_id": "...", "confianca": 0.0}'
)

REGRAS_CONTEXTO = (
    "Regras: use APENAS as informações dos trechos fornecidos. Em 'fonte_id' "
    "coloque o id do trecho usado (ex.: \"ctb-02\"). Se os trechos não "
    "responderem à pergunta, devolva exatamente "
    '{"resposta": "não consta", "fonte_id": null, "confianca": 0.0}. '
    "Não invente leis, números, prazos nem telefones."
)


def _formatar_trechos(resultados) -> str:
    linhas = []
    for trecho, score in resultados:
        linhas.append(f"[{trecho['id']}] (fonte: {trecho['fonte']}) {trecho['texto']}")
    return "\n\n".join(linhas)


def sem_contexto(pergunta: str, busca: Busca | None = None) -> dict:
    """Configuração 1: o modelo responde sem nenhum trecho da base."""
    saida = llm.conversar([
        {"role": "system", "content": SISTEMA + " Se não tiver certeza, devolva "
         '{"resposta": "não consta", "fonte_id": null, "confianca": 0.0}.'},
        {"role": "user", "content": pergunta},
    ])
    return {
        "config": "sem_contexto",
        "pergunta": pergunta,
        "saida_bruta": saida,
        "json": llm.extrair_json(saida),
        "trechos_usados": [],
        "passos": ["pergunta direta ao modelo, sem contexto"],
    }


def rag(pergunta: str, busca: Busca, k: int = 3) -> dict:
    """Configuração 2: RAG simples com limiar de recusa."""
    resultados = busca.buscar(pergunta, k=k)
    passos = [f"busca top-{k}: " + ", ".join(f"{t['id']}={s:.2f}" for t, s in resultados)]

    if not resultados or resultados[0][1] < busca.limiar:
        passos.append(f"score máximo abaixo do limiar ({busca.limiar}): recusa")
        return {
            "config": "rag",
            "pergunta": pergunta,
            "saida_bruta": None,
            "json": dict(RECUSA),
            "trechos_usados": [],
            "passos": passos,
        }

    saida = llm.conversar([
        {"role": "system", "content": SISTEMA + " " + REGRAS_CONTEXTO},
        {"role": "user", "content": f"Trechos:\n\n{_formatar_trechos(resultados)}\n\nPergunta: {pergunta}"},
    ])
    passos.append("resposta gerada com os trechos no prompt")
    return {
        "config": "rag",
        "pergunta": pergunta,
        "saida_bruta": saida,
        "json": llm.extrair_json(saida),
        "trechos_usados": [t["id"] for t, _ in resultados],
        "passos": passos,
    }


def agente(pergunta: str, busca: Busca, k: int = 3) -> dict:
    """Configuração 3: agente com 4 etapas.

    1. BUSCAR: busca; se o score for baixo, reformula a consulta com a própria
       LLM (sinônimos/termos formais) e busca de novo antes de desistir.
    2. GERAR: gera a resposta JSON com os trechos.
    3. VALIDAR: se o JSON for inválido, pede correção uma vez; se a fonte
       citada não estiver entre os trechos recuperados, corrige para o top-1.
    4. CRITICAR: pergunta à LLM se o trecho citado sustenta a resposta;
       se não sustentar, recusa em vez de alucinar.
    """
    passos = []

    # 1. BUSCAR (com reformulação)
    resultados = busca.buscar(pergunta, k=k)
    passos.append(f"busca 1: " + ", ".join(f"{t['id']}={s:.2f}" for t, s in resultados))
    if not resultados or resultados[0][1] < busca.limiar:
        reformulada = llm.conversar([
            {"role": "system", "content": "Reescreva a pergunta do usuário usando termos formais de "
             "zeladoria urbana, trânsito e legislação (ex.: 'buraco' -> 'conservação do pavimento, "
             "via pública, tapa-buraco'). Responda só com a pergunta reescrita."},
            {"role": "user", "content": pergunta},
        ], max_tokens=80).strip()
        passos.append(f"score baixo; consulta reformulada: {reformulada!r}")
        resultados2 = busca.buscar(f"{pergunta} {reformulada}", k=k)
        passos.append("busca 2: " + ", ".join(f"{t['id']}={s:.2f}" for t, s in resultados2))
        if resultados2 and resultados2[0][1] > (resultados[0][1] if resultados else 0):
            resultados = resultados2
    if not resultados or resultados[0][1] < busca.limiar:
        passos.append(f"ainda abaixo do limiar ({busca.limiar}): recusa")
        return {"config": "agente", "pergunta": pergunta, "saida_bruta": None,
                "json": dict(RECUSA), "trechos_usados": [], "passos": passos}

    ids_recuperados = [t["id"] for t, _ in resultados]

    # 2. GERAR
    saida = llm.conversar([
        {"role": "system", "content": SISTEMA + " " + REGRAS_CONTEXTO},
        {"role": "user", "content": f"Trechos:\n\n{_formatar_trechos(resultados)}\n\nPergunta: {pergunta}"},
    ])
    obj = llm.extrair_json(saida)
    passos.append("resposta gerada")

    # 3. VALIDAR JSON (uma nova tentativa se inválido)
    if obj is None or "resposta" not in obj:
        passos.append("JSON inválido: pedindo correção")
        saida = llm.conversar([
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": "A saída abaixo deveria ser um JSON no formato "
             '{"resposta": "...", "fonte_id": "...", "confianca": 0.0} mas está inválida. '
             f"Corrija e devolva apenas o JSON:\n\n{saida}"},
        ])
        obj = llm.extrair_json(saida)
        if obj is None or "resposta" not in obj:
            passos.append("JSON continuou inválido: recusa segura")
            return {"config": "agente", "pergunta": pergunta, "saida_bruta": saida,
                    "json": dict(RECUSA), "trechos_usados": ids_recuperados, "passos": passos}

    resposta_txt = str(obj.get("resposta", "")).strip()
    if resposta_txt.lower().startswith("não consta") or resposta_txt.lower().startswith("nao consta"):
        passos.append("modelo recusou com os trechos em mãos")
        return {"config": "agente", "pergunta": pergunta, "saida_bruta": saida,
                "json": obj, "trechos_usados": ids_recuperados, "passos": passos}

    # 3b. VALIDAR CITAÇÃO: a fonte citada precisa ser um trecho recuperado
    if obj.get("fonte_id") not in ids_recuperados:
        passos.append(f"fonte citada ({obj.get('fonte_id')!r}) não está nos trechos: corrigida para o top-1")
        obj["fonte_id"] = ids_recuperados[0]

    # 4. CRITICAR: o trecho citado sustenta a resposta?
    trecho = busca.por_id(obj["fonte_id"])
    veredito = llm.conversar([
        {"role": "system", "content": "Você é um revisor. Responda apenas SIM ou NAO."},
        {"role": "user", "content": "O TRECHO abaixo sustenta a RESPOSTA? Responda SIM ou NAO.\n\n"
         f"TRECHO: {trecho['texto']}\n\nRESPOSTA: {obj['resposta']}"},
    ], max_tokens=5).strip().upper()
    passos.append(f"autocrítica: {veredito}")
    if veredito.startswith("N"):
        passos.append("resposta não sustentada pelo trecho: recusa segura")
        return {"config": "agente", "pergunta": pergunta, "saida_bruta": saida,
                "json": dict(RECUSA), "trechos_usados": ids_recuperados, "passos": passos}

    return {"config": "agente", "pergunta": pergunta, "saida_bruta": saida,
            "json": obj, "trechos_usados": ids_recuperados, "passos": passos}


CONFIGURACOES = {
    "sem_contexto": sem_contexto,
    "rag": rag,
    "agente": agente,
}


if __name__ == "__main__":
    import sys

    b = Busca()
    pergunta = " ".join(sys.argv[1:]) or "como reporto um buraco na minha rua?"
    for nome, fn in CONFIGURACOES.items():
        r = fn(pergunta, b)
        print(f"\n== {nome} ==")
        print(json.dumps(r["json"], ensure_ascii=False, indent=2))
