"""Rotas acessíveis para pessoas com mobilidade reduzida (extensão do Rota Livre).

Dada uma pergunta como "qual o caminho acessível do Metrô Sé até o Theatro Municipal?",
a IA recomenda, para quem usa cadeira de rodas, a rota pela calçada que:
1. NÃO tem bloqueios — escada, travessia sem rampa (guia rebaixada), inclinação acima
   de 8,33% (limite de rampa da NBR 9050) ou passagem com menos de 0,90 m;
2. entre as acessíveis, tem o menor índice = (buracos + obstáculos) a cada 100 m.

Base: base/calcadas_simuladas.json — DADOS SIMULADOS (as ruas existem, as condições
das calçadas foram geradas para o MVP). Em produção viriam dos relatos do app e do SP156.

Mesmas três configurações do projeto:
1. sem_contexto — a LLM sozinha, sem os dados (tende a inventar rota).
2. rag          — busca os textos das rotas e a LLM decide sozinha (bloqueios + números).
3. agente       — identifica origem/destino, usa a FERRAMENTA que exclui as rotas
                  bloqueadas e ordena as acessíveis; a LLM só explica; a rota citada é
                  validada e trajetos fora da base são recusados.
"""

import json
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import llm
from .busca import STOPWORDS_PT, normalizar

RAIZ = Path(__file__).resolve().parent.parent
DADOS = json.loads((RAIZ / "base" / "calcadas_simuladas.json").read_text(encoding="utf-8"))
TRECHOS, LOCAIS, ROTAS = DADOS["trechos"], DADOS["locais"], DADOS["rotas"]
FONTE = DADOS["fonte"]
RECUSA = {"resposta": "não consta", "rota_id": None, "confianca": 0.0}


# ------------------------------------------------------------------ ferramentas

def nome_local(loc_id):
    return LOCAIS[loc_id]["nome"]


def nomes_trechos(r):
    return [TRECHOS[t]["nome"] for t in r["trechos"]]


def rotas_do_trajeto(origem, destino):
    """Rotas do par (qualquer sentido): acessíveis primeiro (menor índice), bloqueadas depois."""
    lst = [r for r in ROTAS if {r["origem"], r["destino"]} == {origem, destino}]
    return sorted(lst, key=lambda r: (not r["acessivel"], r["indice"], r["metros"]))


def descrever_rota(r):
    situacao = ("ACESSÍVEL para cadeira de rodas" if r["acessivel"]
                else "BLOQUEADA para cadeira de rodas: " + "; ".join(r["bloqueios"]))
    return (f"Rota {r['id']} a pé de {nome_local(r['origem'])} até {nome_local(r['destino'])} "
            f"(vale nos dois sentidos): por {' → '.join(nomes_trechos(r))}. {r['metros']} m, "
            f"{r['buracos']} buracos na calçada, {r['obstaculos']} obstáculos, índice {r['indice']} por 100 m. "
            f"{situacao}.")


def extrair_trajeto(pergunta):
    """Encontra origem e destino citados na pergunta (ordem de aparição)."""
    q = " " + re.sub(r"[^a-z0-9 ]", " ", normalizar(pergunta)) + " "
    q = re.sub(r" +", " ", q)
    achados = []
    for loc_id, info in LOCAIS.items():
        pos = [q.find(" " + a + " ") for a in info["apelidos"] if q.find(" " + a + " ") >= 0]
        if pos:
            achados.append((min(pos), loc_id))
    vistos = []
    for _, loc in sorted(achados):
        if loc not in vistos:
            vistos.append(loc)
    if len(vistos) < 2:
        return None
    return vistos[0], vistos[1]


# ------------------------------------------------------------------ busca (RAG)

_TEXTOS = [descrever_rota(r) for r in ROTAS]
_VET = TfidfVectorizer(preprocessor=normalizar, stop_words=sorted({normalizar(p) for p in STOPWORDS_PT}),
                       ngram_range=(1, 2), sublinear_tf=True)
_MAT = _VET.fit_transform(_TEXTOS)


def buscar_rotas(pergunta, k=6):
    scores = cosine_similarity(_VET.transform([pergunta]), _MAT)[0]
    ordem = scores.argsort()[::-1][:k]
    return [(ROTAS[i], _TEXTOS[i], float(scores[i])) for i in ordem]


# ------------------------------------------------------------------ as 3 configurações

SISTEMA = ("Você é o assistente de rotas acessíveis do Rota Livre, que recomenda em São Paulo o caminho a pé "
           "pela calçada para pessoas em cadeira de rodas ou com mobilidade reduzida. "
           "Responda SEMPRE em português e SOMENTE com um JSON válido no formato "
           '{"resposta": "...", "rota_id": "...", "confianca": 0.0}. ')
REGRA_RECUSA = ('Se não houver dados sobre esse trajeto, devolva exatamente '
                '{"resposta": "não consta", "rota_id": null, "confianca": 0.0}. Não invente ruas nem números.')


def sem_contexto(pergunta):
    saida = llm.conversar([{"role": "system", "content": SISTEMA + REGRA_RECUSA},
                           {"role": "user", "content": pergunta}])
    return {"config": "sem_contexto", "json": llm.extrair_json(saida), "ranking": [],
            "passos": ["pergunta direta ao modelo, sem dados de buracos"]}


def rag(pergunta):
    res = buscar_rotas(pergunta)
    passos = ["busca top-6: " + ", ".join(f"{r['id']}={s:.2f}" for r, _, s in res)]
    trechos = "\n".join(f"[{r['id']}] {t}" for r, t, _ in res)
    saida = llm.conversar([
        {"role": "system", "content": SISTEMA + "Use APENAS os dados fornecidos. NUNCA recomende rota BLOQUEADA para cadeira de rodas. "
         "Entre as rotas ACESSÍVEIS do trajeto pedido, escolha a de menor índice. "
         "Em 'rota_id' coloque o id da rota escolhida. " + REGRA_RECUSA},
        {"role": "user", "content": f"Dados (simulados) de rotas:\n{trechos}\n\nPergunta: {pergunta}"}])
    passos.append("a LLM decidiu sozinha (bloqueios e números) com os trechos recuperados")
    return {"config": "rag", "json": llm.extrair_json(saida), "ranking": [], "passos": passos}


def agente(pergunta):
    passos = []
    trajeto = extrair_trajeto(pergunta)
    if not trajeto:
        passos.append("não identifiquei origem e destino conhecidos na pergunta: recusa")
        return {"config": "agente", "json": dict(RECUSA), "ranking": [], "passos": passos}
    o, d = trajeto
    passos.append(f"trajeto identificado: {nome_local(o)} → {nome_local(d)}")
    ranking = rotas_do_trajeto(o, d)
    if not ranking:
        passos.append("esse trajeto não existe na base de rotas: recusa")
        return {"config": "agente", "json": dict(RECUSA), "ranking": [], "passos": passos}
    bloq = [r["id"] for r in ranking if not r["acessivel"]]
    passos.append("ferramenta: rotas bloqueadas excluídas: " + (", ".join(bloq) or "nenhuma"))
    acess = [r for r in ranking if r["acessivel"]]
    if not acess:
        passos.append("nenhuma rota acessível nesse trajeto: recusa")
        return {"config": "agente", "json": {"resposta": "não consta rota acessível para esse trajeto",
                "rota_id": None, "confianca": 0.0}, "ranking": ranking, "passos": passos}
    passos.append("ferramenta: ranking das acessíveis: " + " < ".join(f"{r['id']} ({r['indice']})" for r in acess))
    melhor = acess[0]
    tabela = "\n".join(f"{i+1}. {descrever_rota(r)}" for i, r in enumerate(ranking))
    saida = llm.conversar([
        {"role": "system", "content": SISTEMA + "O ranking abaixo já foi calculado e está correto. Explique em 1 ou 2 "
         "frases por que a rota 1 é a melhor para quem usa cadeira de rodas, citando as ruas, os buracos e "
         "obstáculos e a distância, e diga por que as rotas bloqueadas foram descartadas. "
         f"Use rota_id = \"{melhor['id']}\"."},
        {"role": "user", "content": f"Pergunta: {pergunta}\n\nRanking (dados simulados):\n{tabela}"}])
    obj = llm.extrair_json(saida)
    if obj is None or "resposta" not in obj:
        passos.append("JSON inválido da LLM: resposta montada pela ferramenta")
        obj = {"resposta": f"Vá por {' → '.join(nomes_trechos(melhor))}: rota acessível, {melhor['metros']} m, "
                           f"{melhor['buracos']} buracos e {melhor['obstaculos']} obstáculos.",
               "rota_id": melhor["id"], "confianca": 0.9}
    if obj.get("rota_id") != melhor["id"]:
        passos.append(f"rota citada ({obj.get('rota_id')!r}) diferente do topo do ranking: corrigida para {melhor['id']}")
        obj["rota_id"] = melhor["id"]
    passos.append("JSON validado e rota conferida com a ferramenta")
    return {"config": "agente", "json": obj, "ranking": ranking, "passos": passos}


CONFIGURACOES = {"sem_contexto": sem_contexto, "rag": rag, "agente": agente}


def responder(pergunta, modo="agente"):
    r = CONFIGURACOES.get(modo, agente)(pergunta)
    rid = (r["json"] or {}).get("rota_id")
    rota = next((x for x in ROTAS if x["id"] == rid), None)
    r["rota"] = ({**rota, "trechos_nomes": nomes_trechos(rota), "fonte": FONTE} if rota else None)
    r["ranking"] = [{**x, "trechos_nomes": nomes_trechos(x)} for x in r["ranking"]]
    return r


# ------------------------------------------------------------------ avaliação

def _recusou(obj):
    return obj is not None and normalizar(str(obj.get("resposta", ""))).strip().startswith("nao consta")


def avaliar():
    casos = json.loads((RAIZ / "avaliacao" / "casos_rotas.json").read_text(encoding="utf-8"))["casos"]
    resultados = {}
    for nome, fn in CONFIGURACOES.items():
        execs = []
        for c in casos:
            t0 = time.time()
            try:
                r = fn(c["pergunta"])
            except Exception as e:
                r = {"config": nome, "json": None, "ranking": [], "passos": [f"erro: {e}"]}
            obj = r["json"]
            if c["tipo"] == "nao_consta":
                correto = _recusou(obj)
            else:
                correto = obj is not None and obj.get("rota_id") == c["gabarito_rota"]
            execs.append({"caso_id": c["id"], "tipo": c["tipo"], "json": obj, "correto": correto,
                          "passos": r["passos"], "duracao_s": round(time.time() - t0, 1)})
            print(f"[{nome}] {c['id']} {'OK ' if correto else 'ERR'}")
        resp = [e for e in execs if e["tipo"] == "respondivel"]
        nc = [e for e in execs if e["tipo"] == "nao_consta"]
        ids = {r["id"] for r in ROTAS}
        BLOQ = {r["id"] for r in ROTAS if not r["acessivel"]}
        respondidos = [e for e in resp if e["json"] and not _recusou(e["json"])]
        resultados[nome] = {
            "metricas": {
                "config": nome,
                "acerto_rota": round(sum(e["correto"] for e in resp) / len(resp), 3),
                "recusa_correta": round(sum(e["correto"] for e in nc) / len(nc), 3),
                "alucinacao": round(sum(1 for e in nc if e["json"] and not _recusou(e["json"])) / len(nc), 3),
                "json_valido": round(sum(1 for e in execs if e["json"] is not None) / len(execs), 3),
                "recomendou_bloqueada": round(sum(1 for e in respondidos if e["json"].get("rota_id") in BLOQ)
                                              / len(respondidos), 3) if respondidos else 0.0,
                "rota_existente": round(sum(1 for e in respondidos if e["json"].get("rota_id") in ids)
                                        / len(respondidos), 3) if respondidos else 0.0,
                "tempo_medio_s": round(sum(e["duracao_s"] for e in execs) / len(execs), 1),
            },
            "execucoes": execs,
        }
    (RAIZ / "avaliacao" / "rotas_resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    linhas = ["# Placar — rotas acessíveis (cadeira de rodas)", "",
              f"{len(casos)} casos ({sum(c['tipo'] == 'respondivel' for c in casos)} trajetos da base, "
              f"{sum(c['tipo'] == 'nao_consta' for c in casos)} fora da base) · dados simulados", "",
              "| Configuração | Acerto da rota | Recomendou rota bloqueada ↓ | Recusa correta | Alucinação ↓ | JSON válido | Rota citada existe | Tempo médio (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for n, r in resultados.items():
        m = r["metricas"]
        linhas.append(f"| {n} | {m['acerto_rota']:.0%} | {m['recomendou_bloqueada']:.0%} | {m['recusa_correta']:.0%} | {m['alucinacao']:.0%} | "
                      f"{m['json_valido']:.0%} | {m['rota_existente']:.0%} | {m['tempo_medio_s']} |")
    (RAIZ / "avaliacao" / "rotas_placar.md").write_text("\n".join(linhas), encoding="utf-8")
    print("\nPlacar salvo em avaliacao/rotas_placar.md")
    return resultados


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["avaliar"]:
        avaliar()
    else:
        p = " ".join(sys.argv[1:]) or "Qual o caminho acessível do Metrô Sé até o Theatro Municipal?"
        for modo in CONFIGURACOES:
            r = responder(p, modo)
            print(f"\n== {modo} ==\n" + json.dumps(r["json"], ensure_ascii=False, indent=2))
