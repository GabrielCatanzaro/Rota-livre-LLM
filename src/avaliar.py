"""Avaliação das três configurações no conjunto de teste → placar.

Métricas por configuração:
- acuracia:        fração de casos corretos (respondíveis: resposta contém as
                   palavras-chave do gabarito; "não consta": recusou).
- alucinacao:      fração dos casos "não consta" em que o modelo INVENTOU uma
                   resposta em vez de recusar (quanto menor, melhor).
- json_valido:     fração de saídas que são JSON válido no formato pedido.
- citacao_correta: fração dos casos respondíveis respondidos em que a fonte
                   citada é um dos trechos do gabarito.

Uso:  python -m src.avaliar            (roda tudo e salva em avaliacao/)
"""

import json
import time
import unicodedata
from pathlib import Path

from .busca import Busca
from .configuracoes import CONFIGURACOES

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CASOS = RAIZ / "avaliacao" / "casos_teste.json"
SAIDA_RESULTADOS = RAIZ / "avaliacao" / "resultados.json"
SAIDA_PLACAR = RAIZ / "avaliacao" / "placar.md"


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def eh_recusa(obj) -> bool:
    if obj is None:
        return False
    return _norm(str(obj.get("resposta", ""))).strip().startswith("nao consta")


def caso_correto(caso: dict, obj) -> bool:
    if obj is None:
        return False
    if caso["tipo"] == "nao_consta":
        return eh_recusa(obj)
    if eh_recusa(obj):
        return False
    resposta = _norm(str(obj.get("resposta", "")))
    chaves = [_norm(p) for p in caso["palavras_chave"]]
    # correto se contém ao menos uma palavra-chave do gabarito
    return any(ch in resposta for ch in chaves) if chaves else True


def avaliar_config(nome: str, casos: list, busca: Busca) -> dict:
    fn = CONFIGURACOES[nome]
    execucoes = []
    for caso in casos:
        inicio = time.time()
        try:
            r = fn(caso["pergunta"], busca)
        except Exception as e:  # servidor fora do ar etc.
            r = {"config": nome, "pergunta": caso["pergunta"], "saida_bruta": f"ERRO: {e}",
                 "json": None, "trechos_usados": [], "passos": [f"erro: {e}"]}
        r["caso_id"] = caso["id"]
        r["tipo"] = caso["tipo"]
        r["correto"] = caso_correto(caso, r["json"])
        r["duracao_s"] = round(time.time() - inicio, 1)
        execucoes.append(r)
        simbolo = "OK " if r["correto"] else "ERR"
        print(f"[{nome}] {caso['id']} {simbolo} ({r['duracao_s']}s)")

    respondiveis = [e for e in execucoes if e["tipo"] == "respondivel"]
    nao_consta = [e for e in execucoes if e["tipo"] == "nao_consta"]
    respondidos = [e for e in respondiveis if e["json"] and not eh_recusa(e["json"])]
    casos_por_id = {c["id"]: c for c in casos}

    def citacao_ok(e):
        fonte = (e["json"] or {}).get("fonte_id")
        return fonte in casos_por_id[e["caso_id"]]["trecho_ids"]

    metricas = {
        "config": nome,
        "acuracia": round(sum(e["correto"] for e in execucoes) / len(execucoes), 3),
        "alucinacao": round(
            sum(1 for e in nao_consta if e["json"] and not eh_recusa(e["json"])) / len(nao_consta), 3
        ) if nao_consta else 0.0,
        "json_valido": round(sum(1 for e in execucoes if e["json"] is not None) / len(execucoes), 3),
        "citacao_correta": round(
            sum(1 for e in respondidos if citacao_ok(e)) / len(respondidos), 3
        ) if respondidos else 0.0,
        "tempo_medio_s": round(sum(e["duracao_s"] for e in execucoes) / len(execucoes), 1),
    }
    return {"metricas": metricas, "execucoes": execucoes}


def main():
    casos = json.loads(CAMINHO_CASOS.read_text(encoding="utf-8"))["casos"]
    busca = Busca()
    resultados = {}
    for nome in CONFIGURACOES:
        print(f"\n=== Avaliando configuração: {nome} ===")
        resultados[nome] = avaliar_config(nome, casos, busca)

    SAIDA_RESULTADOS.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    linhas = [
        "# Placar das três configurações",
        "",
        f"Modelo local via llama.cpp · {len(casos)} casos de teste "
        f"({sum(c['tipo'] == 'respondivel' for c in casos)} respondíveis, "
        f"{sum(c['tipo'] == 'nao_consta' for c in casos)} 'não consta')",
        "",
        "| Configuração | Acurácia | Alucinação ↓ | JSON válido | Citação correta | Tempo médio (s) |",
        "|---|---|---|---|---|---|",
    ]
    for nome, r in resultados.items():
        m = r["metricas"]
        linhas.append(
            f"| {nome} | {m['acuracia']:.0%} | {m['alucinacao']:.0%} | "
            f"{m['json_valido']:.0%} | {m['citacao_correta']:.0%} | {m['tempo_medio_s']} |"
        )
    linhas += ["", "Gerado por `python -m src.avaliar`. Detalhes caso a caso em `resultados.json`."]
    SAIDA_PLACAR.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nPlacar salvo em {SAIDA_PLACAR}")
    print(f"Resultados detalhados em {SAIDA_RESULTADOS}")


if __name__ == "__main__":
    main()
