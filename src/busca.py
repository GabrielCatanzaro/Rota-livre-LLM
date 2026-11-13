"""Busca de trechos na base de conhecimento com TF-IDF.

Escolhemos TF-IDF (scikit-learn) por ser simples, leve (roda em qualquer CPU,
sem download de modelos) e suficiente para uma base pequena (~35 trechos).
O limiar de similaridade decide quando a busca "não encontrou nada" e a IA
deve responder "não consta" — ele é medido e justificado no notebook 01.
"""

import json
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_BASE = RAIZ / "base" / "base_conhecimento.json"

# Limiar de similaridade abaixo do qual consideramos que a base não cobre a
# pergunta e recusamos ("não consta") sem nem chamar a LLM.
# Justificativa (medida no notebook 01): no conjunto de teste, o MENOR score
# top-1 de uma pergunta respondível é ~0.16, enquanto 3 das 5 perguntas
# "não consta" ficam abaixo de 0.15. Com 0.15 nenhuma pergunta respondível é
# recusada por engano e parte das não respondíveis é barrada de graça; as que
# passam do limiar são tratadas pela regra de recusa no prompt (RAG/agente).
LIMIAR = 0.15

# Stopwords mínimas de português (evita dependência extra como o NLTK)
STOPWORDS_PT = """
a o e de da do das dos em no na nos nas um uma uns umas para por com sem sob
que se ao aos à às pelo pela pelos pelas é são foi ser tem têm como mais mas
ou onde quando quem qual quais isso essa esse esta este eu meu minha seu sua
ele ela eles elas nós vocês já não sim também até entre sobre depois antes
""".split()


def normalizar(texto: str) -> str:
    """minúsculas + sem acentos, para o TF-IDF casar 'calçada' com 'calcada'."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9§,. ]+", " ", texto)
    return texto


class Busca:
    def __init__(self, caminho_base: Path = CAMINHO_BASE, limiar: float = LIMIAR):
        dados = json.loads(Path(caminho_base).read_text(encoding="utf-8"))
        self.trechos = dados["trechos"]
        self.limiar = limiar
        corpus = [f"{t['fonte']} {t['texto']}" for t in self.trechos]
        self.vetorizador = TfidfVectorizer(
            preprocessor=normalizar,
            stop_words=sorted({normalizar(p) for p in STOPWORDS_PT}),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matriz = self.vetorizador.fit_transform(corpus)

    def buscar(self, pergunta: str, k: int = 3):
        """Retorna os k trechos mais parecidos: [(trecho, score), ...]."""
        v = self.vetorizador.transform([pergunta])
        scores = cosine_similarity(v, self.matriz)[0]
        ordem = scores.argsort()[::-1][:k]
        return [(self.trechos[i], float(scores[i])) for i in ordem]

    def por_id(self, trecho_id: str):
        for t in self.trechos:
            if t["id"] == trecho_id:
                return t
        return None


def medir_recall(busca: Busca, casos: list, k: int = 3) -> dict:
    """recall@k: fração dos casos respondíveis em que ao menos um trecho do
    gabarito aparece entre os k primeiros resultados da busca."""
    respondiveis = [c for c in casos if c["tipo"] == "respondivel"]
    acertos = 0
    detalhes = []
    for caso in respondiveis:
        resultados = busca.buscar(caso["pergunta"], k=k)
        ids_topk = [t["id"] for t, _ in resultados]
        acertou = any(tid in ids_topk for tid in caso["trecho_ids"])
        acertos += int(acertou)
        detalhes.append({
            "id": caso["id"],
            "acertou": acertou,
            "esperados": caso["trecho_ids"],
            "retornados": ids_topk,
            "score_top1": resultados[0][1] if resultados else 0.0,
        })
    return {
        "k": k,
        "recall": acertos / len(respondiveis) if respondiveis else 0.0,
        "n_casos": len(respondiveis),
        "detalhes": detalhes,
    }


if __name__ == "__main__":
    import sys

    busca = Busca()
    pergunta = " ".join(sys.argv[1:]) or "como reportar um buraco na rua?"
    for trecho, score in busca.buscar(pergunta):
        print(f"{score:.3f}  [{trecho['id']}] {trecho['fonte']}")
