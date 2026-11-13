"""Cliente da LLM local.

A LLM roda no llama.cpp (llama-server), que expõe uma API compatível com a da
OpenAI em http://localhost:8080. Funciona em GPU AMD via Vulkan, sem CUDA.

Também funciona com LM Studio (porta 1234) ou Ollama (porta 11434/v1):
basta ajustar a variável de ambiente LLM_URL.
"""

import json
import os
import re

import requests

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
MODELO = os.environ.get("LLM_MODELO", "local")
TIMEOUT = 180


def conversar(mensagens: list, temperatura: float = 0.2, max_tokens: int = 400) -> str:
    """Envia a conversa para a LLM local e devolve o texto da resposta."""
    resposta = requests.post(
        LLM_URL,
        json={
            "model": MODELO,
            "messages": mensagens,
            "temperature": temperatura,
            "max_tokens": max_tokens,
        },
        timeout=TIMEOUT,
    )
    resposta.raise_for_status()
    return resposta.json()["choices"][0]["message"]["content"]


def extrair_json(texto: str):
    """Extrai o primeiro objeto JSON do texto. Retorna None se for inválido."""
    texto = texto.strip()
    # remove cercas de código, se houver
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE)
    inicio = texto.find("{")
    if inicio == -1:
        return None
    # tenta do primeiro '{' até cada '}' final possível
    for fim in range(len(texto), inicio, -1):
        if texto[fim - 1] != "}":
            continue
        try:
            obj = json.loads(texto[inicio:fim])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def disponivel() -> bool:
    """True se o servidor da LLM está no ar."""
    try:
        base = LLM_URL.split("/v1/")[0]
        requests.get(base + "/health", timeout=5)
        return True
    except requests.RequestException:
        try:
            conversar([{"role": "user", "content": "oi"}], max_tokens=5)
            return True
        except requests.RequestException:
            return False


if __name__ == "__main__":
    print("Servidor disponível?", disponivel())
    print(conversar([{"role": "user", "content": "Responda só: ok"}]))
