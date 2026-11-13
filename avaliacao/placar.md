# Placar das três configurações

Modelo local via llama.cpp · 27 casos de teste (22 respondíveis, 5 'não consta')

| Configuração | Acurácia | Alucinação ↓ | JSON válido | Citação correta | Tempo médio (s) |
|---|---|---|---|---|---|
| sem_contexto | 26% | 20% | 100% | 0% | 2.6 |
| rag | 74% | 0% | 100% | 84% | 2.9 |
| agente | 70% | 0% | 100% | 100% | 4.8 |

Gerado por `python -m src.avaliar`. Detalhes caso a caso em `resultados.json`.