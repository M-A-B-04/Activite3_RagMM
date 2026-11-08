# RAG multimodal agentique — Partie I

Application de questions-réponses sur un PDF contenant du texte, des images et des tableaux.
Le PDF est indexé de façon multimodale dans PGVector ; un agent LangChain interroge l'index
et un LLM local (Qwen via Ollama) génère des réponses qui citent les pages.

## Prérequis

| Outil | Version testée | Rôle |
|---|---|---|
| Windows 10/11, Linux ou macOS | — | |
| RAM | 16 Go minimum  | le modèle de vision occupe ~10 Go sur CPU |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 29.6 | conteneur PostgreSQL + pgvector |
| [Ollama](https://ollama.com/download) | 0.24 | exécution locale des modèles |
| [uv](https://docs.astral.sh/uv/) | 0.10 | gestion de Python et des dépendances |
| Python | ≥ 3.12 (3.13 testé) | installé automatiquement par uv |

GPU optionnelle : le projet tourne entièrement sur CPU.

### Modèles Ollama

| Modèle | Taille | Rôle |
|---|---|---|
| `bge-m3` | 1,2 Go | embeddings du texte (1024 dimensions, multilingue) |
| `qwen3:4b-instruct` | 2,5 Go | LLM de génération et de l'agent |
| `qwen2.5vl:3b` | 3,2 Go | description des figures et transcription des tableaux |

Le modèle CLIP `openai/clip-vit-base-patch32` (~600 Mo) est téléchargé automatiquement
depuis Hugging Face au premier lancement.

### Dépendances Python

Déclarées dans `pyproject.toml` et figées dans `uv.lock` :

```
langchain>=1.4.7
langchain-ollama>=1.1.0
llama-index-core>=0.14.25
llama-index-embeddings-ollama>=0.10.0
llama-index-readers-file>=0.7.0
llama-index-vector-stores-postgres>=0.9.0
ollama>=0.6.3
pillow>=13.0.0
pymupdf>=2.0
streamlit>=1.66.1
torch>=2.14.1
transformers>=5.21.0
```

## Installation

```bash
# 1. Base vectorielle PostgreSQL + pgvector
docker compose up -d

# 2. Modèles locaux (Ollama doit être lancé)
ollama pull bge-m3
ollama pull qwen3:4b-instruct
ollama pull qwen2.5vl:3b

# 3. Environnement Python et dépendances
uv sync
```

## Utilisation

```bash
# Indexer un PDF en ligne de commande (par défaut : attention.pdf)
uv run python ingestion.py attention.pdf

# Interface web : chargement du PDF, questions, réponses avec sources
uv run streamlit run app.py
```

L'interface est disponible sur http://localhost:8501. Le mode « Agent » utilise l'agent
LangChain ; le mode « RAG simple » exécute la chaîne recherche → prompt → LLM.

La première indexation d'un PDF prend du temps (environ 2 minutes par figure ou tableau
sur CPU). Les descriptions sont ensuite gardées en cache dans `media/`.

## Structure

| Fichier | Contenu |
|---|---|
| `ingestion.py` | extraction (LlamaIndex + PyMuPDF), descriptions (VLM), embeddings (bge-m3, CLIP), stockage PGVector |
| `agent.py` | RAG simple + agent LangChain avec les outils `search_text` et `search_images` |
| `app.py` | interface Streamlit |
| `docker-compose.yml` | service PostgreSQL + pgvector |
| `media/` | PDF chargés, captures des figures/tableaux et leurs descriptions |

## Limites connues

- `bge-m3` renvoie des NaN sur certaines petites GPU (testé : NVIDIA T550) ; il est forcé sur CPU.
- Le modèle de vision 3B transcrit mal les tableaux à en-tête sur plusieurs niveaux.
- CLIP ne comprend que l'anglais : l'agent traduit la requête avant la recherche d'images.
- Temps de réponse observé : 20 à 30 secondes par question sur CPU.
