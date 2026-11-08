"""Points 2 et 3 : extraire texte, tableaux et images d'un PDF, décrire les images,
calculer les embeddings et tout stocker dans PGVector."""
import sys
from functools import cache
from pathlib import Path

import ollama
import pymupdf
import torch
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.postgres import PGVectorStore
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Connexion au conteneur PGVector défini dans docker-compose.yml
DB = dict(host="localhost", port="5432", user="postgres", password="postgres", database="maintidoc")
VISION_MODEL = "qwen2.5vl:3b"                 # modèle de vision (Ollama) qui décrit les images
CLIP_NAME = "openai/clip-vit-base-patch32"    # CLIP : vecteurs de 512 dimensions
MEDIA_DIR = Path("media")                     # PDF chargés + images/tableaux extraits

# Modèle d'embeddings texte utilisé par LlamaIndex partout (bge-m3 : 1024 dimensions, multilingue).
# num_gpu=0 : calcul sur le processeur, car bge-m3 renvoie des NaN sur la GPU T550.
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", ollama_additional_kwargs={"num_gpu": 0})


def store(table: str, dim: int) -> PGVectorStore:
    """Ouvre (et crée si besoin) une table vectorielle dans PostgreSQL."""
    return PGVectorStore.from_params(**DB, table_name=table, embed_dim=dim)


@cache  # le modèle CLIP n'est chargé qu'une seule fois
def clip():
    return CLIPModel.from_pretrained(CLIP_NAME).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)


def _normalize(output) -> list[float]:
    """transformers v5 renvoie un objet : le vecteur projeté est dans pooler_output.
    On le normalise (longueur 1) pour que la similarité cosinus soit cohérente."""
    return torch.nn.functional.normalize(output.pooler_output, dim=-1)[0].tolist()


def clip_image_embedding(path: str) -> list[float]:
    model, processor = clip()
    with torch.no_grad():  # pas d'entraînement : on désactive le calcul des gradients
        inputs = processor(images=Image.open(path).convert("RGB"), return_tensors="pt")
        return _normalize(model.get_image_features(**inputs))


def clip_text_embedding(text: str) -> list[float]:
    """Projette une phrase dans le même espace que les images : permet la recherche texte -> image."""
    model, processor = clip()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        return _normalize(model.get_text_features(**inputs))


def caption(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    """Retourne la légende ('Figure N' / 'Table N') la plus proche verticalement du rectangle."""
    blocks = [b for b in page.get_text("blocks") if b[4].lstrip().startswith(("Figure", "Table"))]
    if not blocks:
        return ""
    best = min(blocks, key=lambda b: min(abs(b[1] - rect.y1), abs(b[3] - rect.y0)))
    return " ".join(best[4].split())


def extract(pdf_path: str):
    """Retourne (pages de texte, tableaux, images) avec leurs métadonnées."""
    source = Path(pdf_path).name
    out = MEDIA_DIR / Path(pdf_path).stem
    out.mkdir(parents=True, exist_ok=True)

    # Texte : le lecteur LlamaIndex produit un Document par page
    pages = PyMuPDFReader().load_data(file_path=pdf_path)
    for doc in pages:
        doc.metadata = {"source": source, "page": int(doc.metadata["source"]), "type": "text"}

    tables, images = [], []
    pdf = pymupdf.open(pdf_path)
    for page in pdf:
        n = page.number + 1

        # Tableaux : PyMuPDF les détecte, on en garde une capture (transcrite plus tard par le modèle de vision)
        for i, tab in enumerate(page.find_tables().tables):
            rect = pymupdf.Rect(tab.bbox)
            png = out / f"p{n}_table{i}.png"
            page.get_pixmap(clip=rect, dpi=150).save(png)
            tables.append({"page": n, "path": str(png), "caption": caption(page, rect)})

        # Images : zones des images insérées dans la page
        rects = [page.get_image_bbox(img) for img in page.get_images(full=True)]
        rects = [r for r in rects if not r.is_infinite and r.width > 50 and r.height > 50]
        # Figures dessinées en vectoriel (pas d'image insérée mais beaucoup de tracés)
        drawings = page.get_drawings()
        if not rects and len(drawings) > 100:
            zone = pymupdf.Rect()
            for d in drawings:
                zone |= d["rect"]
            rects = [zone]
        for i, rect in enumerate(rects):
            png = out / f"p{n}_img{i}.png"
            page.get_pixmap(clip=rect, dpi=150).save(png)  # "photo" de la zone, telle qu'affichée
            images.append({"source": source, "page": n, "path": str(png), "caption": caption(page, rect)})
    return pages, tables, images


FIGURE_PROMPT = ("Décris précisément cette figure extraite d'un document technique : type de schéma, "
                 "composants, flèches, textes visibles et ce qu'elle illustre. Réponds en français.")
TABLE_PROMPT = ("Recopie ce tableau en Markdown, ligne par ligne et colonne par colonne, avec exactement "
                "les valeurs visibles. Si l'en-tête a plusieurs niveaux, combine-les dans chaque nom de colonne "
                "(ex : 'BLEU EN-DE', 'BLEU EN-FR'). Laisse vide une cellule vide. "
                "Réponds uniquement avec le tableau.")


def vision(path: str, prompt: str, legend: str) -> str:
    """Envoie l'image et la consigne au modèle de vision.
    Le résultat est gardé dans un .txt à côté de l'image : ré-indexer ne le recalcule pas."""
    cached = Path(path).with_suffix(".txt")
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    if legend:
        prompt += f"\nLégende : {legend}"
    reply = ollama.chat(model=VISION_MODEL, messages=[{"role": "user", "content": prompt, "images": [path]}],
                        options={"num_predict": 600})  # plafonne la longueur : évite les boucles de répétition
    cached.write_text(reply["message"]["content"], encoding="utf-8")
    return reply["message"]["content"]


def ingest(pdf_path: str, progress=print) -> None:
    source = Path(pdf_path).name
    text_store, image_store = store("text_chunks", 1024), store("image_clip", 512)

    # Ré-indexer le même PDF remplace ses anciennes lignes au lieu de les dupliquer
    same_pdf = MetadataFilters(filters=[MetadataFilter(key="source", value=source)])
    text_store.delete_nodes(filters=same_pdf)
    image_store.delete_nodes(filters=same_pdf)

    progress("Extraction du texte, des tableaux et des images…")
    pages, tables, images = extract(pdf_path)
    progress(f"{len(pages)} pages, {len(tables)} tableaux, {len(images)} images")

    # Découpage en passages de 512 tokens avec chevauchement (comme dans le cours)
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(pages)
    for k, tab in enumerate(tables, 1):
        progress(f"Transcription du tableau {k}/{len(tables)} (page {tab['page']})…")
        markdown = vision(tab["path"], TABLE_PROMPT, tab["caption"])
        meta = {"source": source, "page": tab["page"], "type": "table", "image_path": tab["path"]}
        nodes.append(TextNode(text=f"Tableau page {tab['page']}. {tab['caption']}\n{markdown}", metadata=meta))

    image_nodes = []
    for k, img in enumerate(images, 1):
        progress(f"Description de l'image {k}/{len(images)} (page {img['page']})…")
        description = vision(img["path"], FIGURE_PROMPT, img["caption"])
        meta = {"source": source, "page": img["page"], "type": "image", "image_path": img["path"]}
        # La description devient un passage texte interrogeable comme les autres
        nodes.append(TextNode(text=f"Figure page {img['page']}. {img['caption']}\n{description}", metadata=meta))
        # Et l'image elle-même est vectorisée par CLIP
        image_nodes.append(ImageNode(image_path=img["path"], text=description, metadata=meta,
                                     embedding=clip_image_embedding(img["path"])))

    for node in nodes:  # le chemin du fichier n'aide ni la recherche ni le LLM
        node.excluded_embed_metadata_keys = ["image_path"]
        node.excluded_llm_metadata_keys = ["image_path"]

    progress(f"Embeddings et stockage de {len(nodes)} passages (bge-m3 -> PGVector)…")
    VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=text_store))
    image_store.add(image_nodes)  # embeddings CLIP déjà calculés
    progress("Indexation terminée.")


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else "attention.pdf")
