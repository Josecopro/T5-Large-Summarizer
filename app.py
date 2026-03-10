import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ddgs import DDGS
from textwrap import dedent
from datetime import datetime
import time


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo Flan-T5-large...")
def cargar_modelo():
    nombre = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(nombre)
    modelo = T5ForConditionalGeneration.from_pretrained(nombre)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()
    return tokenizer, modelo, device


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------
def generar_queries(tema: str, max_queries: int = 5) -> list[str]:
    tema = tema.strip()
    base = f'"{tema}"'
    queries = [
        base,
        f"{base} que es definicion",
        f"{base} aplicaciones ejemplos",
        f"{base} caracteristicas principales",
        f"{base} ventajas importancia",
    ]
    return queries[:max_queries]


def buscar_en_internet(queries: list[str], tema: str, max_fuentes: int = 15) -> list[dict]:
    resultados = []
    urls_vistas: set[str] = set()
    tema_lower = tema.lower()

    try:
        with DDGS() as ddgs:
            for query in queries:
                try:
                    search_results = list(ddgs.text(query, max_results=10))
                    for r in search_results:
                        titulo = (r.get("title", "") or "").strip()[:200]
                        snippet = (r.get("body", "") or "").strip()[:800]
                        url = (r.get("href", "") or "").strip()
                        if url in urls_vistas:
                            continue
                        texto = f"{titulo} {snippet}".lower()
                        if tema_lower in texto:
                            urls_vistas.add(url)
                            resultados.append({"titulo": titulo, "snippet": snippet, "url": url})
                    time.sleep(0.5)
                except Exception:
                    continue
    except Exception:
        pass

    return resultados[:max_fuentes]


def _generar_texto(prompt_text: str, tokenizer, modelo, device, max_length: int = 512) -> str:
    inputs = tokenizer.encode(
        prompt_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    outputs = modelo.generate(
        inputs,
        max_length=max_length,
        num_beams=5,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generar_resumen(tema: str, resultados: list[dict], tokenizer, modelo, device) -> str:
    if not resultados:
        return "No se encontro informacion relevante para el tema."

    # --- Step 1: Summarize sources in chunks that fit the context window ---
    chunk_size = 4
    resumenes_parciales = []

    for start in range(0, len(resultados), chunk_size):
        chunk = resultados[start : start + chunk_size]
        texto_fuentes = "\n".join(
            f"[{start + j + 1}] {r['titulo']}: {r['snippet']}"
            for j, r in enumerate(chunk)
        )

        prompt_chunk = dedent(f"""\
Summarize the following sources about "{tema}" in Spanish.
Include the reference numbers in brackets when using information from a source.
Write a detailed paragraph.

Sources:
{texto_fuentes}

Detailed summary in Spanish:""")

        parcial = _generar_texto(prompt_chunk, tokenizer, modelo, device, max_length=300)
        if parcial.strip():
            resumenes_parciales.append(parcial.strip())

    # --- Step 2: Merge partial summaries into a final coherent summary ---
    if len(resumenes_parciales) == 1:
        resumen_combinado = resumenes_parciales[0]
    else:
        textos_parciales = "\n\n".join(
            f"Part {i+1}: {r}" for i, r in enumerate(resumenes_parciales)
        )
        prompt_final = dedent(f"""\
You are given partial summaries about "{tema}".
Combine them into one coherent summary in Spanish with 3 paragraphs:
1. Definition and explanation of the topic.
2. Main characteristics and applications with examples.
3. Importance and relevance.

Keep all reference numbers [1], [2], etc. from the partial summaries.
Do not invent information.

Partial summaries:
{textos_parciales}

Final combined summary in Spanish:""")

        resumen_combinado = _generar_texto(prompt_final, tokenizer, modelo, device, max_length=512)

    return resumen_combinado


def crear_documento_txt(tema: str, resultados: list[dict], resumen: str) -> str:
    lineas = [
        "=" * 80,
        f"TEMA: {tema}",
        "=" * 80,
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Cantidad de fuentes usadas: {len(resultados)}",
        "",
        "RESUMEN",
        "-" * 80,
        resumen,
        "",
        "FUENTES",
        "-" * 80,
    ]
    for i, r in enumerate(resultados, 1):
        lineas.append(f"[{i}] {r['titulo']}")
        lineas.append(f"    URL: {r['url']}")
        lineas.append("")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Agente Investigador", page_icon="🔍", layout="centered")

with st.sidebar:
    st.title("Agente Investigador")
    st.markdown(
        "Busca informacion en internet sobre un tema y genera un resumen "
        "automatico con referencias usando **Flan-T5-large**."
    )
    st.markdown("---")
    st.caption("Modelo: google/flan-t5-large")

tokenizer, modelo, device = cargar_modelo()
st.sidebar.success(f"Modelo cargado en **{device}**")

st.header("Busqueda y Resumen con IA")

tema = st.text_input("Tema a investigar", placeholder="Ej: Inteligencia Artificial")

if st.button("Investigar", type="primary", disabled=not tema):
    with st.status("Investigando...", expanded=True) as status:
        st.write("Generando consultas de busqueda...")
        queries = generar_queries(tema)

        st.write("Buscando en DuckDuckGo...")
        resultados = buscar_en_internet(queries, tema)

        if not resultados:
            status.update(label="Sin resultados", state="error")
            st.error("No se encontraron fuentes relevantes para ese tema.")
        else:
            st.write(f"Se encontraron **{len(resultados)}** fuentes. Generando resumen...")
            resumen = generar_resumen(tema, resultados, tokenizer, modelo, device)
            status.update(label="Listo", state="complete")

    if resultados:
        st.subheader("Resumen")
        st.write(resumen)

        st.subheader("Fuentes")
        for i, r in enumerate(resultados, 1):
            with st.expander(f"[{i}] {r['titulo']}"):
                st.write(r["snippet"])
                st.markdown(f"[Abrir enlace]({r['url']})")

        txt = crear_documento_txt(tema, resultados, resumen)
        safe_name = tema[:40].strip().replace(" ", "_")
        st.download_button(
            label="Descargar resumen (.txt)",
            data=txt,
            file_name=f"resumen_{safe_name}.txt",
            mime="text/plain",
        )
