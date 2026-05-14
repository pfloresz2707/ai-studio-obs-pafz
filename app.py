import os
import time
import requests
import gradio as gr
import pandas as pd
import PyPDF2
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

MAX_TEXT_CHARS = 12000
MAX_HF_CHARS = 500
MAX_FILE_TEXT_CHARS = 10000

CLASSIFICATION_LABELS = [
    "educación",
    "negocios",
    "tecnología",
    "finanzas",
    "servicio al cliente"
]


def format_success(result):
    return f"✅ Resultado listo\n\n{result}"


def format_warning(message):
    return f"⚠️ Revisa tu entrada\n\n{message}"


def format_error(message, suggestion=None):
    text = f"❌ No fue posible completar la solicitud\n\n{message}"
    if suggestion:
        text += f"\n\nSugerencia: {suggestion}"
    return text


def is_feedback_message(value):
    return isinstance(value, str) and (value.startswith("❌") or value.startswith("⚠️"))


def format_final_result(result):
    if is_feedback_message(result):
        return result
    return format_success(result)


def clean_text(text):
    if text is None or not isinstance(text, str):
        return ""
    return text.strip()


def validate_text_input(text, task_name):
    text = clean_text(text)
    if not text:
        return None, f"Por favor ingresa un texto para {task_name}."
    if len(text) < 3:
        return None, "El texto es demasiado corto. Ingresa al menos una frase completa."
    return text, None


def split_comments(text):
    return [comment.strip() for comment in text.split("\n") if comment.strip()]


def get_api_error_message(response, provider_name):
    try:
        error_data = response.json()
    except Exception:
        error_data = {}

    message = ""
    status = ""

    if isinstance(error_data, dict):
        error = error_data.get("error", {})
        if isinstance(error, dict):
            message = error.get("message", "")
            status = error.get("status", "")
        else:
            message = str(error_data)

    if not message:
        message = response.text

    status_code = response.status_code

    if provider_name == "Gemini":
        if status_code == 503:
            return format_error(
                "Gemini está temporalmente saturado por alta demanda. Esto no significa que tu archivo o tu API Key estén mal.",
                "Espera unos minutos y vuelve a intentar. También puedes probar con un texto o archivo más corto."
            )
        if status_code == 429:
            return format_error(
                "Se alcanzó temporalmente el límite de uso o de solicitudes de Gemini.",
                "Espera unos minutos antes de volver a intentar o reduce el tamaño del texto."
            )
        if status_code in [401, 403]:
            return format_error(
                "La API Key de Gemini no fue aceptada o no tiene permisos suficientes.",
                "Verifica que GEMINI_API_KEY esté configurada correctamente en Secrets o en tu archivo .env."
            )
        if status_code == 400:
            return format_error(
                "Gemini no pudo procesar la solicitud enviada.",
                "Revisa que el contenido no esté vacío, sea texto legible y no exceda el tamaño recomendado."
            )

    if provider_name == "Hugging Face":
        if status_code == 503:
            return format_error(
                "El modelo de Hugging Face está cargando o temporalmente no está disponible.",
                "Espera unos segundos y vuelve a intentar."
            )
        if status_code == 429:
            return format_error(
                "Se alcanzó temporalmente el límite de uso de Hugging Face.",
                "Espera unos minutos antes de volver a intentar."
            )
        if status_code in [401, 403]:
            return format_error(
                "El token de Hugging Face no fue aceptado o no tiene permisos suficientes.",
                "Verifica que HF_API_KEY esté configurada correctamente en Secrets o en tu archivo .env."
            )

    detail = f"{provider_name} respondió con código {status_code}."
    if status:
        detail += f"\nEstado: {status}"
    if message:
        detail += f"\nDetalle: {message}"

    return format_error(
        detail,
        "Intenta nuevamente. Si el problema continúa, revisa tus credenciales y el tamaño del texto."
    )


def call_gemini(prompt):
    try:
        if not GEMINI_API_KEY:
            return format_error(
                "No se encontró la API Key de Gemini.",
                "Configura GEMINI_API_KEY en Hugging Face Secrets o en tu archivo .env local."
            )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(GEMINI_URL, json=payload, timeout=60)

        if response.status_code in [503, 429]:
            time.sleep(2)
            response = requests.post(GEMINI_URL, json=payload, timeout=60)

        if response.status_code != 200:
            return get_api_error_message(response, "Gemini")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return format_error(
                "Gemini no devolvió una respuesta válida.",
                "Intenta con un texto más corto o revisa que el contenido del archivo sea legible."
            )

        return candidates[0]["content"]["parts"][0]["text"]

    except requests.exceptions.Timeout:
        return format_error(
            "Gemini tardó demasiado en responder.",
            "Intenta nuevamente con un texto más corto o espera unos minutos."
        )
    except requests.exceptions.RequestException as error:
        return format_error(
            f"No se pudo conectar con Gemini. Detalle técnico: {error}",
            "Verifica tu conexión a internet y vuelve a intentar."
        )
    except Exception as error:
        return format_error(
            f"Ocurrió un error inesperado al procesar Gemini. Detalle técnico: {error}",
            "Intenta nuevamente o reduce el tamaño del texto."
        )


def call_huggingface(model_id, payload):
    try:
        if not HF_API_KEY:
            return format_error(
                "No se encontró el token de Hugging Face.",
                "Configura HF_API_KEY en Hugging Face Secrets o en tu archivo .env local."
            )

        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code in [503, 429]:
            time.sleep(2)
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return get_api_error_message(response, "Hugging Face")

        return response.json()

    except requests.exceptions.Timeout:
        return format_error(
            "Hugging Face tardó demasiado en responder.",
            "Intenta nuevamente con menos comentarios o un texto más corto."
        )
    except requests.exceptions.RequestException as error:
        return format_error(
            f"No se pudo conectar con Hugging Face. Detalle técnico: {error}",
            "Verifica tu conexión a internet y vuelve a intentar."
        )
    except Exception as error:
        return format_error(
            f"Ocurrió un error inesperado al consultar Hugging Face. Detalle técnico: {error}",
            "Intenta nuevamente o reduce el tamaño del texto."
        )


def summarize_text(text):
    text, error = validate_text_input(text, "resumir")
    if error:
        return format_warning(error)

    prompt = f"""
Resume el siguiente texto en español, de forma clara, breve y estructurada:

{text[:MAX_TEXT_CHARS]}
"""
    result = call_gemini(prompt)
    return format_final_result(result)


def translate_text(text):
    text, error = validate_text_input(text, "traducir")
    if error:
        return format_warning(error)

    prompt = f"""
Traduce el siguiente texto al inglés.
Mantén el sentido original y usa lenguaje natural:

{text[:MAX_TEXT_CHARS]}
"""
    result = call_gemini(prompt)
    return format_final_result(result)


def normalize_sentiment_label(label):
    label = str(label).lower()
    if "positive" in label or label in ["label_2", "pos"]:
        return "positive"
    if "negative" in label or label in ["label_0", "neg"]:
        return "negative"
    if "neutral" in label or label in ["label_1", "neu"]:
        return "neutral"
    return label


def analyze_sentiment(text):
    text, error = validate_text_input(text, "analizar sentimiento")
    if error:
        return format_warning(error)

    try:
        comentarios = split_comments(text)
        if not comentarios:
            return format_warning("Por favor ingresa comentarios separados por salto de línea.")

        positivos, neutrales, negativos = 0, 0, 0
        detalle = []

        for i, comentario in enumerate(comentarios, start=1):
            result = call_huggingface(
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                {"inputs": comentario[:MAX_HF_CHARS]}
            )
            if is_feedback_message(result):
                return result

            scores = result[0] if isinstance(result, list) and len(result) > 0 else result
            if not isinstance(scores, list):
                return format_error(
                    f"Hugging Face devolvió una respuesta inesperada: {result}",
                    "Intenta nuevamente o reduce el texto de entrada."
                )

            top = max(scores, key=lambda item: item.get("score", 0))
            label = normalize_sentiment_label(top.get("label", "neutral"))
            score = round(top.get("score", 0) * 100, 2)

            if label == "positive":
                positivos += 1
                etiqueta = "Positivo"
            elif label == "negative":
                negativos += 1
                etiqueta = "Negativo"
            else:
                neutrales += 1
                etiqueta = "Neutral"

            detalle.append(f"{i}. {etiqueta} ({score}%) — {comentario}")

        resultado = f"""
Resumen del análisis de sentimiento

Total de comentarios analizados: {len(comentarios)}

Comentarios positivos: {positivos}
Comentarios neutrales: {neutrales}
Comentarios negativos: {negativos}

Detalle por comentario:
{chr(10).join(detalle)}
"""
        return format_success(resultado)

    except Exception as error:
        return format_error(
            f"Ocurrió un error al analizar sentimiento. Detalle técnico: {error}",
            "Intenta con menos comentarios o revisa que cada comentario esté separado por salto de línea."
        )


def parse_classification_result(result):
    if isinstance(result, dict):
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        if labels and scores:
            return [{"label": label, "score": score} for label, score in zip(labels, scores)]

    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "label" in result[0]:
            return result
        if result and isinstance(result[0], list):
            return result[0]

    return []


def classify_text(text):
    text, error = validate_text_input(text, "clasificar")
    if error:
        return format_warning(error)

    try:
        comentarios = split_comments(text)
        if not comentarios:
            return format_warning("Por favor ingresa comentarios separados por salto de línea.")

        conteo_categorias = {label: 0 for label in CLASSIFICATION_LABELS}
        detalle = []

        for i, comentario in enumerate(comentarios, start=1):
            result = call_huggingface(
                "facebook/bart-large-mnli",
                {
                    "inputs": comentario[:MAX_HF_CHARS],
                    "parameters": {"candidate_labels": CLASSIFICATION_LABELS}
                }
            )
            if is_feedback_message(result):
                return result

            parsed_result = parse_classification_result(result)
            if not parsed_result:
                return format_error(
                    f"Hugging Face devolvió una respuesta inesperada: {result}",
                    "Intenta nuevamente o reduce el texto de entrada."
                )

            top = max(parsed_result, key=lambda item: item.get("score", 0))
            top_label = top.get("label", "sin categoría")
            top_score = round(top.get("score", 0) * 100, 2)

            if top_label in conteo_categorias:
                conteo_categorias[top_label] += 1
            else:
                conteo_categorias[top_label] = 1

            scores_por_categoria = []
            for item in parsed_result:
                label = item.get("label", "sin etiqueta")
                score = round(item.get("score", 0) * 100, 2)
                scores_por_categoria.append(f"   - {label}: {score}%")

            detalle.append(
                f"{i}. Categoría principal: {top_label} ({top_score}%)\n"
                f"   Comentario: {comentario}\n"
                f"   Scores por categoría:\n"
                f"{chr(10).join(scores_por_categoria)}"
            )

        resumen_conteo = "\n".join([f"- {categoria}: {cantidad}" for categoria, cantidad in conteo_categorias.items()])
        resultado = f"""
Resumen de clasificación

Total de comentarios clasificados: {len(comentarios)}

Conteo por categoría:
{resumen_conteo}

Detalle por comentario:
{chr(10).join(detalle)}

Interpretación:
Cada comentario fue asignado a la categoría con mayor score de confianza dentro de las opciones disponibles.
"""
        return format_success(resultado)

    except Exception as error:
        return format_error(
            f"Ocurrió un error al clasificar el texto. Detalle técnico: {error}",
            "Intenta con menos comentarios o revisa que cada comentario esté separado por salto de línea."
        )


def extract_file_text(file):
    if file is None:
        return ""

    try:
        file_path = file.name
        if not isinstance(file_path, str):
            return "Formato de archivo inválido."

        file_path_lower = file_path.lower()

        if file_path_lower.endswith(".pdf"):
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                    return ""
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text[:MAX_FILE_TEXT_CHARS]

        if file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
            df = pd.read_excel(file_path)
            if df.empty:
                return ""
            return df.head(100).to_string()

        if file_path_lower.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")
            if df.empty:
                return ""
            return df.head(100).to_string()

        return "Formato no soportado. Usa únicamente PDF, Excel o CSV."

    except PyPDF2.errors.PdfReadError:
        return "No se pudo leer el PDF. Verifica que no esté dañado, escaneado como imagen o protegido."
    except pd.errors.EmptyDataError:
        return "El archivo CSV está vacío."
    except Exception as error:
        return format_error(
            f"No fue posible leer el archivo. Detalle técnico: {error}",
            "Verifica que el archivo no esté protegido, dañado o vacío."
        )


def analyze_file(file, action):
    try:
        allowed_actions = ["Resumir", "Traducir", "Sentimiento", "Clasificar"]
        if action not in allowed_actions:
            return format_warning("Selecciona una acción válida: Resumir, Traducir, Sentimiento o Clasificar.")

        text = extract_file_text(file)
        if not text:
            return format_warning(
                "No se pudo extraer texto del archivo. Verifica que el archivo tenga contenido legible. "
                "Si es PDF, asegúrate de que no sea únicamente una imagen escaneada."
            )

        if isinstance(text, str) and text.startswith("Formato no soportado"):
            return format_warning(text)
        if isinstance(text, str) and text.startswith("No se pudo leer"):
            return format_warning(text)
        if isinstance(text, str) and text.startswith("El archivo CSV"):
            return format_warning(text)
        if is_feedback_message(text):
            return text

        if action == "Resumir":
            return summarize_text(text)
        if action == "Traducir":
            return translate_text(text)
        if action == "Sentimiento":
            return analyze_sentiment(text)
        if action == "Clasificar":
            return classify_text(text)

        return format_warning("Acción no válida.")

    except Exception as error:
        return format_error(
            f"Ocurrió un error al analizar el archivo. Detalle técnico: {error}",
            "Intenta con otro archivo o reduce su tamaño."
        )


css = """
body {
    background: #f5f5f5;
}

.gradio-container {
    font-family: Arial, sans-serif !important;
}

.obs-header {
    background: #000000;
    color: white;
    padding: 28px;
    border-bottom: 8px solid #ffd600;
    margin-bottom: 18px;
}

.obs-header h1 {
    color: white !important;
    margin-bottom: 8px;
    font-weight: 900 !important;
}

.obs-header p {
    color: #ffd600;
    font-weight: 700;
}

.status-note {
    background: #fff8cc;
    border-left: 8px solid #ffd600;
    padding: 14px 18px;
    margin-bottom: 18px;
    font-weight: 700;
    color: #111111;
}

button[role="tab"] {
    background: #ffd600 !important;
    color: #000000 !important;
    font-weight: 900 !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #ffd600 !important;
}

button[role="tab"][aria-selected="true"] {
    background: #000000 !important;
    color: #ffd600 !important;
    border: 1px solid #000000 !important;
}

button:not([role="tab"]) {
    background: #ffd600 !important;
    color: #000000 !important;
    font-weight: 900 !important;
    border: none !important;
}

button:not([role="tab"]):hover {
    background: #000000 !important;
    color: white !important;
}

.input-box textarea {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
}

.output-box textarea {
    background: #fff8cc !important;
    border: 2px solid #ffd600 !important;
    font-weight: 600 !important;
}

.input-box label,
.output-box label,
.dropdown-box label {
    font-weight: 900 !important;
    color: #111111 !important;
}

.file-box {
    border: 2px dashed #111111 !important;
    background: #ffffff !important;
}
"""


with gr.Blocks(css=css, theme=gr.themes.Default()) as app:
    gr.HTML(
        """
        <div class="obs-header">
            <h1>AI Studio OBS</h1>
            <p>Aplicación Multi-IA con Gemini + Hugging Face</p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="status-note">
            Selecciona una pestaña, llena el campo de entrada o carga un archivo y presiona el botón de ejecución.
            El resultado aparecerá en el recuadro amarillo con el mensaje “✅ Resultado listo”.
            Si el servicio de IA está saturado, la app mostrará una explicación clara y una sugerencia para volver a intentar.
        </div>
        """
    )

    with gr.Tab("Resumen"):
        text_input = gr.Textbox(
            label="Entrada del usuario: texto a resumir",
            lines=8,
            placeholder="Pega aquí el texto que quieres resumir...",
            elem_classes=["input-box"]
        )
        output = gr.Textbox(label="Resultado generado por IA", lines=8, elem_classes=["output-box"])
        btn = gr.Button("Ejecutar resumen")
        btn.click(summarize_text, inputs=text_input, outputs=output)

    with gr.Tab("Traducción"):
        text_input = gr.Textbox(
            label="Entrada del usuario: texto a traducir",
            lines=8,
            placeholder="Pega aquí el texto que quieres traducir al inglés...",
            elem_classes=["input-box"]
        )
        output = gr.Textbox(label="Resultado generado por IA", lines=8, elem_classes=["output-box"])
        btn = gr.Button("Ejecutar traducción")
        btn.click(translate_text, inputs=text_input, outputs=output)

    with gr.Tab("Sentimiento"):
        text_input = gr.Textbox(
            label="Entrada del usuario: comentarios para analizar sentimiento",
            lines=8,
            placeholder="Pega aquí los comentarios, uno por línea. Ejemplo:\nLa app es muy fácil de usar.\nTuve problemas con el pago.\nLa experiencia fue normal.",
            elem_classes=["input-box"]
        )
        output = gr.Textbox(label="Resultado generado por IA", lines=12, elem_classes=["output-box"])
        btn = gr.Button("Analizar sentimiento")
        btn.click(analyze_sentiment, inputs=text_input, outputs=output)

    with gr.Tab("Clasificación"):
        text_input = gr.Textbox(
            label="Entrada del usuario: comentarios para clasificar en categorías",
            lines=8,
            placeholder="""Pega aquí los comentarios, uno por línea.

Categorías disponibles:
• educación
• negocios
• tecnología
• finanzas
• servicio al cliente

Ejemplo:
La app es fácil de usar.
No pude aplicar mi cupón de descuento.
Me gustaría recibir información sobre cursos.
""",
            elem_classes=["input-box"]
        )
        output = gr.Textbox(label="Resultado generado por IA", lines=14, elem_classes=["output-box"])
        btn = gr.Button("Clasificar texto")
        btn.click(classify_text, inputs=text_input, outputs=output)

    with gr.Tab("Analizar archivo"):
        file_input = gr.File(
            label="Entrada del usuario: sube un archivo PDF, Excel o CSV",
            file_types=[".pdf", ".xlsx", ".xls", ".csv"],
            elem_classes=["file-box"]
        )
        action = gr.Dropdown(
            choices=["Resumir", "Traducir", "Sentimiento", "Clasificar"],
            label="Entrada del usuario: selecciona la acción",
            value="Resumir",
            elem_classes=["dropdown-box"]
        )
        output = gr.Textbox(label="Resultado generado por IA", lines=14, elem_classes=["output-box"])
        btn = gr.Button("Analizar archivo")
        btn.click(analyze_file, inputs=[file_input, action], outputs=output)


app.launch(share=True)
