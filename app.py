from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import os
from werkzeug.utils import secure_filename
from qreader import QReader
from PIL import Image
import io
import fitz  # PyMuPDF
import numpy as np
import warnings
from urllib.parse import quote_plus
from dotenv import load_dotenv
import cv2
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForTokenClassification
import re

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configuración
def load_config():
    api_key = os.getenv('OPENAI_API_KEY')
    
    config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME')
    }
    
    # Validar que todas las variables estén configuradas
    if not all([api_key, config['user'], config['password'], config['host'], config['database']]):
        raise ValueError("Faltan variables de entorno. Revisa el archivo .env")
    
    return api_key, config

# Inicializar agente SQL
def init_sql_agent():
    api_key, config = load_config()
    
    # Suprimir warnings de tipos geoespaciales
    warnings.filterwarnings('ignore', message='Did not recognize type.*of column')
    
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    
    # Codificar password para caracteres especiales
    encoded_password = quote_plus(config['password'])
    connection_string = f"mysql+mysqlconnector://{config['user']}:{encoded_password}@{config['host']}/{config['database']}"
    
    db = SQLDatabase.from_uri(connection_string)
    
    return create_sql_agent(
        llm, 
        db=db, 
        verbose=False,
        agent_executor_kwargs={
            "handle_parsing_errors": True
        },
        prefix="Eres un asistente SQL que responde SIEMPRE en español. Analiza la base de datos y responde las preguntas del usuario en español de manera clara y concisa."
    )

agent = init_sql_agent()

# Inicializar modelos para INE
print("Cargando docTR...")
doctr_model = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch="crnn_vgg16_bn",
    pretrained=True
)

print("Cargando TrOCR...")
processor_trocr = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
trocr.eval()

print("Cargando modelo NER...")
model_path = "./modelo_ner_final_seccion"
tokenizer_ner = AutoTokenizer.from_pretrained(model_path)
model_ner = AutoModelForTokenClassification.from_pretrained(model_path)
id2label = model_ner.config.id2label
label2id = model_ner.config.label2id

@app.route('/')
def index():
    return render_template('menu.html')

@app.route('/sql')
def sql_agent():
    return render_template('sql_agent.html')

@app.route('/curp')
def curp_scanner():
    return render_template('curp_scanner.html')

@app.route('/query', methods=['POST'])
def query_database():
    try:
        user_question = request.json.get('question', '')
        if not user_question:
            return jsonify({'error': 'No se proporcionó pregunta'}), 400
        
        # Agregar instrucción en español
        spanish_question = f"Responde en español: {user_question}"
        result = agent.invoke(spanish_question)
        return jsonify({'result': result['output']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def ocr_hibrido(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("temp_doc.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    doc = DocumentFile.from_images(["temp_doc.png"])
    detect_result = doctr_model(doc)
    texto_final = []
    page = detect_result.pages[0]
    
    for block in page.blocks:
        for line in block.lines:
            x_min = min(w.geometry[0][0] for w in line.words)
            y_min = min(w.geometry[0][1] for w in line.words)
            x_max = max(w.geometry[1][0] for w in line.words)
            y_max = max(w.geometry[1][1] for w in line.words)
            
            h, w, _ = image_rgb.shape
            xmin = int(x_min * w)
            ymin = int(y_min * h)
            xmax = int(x_max * w)
            ymax = int(y_max * h)
            
            line_img = image_rgb[ymin:ymax, xmin:xmax]
            if line_img.size == 0:
                continue
            
            pil_img = Image.fromarray(line_img)
            pixel_values = processor_trocr(images=pil_img, return_tensors="pt").pixel_values
            generated_ids = trocr.generate(pixel_values)
            text = processor_trocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texto_final.append(text)
    
    return "\n".join(texto_final)

def predecir_texto(texto):
    tokens = texto.split()
    inputs = tokenizer_ner(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model_ner(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    word_ids = inputs.word_ids(batch_index=0)
    predicted_labels = []
    previous_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            predicted_labels.append(id2label[predictions[0][i].item()])
        previous_word_idx = word_idx
    
    resultado = []
    for token, label in zip(tokens, predicted_labels):
        resultado.append({
            "token": token,
            "etiqueta": label
        })
    
    return resultado

def extraer_entidades(texto):
    predicciones = predecir_texto(texto)
    entidades = {}
    entidad_actual = None
    tokens_entidad = []
    
    for pred in predicciones:
        token = pred["token"]
        label = pred["etiqueta"]
        
        if label.startswith("B-"):
            if entidad_actual and tokens_entidad:
                tipo = entidad_actual.replace("B-", "")
                if tipo not in entidades:
                    entidades[tipo] = []
                entidades[tipo].append(" ".join(tokens_entidad))
            
            entidad_actual = label
            tokens_entidad = [token]
            
        elif label.startswith("I-") and entidad_actual:
            tokens_entidad.append(token)
            
        else:
            if entidad_actual and tokens_entidad:
                tipo = entidad_actual.replace("B-", "")
                if tipo not in entidades:
                    entidades[tipo] = []
                entidades[tipo].append(" ".join(tokens_entidad))
            
            entidad_actual = None
            tokens_entidad = []
    
    if entidad_actual and tokens_entidad:
        tipo = entidad_actual.replace("B-", "")
        if tipo not in entidades:
            entidades[tipo] = []
        entidades[tipo].append(" ".join(tokens_entidad))
    
    return entidades

@app.route('/escanear_ine', methods=['POST'])
def escanear_ine():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file_data = file.read()
        
        # Procesar imagen directamente
        image = Image.open(io.BytesIO(file_data))
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Procesar con OCR híbrido
        texto_extraido = ocr_hibrido(image_bgr)
        
        # Limpiar texto
        texto_corrido = texto_extraido.replace('\n', ' ').strip()
        texto_corrido = re.sub(r'\s+', ' ', texto_corrido)
        
        # Extraer entidades
        entidades = extraer_entidades(texto_corrido)
        
        return jsonify(entidades)
    
    except Exception as e:
        return jsonify({'error': f'Error procesando archivo: {str(e)}'}), 500

@app.route('/scan_curp', methods=['POST'])
def scan_curp():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        file_data = file.read()
        file_type = file.content_type
        
        # Procesar según tipo de archivo
        if file_type == 'application/pdf':
            # Convertir PDF a imagen con mayor resolución
            pdf_doc = fitz.open(stream=file_data, filetype="pdf")
            page = pdf_doc[0]  # Primera página
            # Aumentar resolución para mejor detección QR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            pdf_doc.close()
        else:
            # Procesar como imagen
            image = Image.open(io.BytesIO(file_data))
        
        # Convertir PIL Image a numpy array para qreader
        image_array = np.array(image)
        
        # Inicializar QReader con configuración más sensible
        qreader = QReader(model_size='s', min_confidence=0.5)
        
        # Detectar y decodificar QR
        qr_codes = qreader.detect_and_decode(image=image_array)
        
        # Si no encuentra QR, intentar con diferentes configuraciones
        if not qr_codes or len(qr_codes) == 0:
            # Intentar con modelo más grande
            qreader_large = QReader(model_size='l', min_confidence=0.3)
            qr_codes = qreader_large.detect_and_decode(image=image_array)
            
            if not qr_codes or len(qr_codes) == 0:
                # Convertir a escala de grises y intentar nuevamente
                gray_image = image.convert('L')
                gray_array = np.array(gray_image)
                qr_codes = qreader.detect_and_decode(image=gray_array)
                
                if not qr_codes or len(qr_codes) == 0:
                    return jsonify({
                        'error': 'No se encontró código QR en el documento',
                        'debug': f'Tamaño imagen: {image.size}, Tipo: {file_type}'
                    }), 400
        
        # Filtrar QRs válidos (no vacíos)
        valid_qrs = [qr for qr in qr_codes if qr and qr.strip()]
        
        if not valid_qrs:
            return jsonify({
                'error': 'Los códigos QR encontrados están vacíos',
                'debug': f'QRs detectados: {len(qr_codes)}'
            }), 400
        
        # Buscar el QR con datos de CURP (no URLs)
        curp_qr = None
        for qr in valid_qrs:
            # Saltar URLs del gobierno
            if 'consultas.curp.gob.mx' in qr or 'http' in qr:
                continue
            # Buscar QR que contenga datos separados por pipes
            if '|' in qr and len(qr.split('|')) > 3:
                curp_qr = qr
                break
        
        # Si no encuentra QR de datos, usar el primero no-URL
        if not curp_qr:
            non_url_qrs = [qr for qr in valid_qrs if 'http' not in qr and 'consultas.curp.gob.mx' not in qr]
            if non_url_qrs:
                curp_qr = non_url_qrs[0]
            else:
                return jsonify({
                    'error': 'No se encontró QR con datos de CURP, solo URLs',
                    'debug': f'QRs encontrados: {len(valid_qrs)}, Primer QR: {valid_qrs[0][:100]}...'
                }), 400
        
        qr_data = curp_qr
        total_qrs_found = len(valid_qrs)
        
        # Parsear datos del QR de CURP
        def parse_curp_qr(qr_text):
            # Formato esperado: LORS950831HTSPSM04||LOPEZ|ROSALES|SAMUEL RICARDO|HOMBRE|31/08/1995|TAMAULIPAS|28
            parts = qr_text.split('|')
            
            # Filtrar partes vacías
            parts = [part for part in parts if part.strip()]
            
            # Intentar diferentes formatos
            if len(parts) >= 7:  # Formato con 7+ campos
                return {
                    "curp": parts[0],
                    "apellido_paterno": parts[1] if len(parts) > 1 else "No detectado",
                    "apellido_materno": parts[2] if len(parts) > 2 else "No detectado",
                    "nombres": parts[3] if len(parts) > 3 else "No detectado",
                    "nombre_completo": f"{parts[3]} {parts[1]} {parts[2]}".strip() if len(parts) > 3 else "No detectado",
                    "sexo": parts[4] if len(parts) > 4 else "No detectado",
                    "fecha_nacimiento": parts[5] if len(parts) > 5 else "No detectado",
                    "entidad_nacimiento": parts[6] if len(parts) > 6 else "No detectado",
                    "codigo_verificador": parts[7] if len(parts) > 7 else "No detectado",
                    "total_partes": len(parts),
                    "total_qrs_encontrados": total_qrs_found,
                    "qr_completo": qr_text
                }
            else:
                # Mostrar todas las partes encontradas para debug
                result = {
                    "curp": parts[0] if len(parts) > 0 else "No detectado",
                    "apellido_paterno": "Ver contenido QR",
                    "apellido_materno": "Ver contenido QR",
                    "nombres": "Ver contenido QR",
                    "nombre_completo": "Ver contenido QR",
                    "sexo": "Ver contenido QR",
                    "fecha_nacimiento": "Ver contenido QR",
                    "entidad_nacimiento": "Ver contenido QR",
                    "codigo_verificador": "Ver contenido QR",
                    "total_partes": len(parts),
                    "total_qrs_encontrados": total_qrs_found,
                    "qr_completo": qr_text
                }
                
                # Agregar cada parte encontrada para debug
                for i, part in enumerate(parts):
                    result[f"parte_{i}"] = part
                    
                return result
        
        result = parse_curp_qr(qr_data)
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': f'Error procesando archivo: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)