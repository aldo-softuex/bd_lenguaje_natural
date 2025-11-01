# Sistema de Gestión Inteligente

Aplicación web con dos funcionalidades principales:
- **Agente SQL**: Consultas en lenguaje natural a base de datos MySQL
- **Escáner CURP**: Extracción de datos de códigos QR en documentos CURP

## Instalación

1. Clona el repositorio
2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Configura variables de entorno:
```bash
cp .env.example .env
# Edita .env con tus credenciales
```

## Configuración

Edita el archivo `.env` con tus credenciales:

```env
# Base de Datos
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_HOST=tu_host
DB_NAME=tu_database

# API Keys
OPENAI_API_KEY=tu_openai_key
```

## Ejecución

```bash
python app.py
```

Visita: http://localhost:5000

## Funcionalidades

### 🤖 Agente SQL
- Consultas en lenguaje natural
- Respuestas en español
- Conexión segura a MySQL

### 📄 Escáner CURP
- Sube imágenes o PDFs
- Detección automática de QR
- Extracción de datos personales

## Tecnologías

- Flask
- LangChain
- OpenAI GPT
- QReader
- PyMuPDF
- MySQL