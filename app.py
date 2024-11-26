from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import openai
from flask import Flask,render_template
from dotenv import load_dotenv


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Load models
load_dotenv()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
openai.api_key['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    upload_folder = os.path.abspath('./uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)

    try:
        print(f"Attempting to save file to: {filepath}")
        print(f"Upload folder writable: {os.access(upload_folder, os.W_OK)}")
        print(f"Sanitized filename: {filename}")

        # Save file (manual write for testing)
        with open(filepath, 'wb') as f:
            f.write(file.read())

        # Process file...
        return jsonify({'message': 'File uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    import os

    # Get the query from the request
    data = request.json
    print(data, 'data')
    query = data.get('query')
    print(query, 'query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    # Locate the most recent file in the uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']
    print(upload_folder, 'upload_folder')
    files = sorted(
        [os.path.join(upload_folder, f) for f in os.listdir(upload_folder)],
        key=os.path.getctime,
        reverse=True
    )
    print(files, 'files')
    if not files:
        return jsonify({'error': 'No files found in the uploads folder'}), 400

    latest_file = files[0]
    print(latest_file, 'latest_file')
    app.logger.info(f"Processing file: {latest_file}")

    # Extract the text based on the file type
    if latest_file.endswith('.pdf'):
        document_text = extract_text_from_pdf(latest_file)
    elif latest_file.endswith('.txt'):
        with open(latest_file, 'r') as f:
            document_text = f.read()
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Split the text into chunks and encode them
    chunks = document_text.split("\n\n")
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    # Embed the query and find similar chunks
    try:
        query_embedding = embedding_model.encode(query)
        similarities = [util.cos_sim(query_embedding, chunk_embedding).item() for chunk_embedding in embeddings]
        most_similar_chunk = chunks[similarities.index(max(similarities))]
        
        # Use LLM for answering
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Document excerpt: {most_similar_chunk}\n\nQuestion: {query}"}
            ],
            max_tokens=150
        )
        answer = response['choices'][0]['message']['content'].strip()
        return jsonify({'answer': answer}), 200

    except Exception as e:
        app.logger.error(f"Error processing question: {e}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
