from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from video_editor import VideoEditor
import threading
import time
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['TEMP_FOLDER'] = 'temp'

# Ensure all required folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global variables for progress tracking
processing_status = {}
processing_locks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

def cleanup_files(input_path, processed_path):
    """Clean up input and processed files"""
    try:
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Deleted input file: {input_path}")
        if os.path.exists(processed_path):
            os.remove(processed_path)
            print(f"Deleted processed file: {processed_path}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def process_video(input_path, output_filename, effects=None, session_id=None):
    """Process video with progress tracking per session"""
    if session_id not in processing_status:
        processing_status[session_id] = {
            'current_task': '',
            'progress': 0,
            'status': 'idle',
            'error': None
        }
    
    status = processing_status[session_id]
    
    try:
        status['status'] = 'processing'
        editor = VideoEditor()
        
        # Set default effects if none provided
        if effects is None:
            effects = {
                'saturation': True,
                'contrast': True,
                'vignette': True,
                'intensity': 70
            }
        
        # Process video with color grading while keeping original audio
        status['current_task'] = 'Applying color grading and processing video'
        status['progress'] = 0
        
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        editor.process_video(input_path, output_path, effects)
        
        # Clean up input file after processing
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Deleted input file after processing: {input_path}")
        
        status['progress'] = 100
        status['status'] = 'completed'
        
    except Exception as e:
        status['error'] = str(e)
        status['status'] = 'error'
        print(f"Error during video processing: {str(e)}")
        # Clean up files in case of error
        cleanup_files(input_path, os.path.join(app.config['PROCESSED_FOLDER'], output_filename))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate unique session ID
        session_id = str(time.time())
        
        # Reset status for this session
        processing_status[session_id] = {
            'current_task': 'Uploading',
            'progress': 0,
            'status': 'uploading',
            'error': None
        }
        
        # Get effect choices
        effects = request.form.get('effects')
        if effects:
            effects = json.loads(effects)
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f'processed_{filename}'
        
        # Save uploaded file
        file.save(input_path)
        
        # Start processing in background
        thread = threading.Thread(
            target=process_video,
            args=(input_path, output_filename, effects, session_id)
        )
        thread.start()
        
        return jsonify({
            'message': 'Upload successful, processing started',
            'filename': output_filename,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id in processing_status:
        return jsonify(processing_status[session_id])
    return jsonify({
        'error': 'Session not found',
        'status': 'error'
    }), 404

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Send file and delete after sending
        response = send_file(file_path, as_attachment=True)
        
        # Delete the file after sending
        @response.call_on_close
        def on_close():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted processed file after download: {file_path}")
            except Exception as e:
                print(f"Error deleting file after download: {str(e)}")
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 