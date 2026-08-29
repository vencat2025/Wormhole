from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Route for home page that displays the image
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files (images, css, etc.)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Ensure the static folder exists
    os.makedirs('static', exist_ok=True)
    # Run the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
