from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

# Route for home page displaying the image
@app.route('/')
def home():
    # Simple HTML that references the image in the static folder
    html = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Image Display</title>
      </head>
      <body>
        <h1>My Image</h1>
        <img src="/static/image.jpg" alt="Image" style="max-width:100%; height:auto;" />
      </body>
    </html>
    '''
    return render_template_string(html)

# Optional: serve static files (Flask does this automatically for the 'static' folder)
# If you need a custom static route:
# @app.route('/static/<path:filename>')
# def static_files(filename):
#     return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)
