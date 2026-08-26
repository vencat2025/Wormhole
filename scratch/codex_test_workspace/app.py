from flask import Flask, render_template_string
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''\
<!DOCTYPE html>\
<html lang="en">\
<head>\
    <meta charset="UTF-8">\
    <title>Image Page</title>\
</head>\
<body>\
    <img src="/static/image.jpg" alt="Sample Image">\
</body>\
</html>''')

if __name__ == '__main__':\
    app.run(debug=True)
