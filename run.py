from flask import Flask, render_template
from app.routes import register_blueprints
import os

app = Flask(__name__)

# Secret key
app.secret_key = "301be1d2e369afc52d3cd35c1d5e6789"

# MySQL Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mycyberbully'

# Upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Register Blueprints
register_blueprints(app)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error/500.html'), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
