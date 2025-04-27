from flask import Blueprint

# Import each route blueprint
from app.routes.auth_routes import auth_bp
from app.routes.admin_routes import admin_bp
from app.routes.policymaker_routes import policymaker_bp

def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(policymaker_bp)
