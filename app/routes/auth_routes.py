from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app.utils import get_db_connection, allowed_file
from flask import current_app, send_from_directory
import mysql.connector

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role")

        print(f"🔍 Received login request: {email} | Role: {role}")

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            print("✅ Connected to MySQL successfully!")

            if role == "admin":
                query = "SELECT * FROM admin WHERE admin_email = %s AND admin_pwrd = %s"
            elif role == "policymaker":
                query = "SELECT * FROM policymaker WHERE pm_email = %s AND pm_pwrd = %s"
            else:
                flash("Invalid role!", "danger")
                return redirect(url_for("login"))

            print("🔍 Executing query:", query)
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
            print(f"🔍 Query Result: {user}")

            cursor.close()
            conn.close()
            print("✅ MySQL Connection closed successfully!")

            if user:
                session["email"] = email
                session["role"] = role
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials!", "danger")

        except mysql.connector.Error as err:
            print(f"❌ MySQL Error: {err}")  # Log MySQL errors
            flash("Database connection error!", "danger")

        except Exception as e:
            print(f"❌ Unexpected Error: {e}")  # Catch unexpected errors
            flash("Unexpected error occurred!", "danger")

    return render_template("auth/login.html")

@auth_bp.route("/dashboard")
def dashboard():
    if "email" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))

    if session["role"] == "admin":
        return redirect(url_for("upload_file"))  # Redirect admin to upload page
    elif session["role"] == "policymaker":
        return redirect(url_for("visualise"))
    
    return redirect(url_for("login"))

@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))