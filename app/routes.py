from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from werkzeug.utils import secure_filename
from app.model import classify_tweet_batch, preprocess_text, sentiment_model, sentiment_tokenizer, cyberbullying_model, cyberbullying_tokenizer
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import chardet
from io import StringIO
import mysql.connector
from app.utils import read_csv_with_encoding
from io import BytesIO
import base64
from wordcloud import WordCloud
from app import model
import swifter
from app.rag_embedder import store_tweets_in_chroma
from app.rag_chatbot import rag_chatbot_query
import math

app = Flask(__name__)

# 🔹 Replace with your MySQL credentials
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mycyberbully'

# Set up MySQL connection
def get_db_connection():
    return mysql.connector.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB']
    )

app.secret_key = "301be1d2e369afc52d3cd35c1d5e6789"

@app.route("/", methods=["GET", "POST"])
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

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "email" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))

    if session["role"] == "admin":
        return redirect(url_for("upload_file"))  # Redirect admin to upload page
    elif session["role"] == "policymaker":
        return redirect(url_for("visualise"))
    
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        month = request.form.get("month")
        file = request.files["file"]

        if not file or file.filename == "" or not allowed_file(file.filename):
            flash("Invalid file", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Process in chunks to save RAM
            chunksize = 100000  # 100k tweets at a time
            batch_size = 256    # Sentiment + Cyberbullying batch size

            all_texts = []
            all_sentiments = []
            all_cyberbullying = []
            total_rows = 0

            for chunk in pd.read_csv(filepath, chunksize=chunksize):
                if "text" not in chunk.columns:
                    flash("CSV must contain a 'text' column", "danger")
                    return redirect(request.url)

                texts = chunk["text"].astype(str).tolist()

                # Batch classify in sub-batches
                results = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_results = classify_tweet_batch(
                        batch,
                        sentiment_model=sentiment_model,
                        sentiment_tokenizer=sentiment_tokenizer,
                        cyber_model=cyberbullying_model,
                        cyber_tokenizer=cyberbullying_tokenizer
                    )
                    results.extend(batch_results)

                sentiments = [r[0] for r in results]
                cyber_types = [r[1] for r in results]

                chunk["sentiment"] = sentiments
                chunk["cyberbullying_type"] = cyber_types
                chunk["month"] = month
                chunk["file_name"] = filename

                all_texts.extend(chunk["text"].tolist())
                all_sentiments.extend(sentiments)
                all_cyberbullying.extend(cyber_types)
                total_rows += len(chunk)

                # Insert batch to database (every 100k)
                batch_data = chunk[["text", "sentiment", "cyberbullying_type", "month", "file_name"]].values.tolist()
                cursor.executemany("""
                    INSERT INTO tweets (text, sentiment, cyberbullying_type, month, file_name)
                    VALUES (%s, %s, %s, %s, %s)
                """, batch_data)
                conn.commit()

            # Embed all into ChromaDB (if needed)
            full_df = pd.DataFrame({
                "text": all_texts,
                "sentiment": all_sentiments,
                "cyberbullying_type": all_cyberbullying,
                "month": month,
                "file_name": filename
            })
            store_tweets_in_chroma(full_df)

            cursor.close()
            conn.close()
            flash(f"Successfully processed and saved {total_rows} tweets!", "success")
            return redirect(url_for("visualise", filename=filename))

        except Exception as e:
            print(f"❌ Upload Error: {e}")
            flash("Database error during upload.", "danger")
            return redirect(request.url)

    return render_template("upload.html")

# Route: Visualization
@app.route("/visualise", methods=["GET", "POST"])
def visualise():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT month FROM tweets ORDER BY month")
    months = [row["month"] for row in cursor.fetchall()]

    selected_months = request.form.getlist("months") if request.method == "POST" else [months[-1]]
    chart_type = request.form.get("chart_type") or "bar"

    if not selected_months:
        flash("Please select at least one month.", "warning")
        return redirect(url_for("visualise"))

    query = f"""
        SELECT text, sentiment, cyberbullying_type, month
        FROM tweets
        WHERE month IN ({','.join(['%s'] * len(selected_months))})
    """
    cursor.execute(query, selected_months)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    cursor.close()
    conn.close()

    if df.empty:
        flash("No data found for selected month(s).", "info")
        return render_template("visualisation.html", months=months, selected_months=selected_months)

    # Count stats
    sentiment_positive_count = df[df["sentiment"] == "Positive"].shape[0]
    sentiment_negative_count = df[df["sentiment"] == "Negative"].shape[0]
    cyberbullying_count = df[df["cyberbullying_type"] != "not_cyberbullying"].shape[0]
    non_cyberbullying_count = df[df["cyberbullying_type"] == "not_cyberbullying"].shape[0]

    # Sentiment chart
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_fig = go.Figure(data=[
        go.Bar(x=sentiment_counts.index, y=sentiment_counts.values)
    ] if chart_type == "bar" else [
        go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)
    ])
    sentiment_fig.update_layout(title="Sentiment Distribution")

    # Cyber chart
    cyber_counts = df["cyberbullying_type"].value_counts()
    cyber_fig = go.Figure(data=[
        go.Bar(x=cyber_counts.index, y=cyber_counts.values)
    ] if chart_type == "bar" else [
        go.Pie(labels=cyber_counts.index, values=cyber_counts.values)
    ])
    cyber_fig.update_layout(title="Cyberbullying Distribution")

    # Summary Table
    summary_table = []
    prev = {}
    for month in sorted(selected_months):
        row = {"month": month}
        month_df = df[df["month"] == month]
        counts = month_df["sentiment"].value_counts().to_dict()
        for sent in ["Positive", "Negative"]:
            count = counts.get(sent, 0)
            change = count - prev.get(sent, 0) if prev else 0
            row[sent] = {
                "count": count,
                "change": f"{'+' if change >= 0 else ''}{change}",
                "color": "green" if change > 0 else "red" if change < 0 else "black"
            }
        summary_table.append(row)
        prev = counts

    def generate_key_insights(summary_table):
        insights = []
        if not summary_table or len(summary_table) < 2:
            return ["Not enough data for insights."]
        
        for i in range(1, len(summary_table)):
            current = summary_table[i]
            previous = summary_table[i - 1]
            month = current["month"]

            for sentiment, value in current.items():
                if sentiment == "month":
                    continue

                delta = value["change"]
                if delta is None:
                    continue

                if delta > 10:
                    insights.append(f"In {month}, <strong>{sentiment.lower()}</strong> sentiment increased significantly by {delta}%.")
                elif delta < -10:
                    insights.append(f"In {month}, <strong>{sentiment.lower()}</strong> sentiment dropped by {abs(delta)}%.")

        return insights
    
    key_insights = generate_key_insights(summary_table)

    df["clean_text"] = df["text"].apply(preprocess_text)
# Word clouds
    wordclouds = {}
    for cb_type in df["cyberbullying_type"].dropna().unique():
        # Filter by cyberbullying type and use cleaned text
        texts = df[df["cyberbullying_type"] == cb_type]["clean_text"].dropna().str.cat(sep=" ")
        
        if not texts.strip():
            continue  # Skip if no content to visualize

        wc = WordCloud(width=500, height=300, background_color="white").generate(texts)
        
        buffer = BytesIO()
        wc.to_image().save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        wordclouds[cb_type] = img_str

    # Policy brief
    top_sentiment = df["sentiment"].value_counts().idxmax()
    top_cb_type = df["cyberbullying_type"].value_counts().idxmax()
    policy_brief = f"""
    <p><strong>Month(s):</strong> {', '.join(selected_months)}</p>
    <p><strong>Top Sentiment:</strong> {top_sentiment}</p>
    <p><strong>Most Frequent Cyberbullying Type:</strong> {top_cb_type}</p>
    <p>This brief highlights the dominant trends in online discourse and cyberbullying behavior. Please review and take action as necessary.</p>
    """

    return render_template("visualisation.html",
                           months=months,
                           selected_months=selected_months,
                           chart_type=chart_type,
                           sentiment_plot=sentiment_fig.to_html(),
                           cyber_plot=cyber_fig.to_html(),
                           sentiment_positive_count=sentiment_positive_count,
                           sentiment_negative_count=sentiment_negative_count,
                           cyberbullying_count=cyberbullying_count,
                           non_cyberbullying_count=non_cyberbullying_count,
                           sentiment_data=df[["text", "sentiment", "month"]],
                           cyberbullying_data=df[["text", "cyberbullying_type", "month"]],
                           summary_table=summary_table,
                           key_insights=key_insights,
                           wordclouds=wordclouds,
                           policy_brief=policy_brief)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_question = request.json.get("message")
    if not user_question:
        return jsonify({"answer": "Please provide a valid question."})

    answer = rag_chatbot_query(user_question)
    return jsonify({"answer": answer})

@app.route('/download/<filename>')
def download_file(filename):
    directory = os.path.join(app.config['UPLOAD_FOLDER'])
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        flash("Requested file not found.", "danger")
        return redirect(url_for("upload_file"))
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
