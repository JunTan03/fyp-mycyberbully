from flask import Blueprint, render_template, request, jsonify, flash, url_for, redirect, current_app
import pandas as pd
import os
from app.model import preprocess_text
from app.rag_chatbot import rag_chatbot_query
from app.utils import get_db_connection, allowed_file
from io import BytesIO
import base64
from wordcloud import WordCloud
import plotly.graph_objects as go
from flask import current_app, send_from_directory

policymaker_bp = Blueprint('policymaker', __name__)

@policymaker_bp.route("/visualise", methods=["GET", "POST"])
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
        return render_template("visusalisation/visualisation.html", months=months, selected_months=selected_months)

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

    return render_template("visualisation/visualisation.html",
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

@policymaker_bp.route("/chatbot", methods=["POST"])
def chatbot():
    user_question = request.json.get("message")
    if not user_question:
        return jsonify({"answer": "Please provide a valid question."})
    answer = rag_chatbot_query(user_question)
    return jsonify({"answer": answer})

@policymaker_bp.route('/download/<filename>')
def download_file(filename):
    directory = os.path.join(current_app.config['UPLOAD_FOLDER'])
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        flash("Requested file not found.", "danger")
        return redirect(url_for("upload_file"))
    return send_from_directory(directory, filename, as_attachment=True)
