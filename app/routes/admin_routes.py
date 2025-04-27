from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import os
import pandas as pd
from werkzeug.utils import secure_filename
from app.model import classify_tweet_batch, sentiment_model, sentiment_tokenizer, cyberbullying_model, cyberbullying_tokenizer
from app.utils import read_csv_with_encoding, get_db_connection, allowed_file
from app.rag_embedder import store_tweets_in_chroma
from flask import current_app, send_from_directory

admin_bp = Blueprint('admin', __name__)

UPLOAD_FOLDER = "uploads"

@admin_bp.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        month = request.form.get("month")
        file = request.files["file"]

        if not file or file.filename == "" or not allowed_file(file.filename):
            flash("Invalid file", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
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

    return render_template("admin/upload.html")
