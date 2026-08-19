from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask import send_from_directory
from datetime import datetime
from groq import Groq
import pandas as pd
import io
import os
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler   # ganti StandardScaler
from kneed import KneeLocator 
from flask import send_file
from sklearn.metrics import silhouette_score

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

AI_KEY = os.getenv('GROQ_API_KEY', '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')

client = Groq(api_key=AI_KEY)

# Global variable to store uploaded data
uploaded_data = None
clustered_data = None   # tambahan baru: menyimpan df_clean lengkap dengan kolom cluster & label

def analyze_business_data(df):
    """Menganalisis data bisnis berdasarkan rating dan jumlah ulasan"""
    try:
        # Pastikan kolom yang diperlukan ada
        required_columns = ['nama', 'rating', 'jumlah_ulasan']
        for col in required_columns:
            if col not in df.columns:
                return None, f"Kolom '{col}' tidak ditemukan dalam file CSV"
        
        # Buat salinan dataframe untuk menghindari SettingWithCopyWarning
        df_clean = df.copy()
        
        # Konversi kolom numerik
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        df_clean['jumlah_ulasan'] = pd.to_numeric(df_clean['jumlah_ulasan'], errors='coerce')
        
        if 'website' in df_clean.columns:
            df_clean['website'] = df_clean['website'].fillna('')
        else:
            df_clean['website'] = ''

        if 'nomor_telepon' not in df_clean.columns:
            df_clean['nomor_telepon'] = ''
            
        if 'nama' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['nama'])
        
        # Hapus baris dengan nilai NaN
        df_clean = df_clean.dropna(subset=['rating', 'jumlah_ulasan'])
        
        if len(df_clean) == 0:
            return None, "Tidak ada data yang valid untuk dianalisis"
        
        # Reset index untuk menghindari masalah indexing
        df_clean = df_clean.reset_index(drop=True)
        
        # Analisis 1: Bisnis dengan rating tertinggi
        if not df_clean.empty and 'rating' in df_clean.columns:
            highest_rated_idx = df_clean['rating'].idxmax()
            highest_rated = df_clean.loc[highest_rated_idx]
        else:
            highest_rated = pd.Series({'nama': 'Tidak ada data', 'rating': 0, 'jumlah_ulasan': 0, 'kategori_usaha': 'Tidak tersedia'})
        
        # Analisis 2: Bisnis dengan jumlah ulasan terbanyak
        if not df_clean.empty and 'jumlah_ulasan' in df_clean.columns:
            most_reviewed_idx = df_clean['jumlah_ulasan'].idxmax()
            most_reviewed = df_clean.loc[most_reviewed_idx]
        else:
            most_reviewed = pd.Series({'nama': 'Tidak ada data', 'rating': 0, 'jumlah_ulasan': 0, 'kategori_usaha': 'Tidak tersedia'})
        
        # Analisis 3: Bisnis dengan rating terendah (minimal 10 ulasan)
        low_rated_filtered = df_clean[df_clean['jumlah_ulasan'] >= 10]
        if len(low_rated_filtered) > 0:
            lowest_rated_idx = low_rated_filtered['rating'].idxmin()
            lowest_rated = low_rated_filtered.loc[lowest_rated_idx]
        elif len(df_clean) > 0:
            lowest_rated_idx = df_clean['rating'].idxmin()
            lowest_rated = df_clean.loc[lowest_rated_idx]
        else:
            lowest_rated = pd.Series({'nama': 'Tidak ada data', 'rating': 0, 'jumlah_ulasan': 0, 'kategori_usaha': 'Tidak tersedia'})
        
        # Analisis 4: Rata-rata rating dan ulasan
        avg_rating = df_clean['rating'].mean() if not df_clean.empty else 0
        avg_reviews = df_clean['jumlah_ulasan'].mean() if not df_clean.empty else 0
        
        # Analisis 5: Kategori usaha unik
        categories = {}
        if 'kategori_usaha' in df_clean.columns and not df_clean.empty:
            categories = df_clean['kategori_usaha'].value_counts().head(10).to_dict()
        
        # Analisis 6: Top 10 bisnis berdasarkan rating
        top_10_rating = []
        if not df_clean.empty:
            top_10_rating_df = df_clean.nlargest(10, 'rating')
            # Pastikan kolom yang diperlukan ada
            columns_to_include = ['nama', 'rating', 'jumlah_ulasan', 'nomor_telepon', 'website']
            if 'kategori_usaha' in top_10_rating_df.columns:
                columns_to_include.append('kategori_usaha')
            top_10_rating = top_10_rating_df[columns_to_include].to_dict('records')
        
        # Analisis 7: Top 10 bisnis berdasarkan jumlah ulasan
        top_10_reviews = []
        if not df_clean.empty:
            top_10_reviews_df = df_clean.nlargest(10, 'jumlah_ulasan')
            columns_to_include = ['nama', 'rating', 'jumlah_ulasan', 'nomor_telepon', 'website']
            if 'kategori_usaha' in top_10_reviews_df.columns:
                columns_to_include.append('kategori_usaha')
            top_10_reviews = top_10_reviews_df[columns_to_include].to_dict('records')
            
        # Analisis 8: Distribusi rating
        rating_distribution = {}
        if not df_clean.empty:
            rating_distribution = df_clean['rating'].value_counts().sort_index().to_dict()
        
        # Analisis 9: Statistik lengkap
        stats = {
            'total_businesses': len(df_clean),
            'avg_rating': round(avg_rating, 2) if not pd.isna(avg_rating) else 0,
            'avg_reviews': round(avg_reviews, 2) if not pd.isna(avg_reviews) else 0,
            'max_rating': round(df_clean['rating'].max(), 2) if not df_clean.empty else 0,
            'min_rating': round(df_clean['rating'].min(), 2) if not df_clean.empty else 0,
            'max_reviews': int(df_clean['jumlah_ulasan'].max()) if not df_clean.empty else 0,
            'min_reviews': int(df_clean['jumlah_ulasan'].min()) if not df_clean.empty else 0,
            'total_reviews': int(df_clean['jumlah_ulasan'].sum()) if not df_clean.empty else 0
        }

        # Analisis 10: K-means clustering berdasarkan rating, jumlah ulasan, dan status website
        cluster_analysis = {
            'enabled': False,
            'num_clusters': 0,
            'clusters': [],
            'cluster_centroids': [],
            'cluster_sample': [],
            'silhouette_score': None,
            'elbow_k': None
        }

        try:
            # Siapkan atribut website_status (1 jika ada website, 0 jika kosong)
            df_clean['website_status'] = df_clean['website'].apply(
                lambda x: 1 if isinstance(x, str) and x.strip() != '' else 0
            )

            # Minimal data untuk clustering yang bermakna
            if len(df_clean) >= 6:
                features = df_clean[['rating', 'jumlah_ulasan', 'website_status']].copy()

                if features['rating'].nunique() >= 2 or features['jumlah_ulasan'].nunique() >= 2:
                    # --- Normalisasi (sesuai Colab: MinMaxScaler) ---
                    scaler = MinMaxScaler()
                    features_scaled = scaler.fit_transform(features)

                    # --- Elbow Method otomatis ---
                    max_k = min(10, len(df_clean) - 1)  # batasi K agar tidak melebihi jumlah data
                    k_range = list(range(1, max_k + 1))
                    inertia = []

                    for k_test in k_range:
                        km_test = KMeans(
                            n_clusters=k_test,
                            random_state=42,
                            n_init=10,
                            init='k-means++'
                        )
                        km_test.fit(features_scaled)
                        inertia.append(km_test.inertia_)

                    # Deteksi titik elbow; fallback ke K=2 jika gagal terdeteksi atau data sedikit
                    elbow_k = None
                    if len(k_range) >= 3:
                        try:
                            knee = KneeLocator(
                                k_range, inertia,
                                curve='convex', direction='decreasing'
                            )
                            elbow_k = knee.elbow
                        except Exception:
                            elbow_k = None

                    if elbow_k is None or elbow_k < 2:
                        elbow_k = 2 if max_k >= 2 else 1

                    # --- Validasi dengan Silhouette Score ---
                    best_k = elbow_k
                    best_score = None

                    if max_k >= 2:
                        silhouette_results = {}
                        for k_test in range(2, max_k + 1):
                            km_test = KMeans(
                                n_clusters=k_test,
                                random_state=42,
                                n_init=10,
                                init='k-means++'
                            )
                            labels_test = km_test.fit_predict(features_scaled)
                            score = silhouette_score(features_scaled, labels_test)
                            silhouette_results[k_test] = score

                        # Pilih K dengan silhouette tertinggi sebagai validasi akhir
                        best_k = max(silhouette_results, key=silhouette_results.get)
                        best_score = silhouette_results[best_k]

                    n_clusters = best_k

                    # --- K-Means final dengan K tervalidasi ---
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10,
                        init='k-means++',
                        max_iter=300
                    )
                    labels = kmeans.fit_predict(features_scaled)
                    df_clean['cluster'] = labels

                    if best_score is None and n_clusters >= 2:
                        best_score = silhouette_score(features_scaled, labels)

                    cluster_stats = df_clean.groupby('cluster').agg(
                        count=('nama', 'count'),
                        avg_rating=('rating', 'mean'),
                        avg_reviews=('jumlah_ulasan', 'mean'),
                        avg_website=('website_status', 'mean')
                    ).reset_index()

                    # Urutkan cluster berdasarkan rata-rata jumlah ulasan (indikator utama potensi,
                    # konsisten dengan pendekatan ranking di notebook Colab)
                    cluster_stats = cluster_stats.sort_values(
                        by=['avg_reviews', 'avg_rating'], ascending=False
                    ).reset_index(drop=True)

                    # Label dinamis mengikuti jumlah cluster (bukan fixed 3 label lagi)
                    label_map = {}
                    total_clusters = len(cluster_stats)
                    for rank, (_, row) in enumerate(cluster_stats.iterrows()):
                        if total_clusters == 2:
                            name = 'Potensi Tinggi' if rank == 0 else 'Potensi Standar'
                        elif rank == 0:
                            name = 'Potensi Tinggi'
                        elif rank == total_clusters - 1:
                            name = 'Potensi Rendah'
                        else:
                            name = 'Potensi Menengah'
                        label_map[int(row['cluster'])] = name

                    cluster_summary = []
                    for _, row in cluster_stats.iterrows():
                        cluster_summary.append({
                            'cluster_id': int(row['cluster']),
                            'label': label_map[int(row['cluster'])],
                            'count': int(row['count']),
                            'avg_rating': round(float(row['avg_rating']), 2),
                            'avg_reviews': round(float(row['avg_reviews']), 2),
                            'avg_website': round(float(row['avg_website']), 2)
                        })

                    # Centroid dikembalikan ke skala asli (inverse_transform sesuai MinMaxScaler)
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    centroid_list = []
                    for cluster_id, centroid in enumerate(centroids):
                        centroid_list.append({
                            'cluster_id': cluster_id,
                            'avg_rating': round(float(centroid[0]), 2),
                            'avg_reviews': round(float(centroid[1]), 2),
                            'avg_website': round(float(centroid[2]), 2)
                        })

                    sample_businesses = []
                    for cluster in cluster_summary:
                        sample = df_clean[df_clean['cluster'] == cluster['cluster_id']]
                        sample = sample.sort_values(
                            by=['rating', 'jumlah_ulasan'], ascending=[False, False]
                        ).head(3)
                        sample_businesses.append({
                            'cluster_id': cluster['cluster_id'],
                            'label': cluster['label'],
                            'top_businesses': sample['nama'].tolist()
                        })

                    cluster_analysis = {
                        'enabled': True,
                        'num_clusters': n_clusters,
                        'clusters': cluster_summary,
                        'cluster_centroids': centroid_list,
                        'cluster_sample': sample_businesses,
                        'silhouette_score': round(float(best_score), 4) if best_score is not None else None,
                        'elbow_k': elbow_k
                    }
                    
        except Exception as clustering_error:
            print(f"KMeans clustering skipped: {clustering_error}")
            cluster_analysis = {
                'enabled': False,
                'num_clusters': 0,
                'clusters': [],
                'cluster_centroids': [],
                'cluster_sample': [],
                'silhouette_score': None,
                'elbow_k': None
            }
        
        # Handle missing values in the series
        def get_series_value(series, key, default='Tidak tersedia'):
            try:
                value = series.get(key, default)
                return value if not pd.isna(value) else default
            except:
                return default
            
         # Siapkan data segmentasi final (untuk ditampilkan terurut per cluster)
        segmentation_final = []
        if cluster_analysis.get('enabled'):
            label_lookup = {c['cluster_id']: c['label'] for c in cluster_analysis['clusters']}
            df_seg = df_clean.copy()
            df_seg['kategori_cluster'] = df_seg['cluster'].map(label_lookup)

            # Urutkan: cluster dengan rata-rata ulasan tertinggi tampil duluan,
            # lalu di dalam cluster diurutkan dari rating & ulasan tertinggi (sama seperti Colab)
            cluster_order = [c['cluster_id'] for c in cluster_analysis['clusters']]  # sudah terurut desc
            df_seg['cluster_rank'] = df_seg['cluster'].apply(lambda x: cluster_order.index(x))
            df_seg = df_seg.sort_values(
                by=['cluster_rank', 'rating', 'jumlah_ulasan'],
                ascending=[True, False, False]
            ).drop(columns=['cluster_rank'])

            segmentation_final = df_seg[[
                'nama', 'rating', 'jumlah_ulasan', 'website_status',
                'cluster', 'kategori_cluster'
            ]].to_dict('records')
        
        results = {
            'highest_rated': {
                'nama': get_series_value(highest_rated, 'nama', 'Tidak ada data'),
                'rating': round(float(get_series_value(highest_rated, 'rating', 0)), 2),
                'jumlah_ulasan': int(get_series_value(highest_rated, 'jumlah_ulasan', 0)),
                'kategori': get_series_value(highest_rated, 'kategori_usaha', 'Tidak tersedia')
            },
            'most_reviewed': {
                'nama': get_series_value(most_reviewed, 'nama', 'Tidak ada data'),
                'rating': round(float(get_series_value(most_reviewed, 'rating', 0)), 2),
                'jumlah_ulasan': int(get_series_value(most_reviewed, 'jumlah_ulasan', 0)),
                'kategori': get_series_value(most_reviewed, 'kategori_usaha', 'Tidak tersedia')
            },
            'lowest_rated': {
                'nama': get_series_value(lowest_rated, 'nama', 'Tidak ada data'),
                'rating': round(float(get_series_value(lowest_rated, 'rating', 0)), 2),
                'jumlah_ulasan': int(get_series_value(lowest_rated, 'jumlah_ulasan', 0)),
                'kategori': get_series_value(lowest_rated, 'kategori_usaha', 'Tidak tersedia')
            },
            'statistics': stats,
            'categories': categories,
            'top_10_rating': top_10_rating,
            'top_10_reviews': top_10_reviews,
            'rating_distribution': rating_distribution,
            'cluster_analysis': cluster_analysis,
            'segmentation_final': segmentation_final,
            'raw_data': df_clean.to_dict('records')  # Semua data bersih
        }
        
        
        return results, None
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_business_data: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error dalam menganalisis data: {str(e)}"

def ai_call(df, analysis_results):
    """Meminta analisis AI berdasarkan data dan hasil analisis"""
    try:
        # Siapkan ringkasan data untuk AI
        data_summary = f"""
        Data bisnis yang dianalisis:
        - Total bisnis: {analysis_results['statistics']['total_businesses']}
        - Rata-rata rating: {analysis_results['statistics']['avg_rating']}
        - Rata-rata jumlah ulasan: {analysis_results['statistics']['avg_reviews']}
        - Total ulasan: {analysis_results['statistics']['total_reviews']}
        
        Bisnis dengan rating tertinggi: {analysis_results['highest_rated']['nama']} 
        (Rating: {analysis_results['highest_rated']['rating']}, 
        Ulasan: {analysis_results['highest_rated']['jumlah_ulasan']},
        Kategori: {analysis_results['highest_rated']['kategori']})
        
        Bisnis dengan ulasan terbanyak: {analysis_results['most_reviewed']['nama']}
        (Rating: {analysis_results['most_reviewed']['rating']}, 
        Ulasan: {analysis_results['most_reviewed']['jumlah_ulasan']},
        Kategori: {analysis_results['most_reviewed']['kategori']})
        
        Bisnis dengan rating terendah: {analysis_results['lowest_rated']['nama']}
        (Rating: {analysis_results['lowest_rated']['rating']}, 
        Ulasan: {analysis_results['lowest_rated']['jumlah_ulasan']},
        Kategori: {analysis_results['lowest_rated']['kategori']})
        """
        if analysis_results.get('cluster_analysis', {}).get('enabled'):
            cluster_lines = ['\nHasil clustering k-means:']
            cluster_lines.append(f"- Jumlah cluster: {analysis_results['cluster_analysis']['num_clusters']}")
            for cluster in analysis_results['cluster_analysis']['clusters']:
                cluster_lines.append(
                    f"- Cluster {cluster['cluster_id']} ({cluster['label']}): {cluster['count']} bisnis, rata-rata rating {cluster['avg_rating']}, rata-rata ulasan {cluster['avg_reviews']}"
                )
            for sample in analysis_results['cluster_analysis'].get('cluster_sample', []):
                if sample.get('top_businesses'):
                    cluster_lines.append(
                        f"  Contoh bisnis cluster {sample['cluster_id']} ({sample['label']}): {', '.join(sample['top_businesses'])}"
                    )
            data_summary += '\n'.join(cluster_lines)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": f"""
                    Berikan analisis insights bisnis berdasarkan data berikut:
                    
                    {data_summary}
                    
                    Berikan analisis tentang:
                    1. Kualitas layanan secara keseluruhan berdasarkan distribusi rating
                    2. Popularitas bisnis berdasarkan jumlah ulasan
                    3. Rekomendasi strategi untuk meningkatkan rating dan ulasan
                    4. Pola atau tren yang terlihat dari data
                    5. Insight tentang kategori usaha yang performa baik
                    6. Saran improvement untuk bisnis dengan rating rendah
                    7. Interpretasi hasil clustering k-means: jelaskan karakteristik setiap cluster dan rekomendasi operasional untuk masing-masing
                    
                    Format respons dalam bahasa Indonesia dengan struktur yang jelas dan actionable insights.
                    """
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        ai_output = chat_completion.choices[0].message.content
    
        return ai_output
    
    except Exception as e:
        print(f"Error AI call: {e}")
        return "Maaf, terjadi kesalahan dalam menganalisis data dengan AI."

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_business():
    global uploaded_data
    
    if request.method == 'POST':
        # Cek apakah file ada dalam request
        if 'csv_file' not in request.files:
            flash('Tidak ada file yang diupload', 'error')
            return redirect(request.url)
        
        file = request.files['csv_file']
        
        # Cek jika user tidak memilih file
        if file.filename == '':
            flash('Silakan pilih file CSV', 'error')
            return redirect(request.url)
        
        # Cek ekstensi file
        if not file.filename.endswith('.csv'):
            flash('Silakan upload file dengan format CSV', 'error')
            return redirect(request.url)
        
        try:
            df = pd.read_csv(file)
            uploaded_data = df.copy()

            analysis_results, error = analyze_business_data(df)

            if error:
                flash(error, 'error')
                return redirect(request.url)

            # simpan hasil segmentasi final untuk dipakai di /get_all_data
            clustered_data = analysis_results.get('segmentation_final', [])

            ai_output = ai_call(df, analysis_results)
            
            flash(f'Berhasil memuat {len(df)} data bisnis!', 'success')
            
            return render_template('analyze_business.html', 
                                 results=analysis_results, 
                                 ai_output=ai_output,
                                 analysis_success=True,
                                 total_data=len(df))
            
        except Exception as e:
            flash(f'Error membaca file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('analyze_business.html', 
                         results=None, 
                         analysis_success=False,
                         total_data=0)

@app.route('/close_analysis', methods=['POST'])
def close_analysis():
    """Route untuk menutup hasil analisis dan mereset data"""
    global uploaded_data
    
    # Reset global data
    uploaded_data = None
    clustered_data = None
    
    flash('Analisis telah ditutup. Anda dapat mengupload file baru untuk analisis.', 'info')
    return redirect(url_for('analyze_business'))

@app.route('/get_all_data', methods=['GET'])
def get_all_data():
    global uploaded_data, clustered_data

    if uploaded_data is None:
        return jsonify({'error': 'Tidak ada data yang diupload'})

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))

        # Prioritaskan data hasil segmentasi (sudah terurut per cluster),
        # fallback ke data mentah jika clustering tidak tersedia
        if clustered_data:
            data_source = clustered_data
            total_data = len(data_source)
            total_pages = math.ceil(total_data / per_page)
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_data)
            data_records = data_source[start_idx:end_idx]
        else:
            df_sorted = uploaded_data.copy()
            if "rating" in df_sorted.columns and "jumlah_ulasan" in df_sorted.columns:
                df_sorted = df_sorted.sort_values(
                    by=["rating", "jumlah_ulasan"],
                    ascending=[False, False]
                )
            total_data = len(df_sorted)
            total_pages = math.ceil(total_data / per_page)
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_data)
            data_chunk = df_sorted.iloc[start_idx:end_idx]

            data_records = []
            for _, row in data_chunk.iterrows():
                record = {}
                for col in df_sorted.columns:
                    value = row[col]
                    record[col] = None if pd.isna(value) else value
                data_records.append(record)

        return jsonify({
            'success': True,
            'data': data_records,
            'has_cluster': bool(clustered_data),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_data': total_data
            }
        })

    except Exception as e:
        print(f"Get all data error: {e}")
        return jsonify({'error': f'Error mengambil data: {str(e)}'})

    
@app.route('/download_results', methods=['GET'])
def download_results():
    global uploaded_data
    
    if uploaded_data is None:
        flash("Tidak ada data yang dianalisis untuk diunduh.", "error")
        return redirect(url_for('analyze_business'))

    try:
        # Jalankan kembali analisis agar sinkron dengan yang ditampilkan
        analysis_results, error = analyze_business_data(uploaded_data)
        if error:
            flash(error, "error")
            return redirect(url_for('analyze_business'))

        # Buat list dict sesuai urutan hasil analisis
        export_data = []

        # Ringkasan utama
        export_data.append({
            "Bagian": "Ringkasan",
            "Keterangan": "Total Bisnis",
            "Nilai": analysis_results['statistics']['total_businesses']
        })
        export_data.append({
            "Bagian": "Ringkasan",
            "Keterangan": "Rata-rata Rating",
            "Nilai": analysis_results['statistics']['avg_rating']
        })
        export_data.append({
            "Bagian": "Ringkasan",
            "Keterangan": "Rata-rata Ulasan",
            "Nilai": analysis_results['statistics']['avg_reviews']
        })
        export_data.append({
            "Bagian": "Ringkasan",
            "Keterangan": "Total Ulasan",
            "Nilai": analysis_results['statistics']['total_reviews']
        })

        # Bisnis dengan performa tertentu
        export_data.append({
            "Bagian": "Bisnis Tertinggi",
            "Keterangan": analysis_results['highest_rated']['nama'],
            "Nilai": f"Rating: {analysis_results['highest_rated']['rating']} | Ulasan: {analysis_results['highest_rated']['jumlah_ulasan']} | Kategori: {analysis_results['highest_rated']['kategori']}"
        })
        export_data.append({
            "Bagian": "Bisnis Ulasan Terbanyak",
            "Keterangan": analysis_results['most_reviewed']['nama'],
            "Nilai": f"Rating: {analysis_results['most_reviewed']['rating']} | Ulasan: {analysis_results['most_reviewed']['jumlah_ulasan']} | Kategori: {analysis_results['most_reviewed']['kategori']}"
        })
        export_data.append({
            "Bagian": "Bisnis Terendah",
            "Keterangan": analysis_results['lowest_rated']['nama'],
            "Nilai": f"Rating: {analysis_results['lowest_rated']['rating']} | Ulasan: {analysis_results['lowest_rated']['jumlah_ulasan']} | Kategori: {analysis_results['lowest_rated']['kategori']}"
        })

        # Top 10 Rating
        for idx, row in enumerate(analysis_results['top_10_rating'], start=1):
            export_data.append({
                "Bagian": "Top 10 Rating",
                "Keterangan": f"{idx}. {row['nama']}",
                "Nilai": f"Rating: {row['rating']} | Ulasan: {row['jumlah_ulasan']} | Kategori: {row.get('kategori_usaha', '-')} | Telepon: {row.get('nomor_telepon', '-')} | Website: {row.get('website', '-')} "
            })

        # Top 10 Ulasan
        for idx, row in enumerate(analysis_results['top_10_reviews'], start=1):
            export_data.append({
                "Bagian": "Top 10 Ulasan",
                "Keterangan": f"{idx}. {row['nama']}",
                "Nilai": f"Rating: {row['rating']} | Ulasan: {row['jumlah_ulasan']} | Kategori: {row.get('kategori_usaha', '-')} | Telepon: {row.get('nomor_telepon', '-')} | Website: {row.get('website', '-')}"
            })

        if analysis_results.get('cluster_analysis', {}).get('enabled'):
            export_data.append({
                "Bagian": "Clustering K-Means",
                "Keterangan": "Jumlah cluster",
                "Nilai": analysis_results['cluster_analysis']['num_clusters']
            })
            for cluster in analysis_results['cluster_analysis']['clusters']:
                export_data.append({
                    "Bagian": "Clustering K-Means",
                    "Keterangan": f"Cluster {cluster['cluster_id']} ({cluster['label']})",
                    "Nilai": f"Count: {cluster['count']} | Avg Rating: {cluster['avg_rating']} | Avg Reviews: {cluster['avg_reviews']}"
                })

        # Convert ke DataFrame
        df_export = pd.DataFrame(export_data)

        # Simpan ke buffer
        buf = io.StringIO()
        df_export.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)

        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"hasil_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    except Exception as e:
        flash(f"Gagal membuat file CSV: {str(e)}", "error")
        return redirect(url_for('analyze_business'))

@app.route('/download_template')
def download_template():
    try:
        # Buat template CSV sesuai kolom wajib
        columns = ["nama", "nomor_telepon", "kategori_usaha", "lokasi", "rating", "jumlah_ulasan", "email", "website"]
        df_template = pd.DataFrame(columns=columns)

        # Simpan ke buffer
        buffer = io.BytesIO()
        df_template.to_csv(buffer, index=False, encoding="utf-8-sig")
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="template_data_bisnis.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        flash(f"Error membuat template CSV: {str(e)}", "error")
        return redirect(url_for("analyze_business"))
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
