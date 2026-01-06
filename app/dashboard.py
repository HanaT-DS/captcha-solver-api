"""
CAPTCHA Factory - Dashboard Streamlit
=====================================
Interface utilisateur interactive pour la demonstration du projet.

Usage:
    streamlit run app/dashboard.py

M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne
"""

import io
import time
import base64
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ajouter le path parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.captcha_generator import CaptchaGenerator
from app.services.solver_service import SolverService


# =============================================================================
# Configuration de la page
# =============================================================================

st.set_page_config(
    page_title="CAPTCHA Factory",
    page_icon="🔓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialisation des services
# =============================================================================

@st.cache_resource
def load_services():
    """Charge les services (cache pour eviter de recharger)."""
    generator = CaptchaGenerator()
    solver = SolverService()
    return generator, solver


generator, solver = load_services()


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def image_to_base64(image: Image.Image) -> str:
    """Convertit une image PIL en base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def solve_image(image: Image.Image, model: str = "easyocr") -> dict:
    """Resout un CAPTCHA."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return solver.solve(buffer.getvalue(), model=model)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/f/f0/Logo-panth%C3%A9on-sorbonne.svg/1200px-Logo-panth%C3%A9on-sorbonne.svg.png", width=150)
    st.markdown("### M2 MoSEF")
    st.markdown("Webscraping & API")
    
    st.markdown("---")
    
    # Statistiques de session
    st.markdown("### Statistiques")
    
    if "total_generated" not in st.session_state:
        st.session_state.total_generated = 0
    if "total_solved" not in st.session_state:
        st.session_state.total_solved = 0
    if "correct_predictions" not in st.session_state:
        st.session_state.correct_predictions = 0
    
    col1, col2 = st.columns(2)
    col1.metric("Generes", st.session_state.total_generated)
    col2.metric("Resolus", st.session_state.total_solved)
    
    if st.session_state.total_solved > 0:
        accuracy = st.session_state.correct_predictions / st.session_state.total_solved * 100
        st.metric("Precision", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    st.markdown("### Modeles disponibles")
    for model in solver.get_available_models():
        st.markdown(f"- {model.upper()}")
    
    st.markdown("---")
    
    if st.button("Reinitialiser les stats"):
        st.session_state.total_generated = 0
        st.session_state.total_solved = 0
        st.session_state.correct_predictions = 0
        st.rerun()


# =============================================================================
# Contenu principal
# =============================================================================

st.markdown('<h1 class="main-header">CAPTCHA Factory</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Generation, Resolution & Comparaison de CAPTCHAs</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Generation & Resolution",
    "Upload & Resolution",
    "Comparaison des Modeles",
    "Benchmark"
])


# =============================================================================
# Tab 1: Generation & Resolution
# =============================================================================

with tab1:
    st.markdown("### Generer et resoudre un CAPTCHA")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Parametres")
        
        length = st.slider("Nombre de caracteres", 3, 8, 5, key="gen_length")
        width = st.slider("Largeur (px)", 150, 300, 200, key="gen_width")
        height = st.slider("Hauteur (px)", 40, 100, 60, key="gen_height")
        noise = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, key="gen_noise")
        
        model_choice = st.selectbox(
            "Modele de resolution",
            solver.get_available_models(),
            key="gen_model"
        )
        
        generate_btn = st.button("Generer & Resoudre", type="primary", use_container_width=True)
    
    with col2:
        if generate_btn:
            with st.spinner("Generation en cours..."):
                # Generer
                image, true_text = generator.generate(
                    length=length,
                    width=width,
                    height=height,
                    noise_level=noise
                )
                st.session_state.total_generated += 1
                
                # Afficher l'image
                st.image(image, caption=f"CAPTCHA genere (Texte: {true_text})")
                
                # Resoudre
                start_time = time.time()
                result = solve_image(image, model=model_choice)
                elapsed = (time.time() - start_time) * 1000
                
                st.session_state.total_solved += 1
                
                # Verifier
                predicted = result["text"]
                is_correct = predicted.lower() == true_text.lower()
                
                if is_correct:
                    st.session_state.correct_predictions += 1
                    st.success(f"CORRECT ! Prediction: `{predicted}`")
                else:
                    st.error(f"ERREUR ! Prediction: `{predicted}` | Reel: `{true_text}`")
                
                # Metriques
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Prediction", predicted)
                col_b.metric("Temps", f"{elapsed:.0f} ms")
                col_c.metric("Confiance", f"{result.get('confidence', 'N/A')}")
        else:
            st.info("Cliquez sur 'Generer & Resoudre' pour commencer")


# =============================================================================
# Tab 2: Upload & Resolution
# =============================================================================

with tab2:
    st.markdown("### Uploader une image CAPTCHA")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=["png", "jpg", "jpeg"],
            key="upload_file"
        )
        
        model_upload = st.selectbox(
            "Modele de resolution",
            solver.get_available_models(),
            key="upload_model"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadee")
    
    with col2:
        if uploaded_file:
            if st.button("Resoudre", type="primary", use_container_width=True):
                with st.spinner("Resolution en cours..."):
                    start_time = time.time()
                    result = solve_image(image, model=model_upload)
                    elapsed = (time.time() - start_time) * 1000
                    
                    st.session_state.total_solved += 1
                    
                    st.markdown("### Resultat")
                    st.markdown(f"""
                    <div style="background-color: #e7f3ff; padding: 2rem; border-radius: 10px; text-align: center;">
                        <h1 style="margin: 0; color: #1f77b4;">{result['text']}</h1>
                        <p style="color: #666;">Resolu en {elapsed:.0f}ms avec {result['model']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result.get("confidence"):
                        st.metric("Confiance", f"{result['confidence']*100:.1f}%")
        else:
            st.info("Uploadez une image pour la resoudre")


# =============================================================================
# Tab 3: Comparaison des Modeles
# =============================================================================

with tab3:
    st.markdown("### Comparer les modeles sur un CAPTCHA")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Parametres du test")
        
        comp_noise = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, key="comp_noise")
        
        if st.button("Generer et Comparer", type="primary", use_container_width=True):
            # Generer un CAPTCHA
            image, true_text = generator.generate(noise_level=comp_noise)
            st.session_state.total_generated += 1
            
            st.session_state.comparison_image = image
            st.session_state.comparison_text = true_text
            st.session_state.comparison_results = {}
            
            # Tester chaque modele
            for model_name in solver.get_available_models():
                start = time.time()
                result = solve_image(image, model=model_name)
                elapsed = (time.time() - start) * 1000
                
                st.session_state.comparison_results[model_name] = {
                    "prediction": result["text"],
                    "time_ms": elapsed,
                    "confidence": result.get("confidence"),
                    "correct": result["text"].lower() == true_text.lower()
                }
            
            st.session_state.total_solved += len(solver.get_available_models())
    
    with col2:
        if "comparison_image" in st.session_state:
            st.image(st.session_state.comparison_image, 
                    caption=f"CAPTCHA (Texte: {st.session_state.comparison_text})")
            
            st.markdown("#### Resultats")
            
            # Tableau des resultats
            results_data = []
            for model, data in st.session_state.comparison_results.items():
                results_data.append({
                    "Modele": model.upper(),
                    "Prediction": data["prediction"],
                    "Correct": "Oui" if data["correct"] else "Non",
                    "Temps (ms)": f"{data['time_ms']:.0f}",
                    "Confiance": f"{data['confidence']*100:.1f}%" if data['confidence'] else "N/A"
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Graphique
            fig = go.Figure(data=[
                go.Bar(
                    x=[d["Modele"] for d in results_data],
                    y=[float(d["Temps (ms)"]) for d in results_data],
                    marker_color=["#2ecc71" if d["Correct"] == "Oui" else "#e74c3c" for d in results_data]
                )
            ])
            fig.update_layout(
                title="Temps de resolution par modele",
                xaxis_title="Modele",
                yaxis_title="Temps (ms)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cliquez sur 'Generer et Comparer' pour voir les resultats")


# =============================================================================
# Tab 4: Benchmark
# =============================================================================

with tab4:
    st.markdown("### Benchmark complet")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_samples = st.slider("Nombre de CAPTCHAs", 5, 50, 20, key="bench_samples")
        bench_noise = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, key="bench_noise")
        
        if st.button("Lancer le benchmark", type="primary", use_container_width=True):
            results = {
                model: {"correct": 0, "times": [], "predictions": []}
                for model in solver.get_available_models()
            }
            
            progress = st.progress(0)
            status = st.empty()
            
            for i in range(n_samples):
                status.text(f"Test {i+1}/{n_samples}...")
                
                # Generer
                image, true_text = generator.generate(noise_level=bench_noise)
                
                # Tester chaque modele
                for model_name in results.keys():
                    try:
                        start = time.time()
                        result = solve_image(image, model=model_name)
                        elapsed = time.time() - start
                        
                        predicted = result["text"]
                        is_correct = predicted.lower() == true_text.lower()
                        
                        results[model_name]["times"].append(elapsed * 1000)
                        results[model_name]["predictions"].append({
                            "true": true_text,
                            "predicted": predicted,
                            "correct": is_correct
                        })
                        
                        if is_correct:
                            results[model_name]["correct"] += 1
                    except Exception:
                        pass
                
                progress.progress((i + 1) / n_samples)
            
            status.text("Benchmark termine !")
            st.session_state.benchmark_results = results
            st.session_state.benchmark_n = n_samples
    
    with col2:
        if "benchmark_results" in st.session_state:
            results = st.session_state.benchmark_results
            n = st.session_state.benchmark_n
            
            st.markdown("#### Resultats du benchmark")
            
            # Resume
            summary_data = []
            for model, data in results.items():
                accuracy = data["correct"] / n * 100
                avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
                summary_data.append({
                    "Modele": model.upper(),
                    "Precision": f"{accuracy:.1f}%",
                    "Temps moyen (ms)": f"{avg_time:.0f}",
                    "Corrects": f"{data['correct']}/{n}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Graphiques
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig_acc = go.Figure(data=[
                    go.Bar(
                        x=[d["Modele"] for d in summary_data],
                        y=[float(d["Precision"].replace("%", "")) for d in summary_data],
                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(summary_data)]
                    )
                ])
                fig_acc.update_layout(
                    title="Precision par modele",
                    yaxis_title="Precision (%)",
                    showlegend=False
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col_b:
                fig_time = go.Figure(data=[
                    go.Bar(
                        x=[d["Modele"] for d in summary_data],
                        y=[float(d["Temps moyen (ms)"]) for d in summary_data],
                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(summary_data)]
                    )
                ])
                fig_time.update_layout(
                    title="Temps moyen par modele",
                    yaxis_title="Temps (ms)",
                    showlegend=False
                )
                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Cliquez sur 'Lancer le benchmark' pour voir les resultats")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #888;">
    CAPTCHA Factory | M2 MoSEF | Universite Paris 1 Pantheon-Sorbonne | 2025
</p>
""", unsafe_allow_html=True)
