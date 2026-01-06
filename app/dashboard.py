"""
CAPTCHA Factory Dashboard - Interface Streamlit.

Dashboard interactif pour:
- Générer des CAPTCHAs
- Tester les modèles
- Comparer les performances
- Visualiser les benchmarks

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import base64
import time
from typing import Dict, Any

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="CAPTCHA Factory",
    page_icon="🔓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CHARGEMENT DES SERVICES
# =============================================================================

@st.cache_resource
def load_services():
    """Charge les services (avec cache)."""
    from app.services.captcha_generator import CaptchaGenerator
    from app.services.solver_service import SolverService
    
    generator = CaptchaGenerator()
    solver = SolverService(preload_models=["trocr"])
    
    return generator, solver


def get_services():
    """Récupère les services, les charge si nécessaire."""
    try:
        return load_services()
    except Exception as e:
        st.error(f"Erreur de chargement des services: {e}")
        st.stop()


# =============================================================================
# STYLES CSS
# =============================================================================

def load_css():
    """Charge les styles CSS personnalisés."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .success-text { color: #28a745; }
    .error-text { color: #dc3545; }
    .warning-text { color: #ffc107; }
    .captcha-image {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Affiche la sidebar avec les informations."""
    with st.sidebar:
        st.title("🔓 CAPTCHA Factory")
        st.markdown("---")
        
        # Infos projet
        st.markdown("### 📚 Projet M2 MoSEF")
        st.markdown("*Université Paris 1 Panthéon-Sorbonne*")
        st.markdown("**Équipe**: Hana & Aymen")
        
        st.markdown("---")
        
        # Modèles disponibles
        st.markdown("### 🤖 Modèles")
        
        _, solver = get_services()
        
        for model in solver.available_models:
            is_loaded = model in solver.loaded_models
            status = "✅" if is_loaded else "⬜"
            st.markdown(f"{status} **{model.upper()}**")
        
        st.markdown("---")
        
        # Statistiques de session
        if "stats" not in st.session_state:
            st.session_state.stats = {
                "generated": 0,
                "solved": 0,
                "correct": 0,
            }
        
        st.markdown("### 📊 Session")
        col1, col2 = st.columns(2)
        col1.metric("Générés", st.session_state.stats["generated"])
        col2.metric("Résolus", st.session_state.stats["solved"])
        
        if st.session_state.stats["solved"] > 0:
            accuracy = st.session_state.stats["correct"] / st.session_state.stats["solved"] * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        if st.button("🔄 Reset Stats"):
            st.session_state.stats = {"generated": 0, "solved": 0, "correct": 0}
            st.rerun()
        
        st.markdown("---")
        st.markdown("*v2.0.0*")


# =============================================================================
# PAGE: GÉNÉRATION & RÉSOLUTION
# =============================================================================

def page_generate_solve():
    """Page de génération et résolution de CAPTCHAs."""
    st.header("🎨 Génération & Résolution")
    
    generator, solver = get_services()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Paramètres")
        
        length = st.slider("Longueur", 3, 8, 5)
        width = st.slider("Largeur (px)", 150, 350, 200)
        height = st.slider("Hauteur (px)", 50, 120, 60)
        noise_level = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, 0.1)
        
        charset_option = st.selectbox(
            "Charset",
            ["CRNN (19 chars)", "Chiffres (0-9)", "Alphanumeric"],
        )
        
        if charset_option == "CRNN (19 chars)":
            charset = "23456789bcdefgmnpwxy"
        elif charset_option == "Chiffres (0-9)":
            charset = "0123456789"
        else:
            charset = "0123456789abcdefghijklmnopqrstuvwxyz"
        
        model = st.selectbox(
            "Modèle de résolution",
            ["cascade", "trocr", "crnn", "florence", "easyocr"],
        )
        
        generate_btn = st.button("🎲 Générer & Résoudre", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📸 Résultat")
        
        if generate_btn or "current_captcha" in st.session_state:
            if generate_btn:
                with st.spinner("Génération..."):
                    image, true_text = generator.generate(
                        length=length, width=width, height=height,
                        noise_level=noise_level, charset=charset,
                    )
                
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                
                with st.spinner(f"Résolution avec {model}..."):
                    start_time = time.time()
                    result = solver.solve(image_bytes, model=model)
                    solve_time = time.time() - start_time
                
                st.session_state.current_captcha = {
                    "image": image, "true_text": true_text,
                    "prediction": result.text, "confidence": result.confidence,
                    "model": result.model, "time": solve_time * 1000,
                }
                
                st.session_state.stats["generated"] += 1
                st.session_state.stats["solved"] += 1
                if result.text.lower() == true_text.lower():
                    st.session_state.stats["correct"] += 1
            
            data = st.session_state.current_captcha
            st.image(data["image"], caption="CAPTCHA généré", use_container_width=False)
            
            is_correct = data["prediction"].lower() == data["true_text"].lower()
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Texte réel", data["true_text"])
            with col_b:
                st.metric("Prédiction", data["prediction"],
                         delta="✓ Correct" if is_correct else "✗ Incorrect",
                         delta_color="normal" if is_correct else "inverse")
            with col_c:
                conf = data["confidence"]
                st.metric("Confiance", f"{conf:.1%}" if conf else "N/A")
            
            st.info(f"⏱️ Temps: {data['time']:.0f}ms | 🤖 Modèle: {data['model']}")


# =============================================================================
# PAGE: UPLOAD & RÉSOLUTION
# =============================================================================

def page_upload():
    """Page d'upload et résolution d'images."""
    st.header("📤 Upload & Résolution")
    
    _, solver = get_services()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Uploader une image")
        uploaded_file = st.file_uploader("Choisir une image CAPTCHA", type=["png", "jpg", "jpeg", "webp"])
        model = st.selectbox("Modèle", ["cascade", "trocr", "crnn", "florence", "easyocr"], key="upload_model")
        expected_text = st.text_input("Texte attendu (optionnel)", placeholder="Pour vérifier l'accuracy")
    
    with col2:
        st.subheader("📊 Résultat")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_container_width=False)
            
            if st.button("🔍 Résoudre", type="primary"):
                image_bytes = uploaded_file.getvalue()
                with st.spinner("Résolution en cours..."):
                    result = solver.solve(image_bytes, model=model)
                
                st.session_state.stats["solved"] += 1
                st.success(f"**Prédiction**: {result.text}")
                if result.confidence:
                    st.metric("Confiance", f"{result.confidence:.1%}")
                st.info(f"⏱️ {result.processing_time_ms:.0f}ms | 🤖 {result.model}")
                
                if expected_text:
                    is_correct = result.text.lower() == expected_text.lower()
                    if is_correct:
                        st.success("✅ Correct !")
                        st.session_state.stats["correct"] += 1
                    else:
                        st.error(f"❌ Incorrect. Attendu: {expected_text}")


# =============================================================================
# PAGE: COMPARAISON
# =============================================================================

def page_compare():
    """Page de comparaison des modèles."""
    st.header("⚖️ Comparaison des Modèles")
    
    generator, solver = get_services()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Paramètres")
        length = st.slider("Longueur", 3, 8, 5, key="comp_length")
        noise_level = st.slider("Bruit", 0.0, 1.0, 0.3, key="comp_noise")
        compare_btn = st.button("🔄 Comparer", type="primary", use_container_width=True)
    
    with col2:
        if compare_btn:
            with st.spinner("Génération..."):
                image, true_text = generator.generate(length=length, noise_level=noise_level)
            
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            st.image(image, caption=f"CAPTCHA: {true_text}")
            
            with st.spinner("Comparaison des modèles..."):
                compare_result = solver.compare(image_bytes, true_text=true_text)
            
            results_data = []
            for name, res in compare_result.results.items():
                is_correct = res.text.lower() == true_text.lower()
                results_data.append({
                    "Modèle": name.upper(), "Prédiction": res.text,
                    "Correct": "✅" if is_correct else "❌",
                    "Confiance": f"{res.confidence:.1%}" if res.confidence else "N/A",
                    "Temps (ms)": f"{res.processing_time_ms:.0f}",
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if compare_result.winner:
                st.success(f"🏆 Meilleur modèle: **{compare_result.winner.upper()}**")


# =============================================================================
# PAGE: BENCHMARK
# =============================================================================

def page_benchmark():
    """Page de benchmark automatisé."""
    st.header("📈 Benchmark Automatisé")
    
    generator, solver = get_services()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        n_samples = st.slider("Nombre d'échantillons", 5, 50, 10)
        noise_level = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, key="bench_noise")
        models_to_test = st.multiselect("Modèles à tester", solver.available_models, default=["trocr"])
        run_btn = st.button("🚀 Lancer le Benchmark", type="primary", use_container_width=True)
    
    with col2:
        if run_btn and models_to_test:
            progress_bar = st.progress(0)
            with st.spinner("Benchmark en cours..."):
                results = solver.benchmark(n_samples=n_samples, noise_level=noise_level, models=models_to_test)
            progress_bar.progress(100)
            st.success("Benchmark terminé !")
            
            summary_data = []
            for name, res in results.items():
                summary_data.append({
                    "Modèle": name.upper(), "Échantillons": res.total_samples,
                    "Corrects": res.correct, "Accuracy": f"{res.accuracy:.1f}%",
                    "Temps moyen": f"{res.avg_time_ms:.0f} ms",
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            fig_acc = px.bar(df_summary, x="Modèle", 
                           y=[float(x.replace("%", "")) for x in df_summary["Accuracy"]],
                           title="Accuracy par modèle", labels={"y": "Accuracy (%)"}, color="Modèle")
            st.plotly_chart(fig_acc, use_container_width=True)


# =============================================================================
# PAGE: INFORMATIONS
# =============================================================================

def page_info():
    """Page d'informations sur le projet."""
    st.header("ℹ️ Informations")
    
    st.markdown("""
    ## 🔓 CAPTCHA Factory
    
    API et Dashboard de résolution automatique de CAPTCHAs visuels.
    
    ### 🎓 Projet Académique
    
    - **Formation**: M2 MoSEF (Modélisation Statistique, Économique et Financière)
    - **Université**: Paris 1 Panthéon-Sorbonne
    - **Équipe**: Hana (CRNN) & Aymen (API/Intégration)
    
    ### 🤖 Modèles Disponibles
    
    | Modèle | Description | Accuracy | Vitesse |
    |--------|-------------|----------|---------|
    | **CRNN** | CNN + GRU bidirectionnel (entraîné par Hana) | 98% | ⚡ Très rapide |
    | **TrOCR** | Transformer pré-entraîné (HuggingFace) | 99% | 🔄 Moyen |
    | **Florence-2** | VLM Microsoft (zero-shot) | ~85% | 🐢 Lent |
    | **EasyOCR** | OCR généraliste | ~60% | 🔄 Moyen |
    """)


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Point d'entrée principal du dashboard."""
    load_css()
    render_sidebar()
    
    st.markdown('<h1 class="main-header">🔓 CAPTCHA Factory</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Résolution automatique de CAPTCHAs visuels</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🎨 Génération", "📤 Upload", "⚖️ Comparaison", "📈 Benchmark", "ℹ️ Infos"])
    
    with tabs[0]:
        page_generate_solve()
    with tabs[1]:
        page_upload()
    with tabs[2]:
        page_compare()
    with tabs[3]:
        page_benchmark()
    with tabs[4]:
        page_info()


if __name__ == "__main__":
    main()