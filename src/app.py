import os
import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 🎨 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="MNIST IA - Pipeline Completo & Inferência",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    /* Cabeçalhos */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Cartões de Métricas */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Cartão de Destaque da Predição */
    .prediction-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-top: 5px;
    }
    .pred-digit {
        font-size: 4.2rem;
        font-weight: 900;
        color: #38bdf8;
        line-height: 1;
        margin: 8px 0;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
    .pred-conf {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Caixas de Texto Estilizadas */
    .concept-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
    .alert-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 2. CARREGAMENTO DO MODELO CAMPEÃO (CACHE)
# ==============================================================================
@st.cache_resource
def carregar_modelo_campeao():
    """Carrega o modelo campeão MLP exportado."""
    caminhos = [
        os.path.join(os.path.dirname(__file__), "..", "models", "modelo_mlp_mnist.joblib"),
        "models/modelo_mlp_mnist.joblib",
        "../models/modelo_mlp_mnist.joblib"
    ]
    for c in caminhos:
        if os.path.exists(c):
            dados = joblib.load(c)
            if isinstance(dados, dict) and "modelo" in dados:
                return dados["modelo"]
            return dados
    return None

modelo_mlp = carregar_modelo_campeao()

# ==============================================================================
# 👁️ 3. PIPELINE DE VISÃO COMPUTACIONAL (OPENCV)
# ==============================================================================
def processar_imagem_para_mnist(imagem_input):
    """
    Pipeline OpenCV do projeto:
    1. Grayscale -> 2. Inversão automática -> 3. Binarização -> 
    4. Bounding Box -> 5. Resize 20x20 proporcional -> 6. Centralização 28x28 -> 7. Normalização [0, 1]
    """
    if isinstance(imagem_input, Image.Image):
        img_np = np.array(imagem_input)
    else:
        img_np = imagem_input

    # 1. Trata canais de cor
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4: # RGBA
            rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # 2. Inversão automática se o fundo for branco/claro
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 3. Binarização suave
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 4. Corte Rente ao Traço (Bounding Box)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos or len(contornos) == 0:
        img_vazia = np.zeros((28, 28), dtype=np.float32)
        return img_vazia, img_vazia.reshape(1, 784)

    maior_contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_contorno)
    
    if w < 2 or h < 2:
        img_vazia = np.zeros((28, 28), dtype=np.float32)
        return img_vazia, img_vazia.reshape(1, 784)

    recorte = thresh[y:y+h, x:x+w]

    # 5. Proporção Centralizada (máximo 20x20 pixels)
    if h > w:
        fator = 20.0 / h
        novo_h = 20
        novo_w = max(1, int(w * fator))
    else:
        fator = 20.0 / w
        novo_w = 20
        novo_h = max(1, int(h * fator))

    recorte_redimensionado = cv2.resize(recorte, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    # 6. Centralização exata na matriz 28x28 preta
    tela_mnist = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - novo_h) // 2
    offset_x = (28 - novo_w) // 2
    tela_mnist[offset_y:offset_y+novo_h, offset_x:offset_x+novo_w] = recorte_redimensionado

    # 7. Normalização [0.0, 1.0]
    img_normalizada = tela_mnist.astype(np.float32) / 255.0
    vetor_features = img_normalizada.reshape(1, 784)

    return tela_mnist, vetor_features

# ==============================================================================
# 🎛️ 4. BARRA LATERAL (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.title("🧠 MNIST AI Pipeline")
    st.markdown("**Mini-Projeto Avaliativo (Módulo 2)**")
    st.markdown("*Desenvolvimento de IA para Análise Preditiva*")
    st.markdown("---")
    
    st.markdown("### 📌 Sumário das Fases")
    st.markdown("1. **Fase 1:** Carga e EDA do MNIST")
    st.markdown("2. **Fase 2:** Split Estratificado e Normalização")
    st.markdown("3. **Fase 3:** Treinamento dos 4 Modelos")
    st.markdown("4. **Fase 4:** Benchmark e Matrizes de Confusão")
    st.markdown("5. **Fase 5.1/5.2:** Testes OOD e Falsa Certeza")
    st.markdown("6. **Fase 5.3:** Inferência Real com OpenCV")
    
    st.markdown("---")
    st.markdown("### 🏆 Modelo Campeão")
    st.markdown("• **Arquitetura:** `Rede Neural MLP (128->64->10)`")
    st.markdown("• **Acurácia Benchmark:** `97.39%`")
    st.markdown("• **Acurácia Mundo Real:** `80.00%`")
    st.markdown("• **Status:** :green[**Produção Ativa**]")
    st.caption("© 2026 - Pipeline de Machine Learning & Visão Computacional.")

# ==============================================================================
# 🌟 5. CABEÇALHO PRINCIPAL
# ==============================================================================
st.markdown('<div class="main-header">🤖 Desenvolvimento de IA para Reconhecimento de Dígitos (MNIST)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Apresentação interativa do ciclo completo de Machine Learning — da Análise Exploratória à Inferência em Produção.</div>', unsafe_allow_html=True)

# ==============================================================================
# 📑 6. ABAS DEDICADAS A CADA FASE DO PROJETO
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Fase 1: EDA & Carga",
    "⚙️ Fase 2: Split & Prep",
    "🧠 Fase 3: Treinamento (4 Modelos)",
    "📊 Fase 4: Avaliação & Benchmark",
    "🧪 Fase 5.1/5.2: OOD & Falsa Certeza",
    "✍️ Fase 5.3: Inferência ao Vivo (Canvas & Upload)"
])

# ------------------------------------------------------------------------------
# 📥 TAB 1: FASE 1 - EDA & CARGA DO DATASET
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("## 📥 Fase 1: Carregamento e Análise Exploratória de Imagens (EDA)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-title">Total de Imagens</div><div class="metric-value">70.000</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-title">Dimensão da Imagem</div><div class="metric-value">28 × 28 px</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-title">Features (Pixels)</div><div class="metric-value">784</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-title">Classes Alvo</div><div class="metric-value">10 (0 a 9)</div></div>""", unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🔬 Considerações Técnicas da Fase 1:
    
    * **Fonte e Estrutura dos Dados:** O conjunto foi importado via `fetch_openml('mnist_784', version=1, as_frame=False)`. Cada uma das 70.000 instâncias consiste em um vetor unidimensional de 784 valores numéricos que representam uma grade bidimensional de $28 \times 28$ pixels em tons de cinza.
    * **Distribuição Balanceada:** Todas as 10 classes (dígitos de 0 a 9) apresentam aproximadamente **7.000 amostras cada** (~10% do total), caracterizando um dataset perfeitamente equilibrado sem necessidade de balanceamento sintético.
    * **Integridade:** Não existem valores ausentes (`NaN`), e a intensidade de cada pixel varia na escala inteira de 0 (fundo preto absoluto) a 255 (intensidade máxima do traço branco).
    """)

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("#### 📊 1.1 Distribuição de Amostras por Dígito:")
        if os.path.exists("src/assets/fase1_distribuicao_classes.png"):
            st.image("src/assets/fase1_distribuicao_classes.png", use_column_width=True)

    with col_plot2:
        st.markdown("#### 🖼️ 1.2 Grade Visual de Amostras ($2 \\times 5$):")
        if os.path.exists("src/assets/fase1_grade_amostras_2x5.png"):
            st.image("src/assets/fase1_grade_amostras_2x5.png", use_column_width=True)

# ------------------------------------------------------------------------------
# ⚙️ TAB 2: FASE 2 - PRÉ-PROCESSAMENTO & SPLIT
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("## ⚙️ Fase 2: Divisão Estratificada e Normalização de Features")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Conjunto de Treinamento (80%)</div>
            <div class="metric-value">56.000 Imagens</div>
            <p style="color: #94a3b8; margin-top: 5px;">Utilizado para ajuste de pesos e parâmetros</p>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Conjunto de Teste Rigoroso (20%)</div>
            <div class="metric-value">14.000 Imagens</div>
            <p style="color: #94a3b8; margin-top: 5px;">Dados inéditos para avaliação do benchmark</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🔬 Considerações Técnicas da Fase 2:
    
    1. **Estratificação (`stratify=y`):**
       - A divisão com `train_test_split(..., stratify=y, test_size=0.2, random_state=42)` assegura que a proporção exata de cada classe seja idêntica tanto no treino quanto no teste (~10% por dígito), evitando viés de amostragem.
       
    2. **Normalização Min-Max $[0.0, 1.0]$:**
       - Realizada dividindo todos os valores de pixel por `255.0` (`X = X / 255.0`).
       - **Justificativa Matemática:** Acelera a convergência do algoritmo de descida do gradiente (**Adam**) na Rede Neural, evita a saturação das ativações ReLU e estabiliza o cálculo dos hiperplanos de margem máxima do **SVM (Kernel RBF)**.
    """)

    st.markdown("#### 📊 Comprovação de Proporção Idêntica por Classe:")
    df_prop = pd.DataFrame({
        "Dígito": [str(i) for i in range(10)],
        "Treino (%)": ["9.86%", "11.25%", "9.99%", "10.20%", "9.75%", "9.02%", "9.82%", "10.42%", "9.75%", "9.94%"],
        "Teste (%)": ["9.86%", "11.25%", "9.99%", "10.20%", "9.75%", "9.02%", "9.82%", "10.42%", "9.75%", "9.94%"],
        "Status": ["Perfeito ✅"] * 10
    })
    st.dataframe(df_prop, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# 🧠 TAB 3: FASE 3 - TREINAMENTO DOS 4 MODELOS
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("## 🧠 Fase 3: Treinando os 4 Modelos")
    
    st.markdown("""
    Nesta fase, colocamos a mão na massa para treinar **4 modelos diferentes** de Inteligência Artificial para reconhecer os números desenhados do MNIST: **Random Forest**, **LightGBM**, **SVM** e uma **Rede Neural**.
    
    ---
    #### 🧪 Testando vários valores para não escolher no escuro
    Assim como no projeto do Módulo 1, adotei formato de criar **laços de experimentação (`for`)** antes de definir qualquer modelo final. Essa abordagem é fundamental porque, assim não precisamos adivinhar; vemos na prática os números de acurácia, F1-Score e tempo de cada combinação e assegura que o modelo salvo no `modelos_campeoes` é realmente o melhor representante daquela técnica:
    """)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        <div class="concept-box">
            <h4>🌲 1. Random Forest (Ensemble Bagging)</h4>
            <p>• Testei profundidades de 5, 10 e 20 (com 100 árvores). O modelo com <b>profundidade 20</b> foi o melhor e virou o nosso campeão.</p>
            <p>• <b>Tempo:</b> ~25s | <b>Acurácia Treino:</b> 100.0% | <b>Acurácia Teste:</b> 96.69%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="concept-box">
            <h4>⚡ 2. LightGBM (Gradient Boosting)</h4>
            <p>• Testei velocidades de aprendizado de 0.05 e 0.1 (com 100 árvores). A <b>velocidade 0.1</b> foi a campeã por ser rápida e muito precisa.</p>
            <p>• <b>Tempo:</b> ~14s | <b>Acurácia Treino:</b> 100.0% | <b>Acurácia Teste:</b> 97.44%</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="concept-box">
            <h4>🔶 3. SVM (Support Vector Machine - Kernel RBF)</h4>
            <p>• Testei o valor de <i>C</i> em 1.0 e 5.0, e o modelo com <b>C=5.0</b> foi o campeão com o melhor desempenho.</p>
            <p>• <b>Tempo:</b> ~35 min (Processamento Pesado) | <b>Acurácia Treino:</b> 99.91% | <b>Acurácia Teste:</b> 98.33%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="concept-box">
            <h4>🧠 4. Rede Neural Artificial (MLP com Keras)</h4>
            <p>• Estrutura: <code>Sequential([Dense(128, relu), Dense(64, relu), Dense(10, softmax)])</code> com otimizador Adam (lr=0.001).</p>
            <p>• <b>Tempo:</b> ~36s | <b>Acurácia Treino:</b> 98.92% | <b>Acurácia Teste:</b> 97.39%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    #### ⚠️ NOTA IMPORTANTE E OBSERVAÇÕES DE ENGENHARIA:
    * **O caso do LightGBM:** Quando treinei com `learning_rate = 0.1`, ele bateu **100% de acerto nos dados de treino**. Na mesma hora fiquei na dúvida e pensei: *"Será que o modelo apenas decorou as respostas?"* (sobreajuste). Fiquei desconfiado e decidi revisar o código com calma para ter certeza de que não havia nenhum erro ou mistura de dados. O código estava certo, além de que nos dados de teste manteve uma acurácia bastante boa de 97.44%, provando que o resultado era real e impressionante.
    * **O peso do SVM e a ajuda da CPU:** O SVM foi o modelo mais demorado de todos (levou quase 35 min durante o processo). Como o meu computador tem um processador i5 de 4 núcleos, sem GPU dedicada, pesquisei e perguntei para a IA como fazer para não sobrecarregar e nem travar a máquina. Fizemos ajustes no código:
      - Usamos `n_jobs=-1` no Random Forest e no LightGBM para fazer o computador usar os **4 núcleos do processador ao mesmo tempo**.
      - No SVM, aumentamos a memória de trabalho com `cache_size=1000` (1 GB) para o computador não sofrer ao recalcular as distâncias.
    * **E por que não fizemos laço na Rede Neural?** Na Rede Neural, montamos uma estrutura com 128 neurônios na primeira camada, 64 na segunda e 10 na saída. Como as redes neurais já exigem muito esforço da CPU por natureza, decidi não colocar um laço `for` nela para não deixar o computador muito pesado e lento. Essa configuração direta já foi excelente: treinou em apenas **36 segundos** e alcançou mais de **97% de acerto**.
    """)

# ------------------------------------------------------------------------------
# 📊 TAB 4: FASE 4 - AVALIAÇÃO COMPARATIVA & BENCHMARK
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("## 📊 Fase 4: Avaliação Comparativa de Desempenho (Benchmark dos Modelos)")
    
    st.markdown("### 📋 Tabela Geral de Métricas Consolidadas (14.000 Imagens de Teste):")
    dados_benchmark = {
        "Família / Modelo": ["Random Forest", "LightGBM", "SVM (Kernel RBF)", "Rede Neural (MLP)"],
        "Acurácia Treino": ["100.00%", "100.00%", "99.91%", "98.92%"],
        "Acurácia Teste": ["96.69%", "97.44%", "98.33%", "97.39%"],
        "Precisão (Weighted)": ["96.69%", "97.43%", "98.33%", "97.39%"],
        "Recall (Weighted)": ["96.69%", "97.44%", "98.33%", "97.39%"],
        "F1-Score (Weighted)": ["96.69%", "97.43%", "98.33%", "97.39%"],
        "Tempo de Treinamento": ["~25 segundos", "~14 segundos", "~35 minutos", "~36 segundos"]
    }
    df_metrics = pd.DataFrame(dados_benchmark)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 4.1 Matrizes de Confusão Comparativas ($2 \\times 2$ Grid):")
    if os.path.exists("src/assets/fase4_matrizes_confusao_2x2.png"):
        st.image("src/assets/fase4_matrizes_confusao_2x2.png", use_column_width=True)

    st.markdown("""
    ---
    ### 📌 Conclusão Técnica da Fase 4: Análise Comparativa dos Resultados
    
    Com a consolidação das Matrizes de Confusão, da Tabela Geral e dos Relatórios de Classificação (`classification_report`), podemos extrair conclusões claras sobre os modelos e o comportamento dos dados:
    
    #### 1. Dígitos com Maior Taxa de Confusão
    Analisando os relatórios individuais e os mapas de calor, observamos padrões claros:
    * **O Dígito mais difícil (Número 9):** Foi o dígito com menor pontuação em todos os modelos.
      * **4 vs 9 (A maior confusão do projeto):** Foi o erro mais frequente em todos os modelos. Pode ser a causa de que muitas pessoas escrevem o número 4 fechado na parte superior, tornando-o quase idêntico a um 9.
      * **3 vs 5:** Ambos os dígitos compartilham a mesma curva arredondada na parte inferior.
      * **7 vs 1 e 7 vs 9:** Traços verticais ou inclinados do 7 geraram pequenas confusões pontuais com o 1 e com o 9.
    * **Dígitos mais fáceis:** Os números **0** e **1** foram os mais fáceis para todos os algoritmos (F1-Score acima de 98.5%), devido aos seus formatos geométricos bem distintos (círculo fechado e linha vertical simples).
    
    #### 2. Modelo Campeão em Desempenho
    O modelo com a **melhor performance geral foi o SVM (Kernel RBF)**. Manteve uma consistência impressionante em todas as classes, sendo o modelo que menos errou nas distinções difíceis (como 4 vs 9 e 3 vs 5), mas com um custo computacional muito maior que os outros modelos, então acho importante tomar isso em conta, já que, por exemplo a **Rede Neural MLP** teve uma performance muito boa e com um custo computacional bem baixo.
    
    O **LightGBM (97,44%)** e a **Rede Neural MLP (97,39%)** ficaram praticamente iguais, com desempenhos de alto nível; porém, o modelo LightGBM demorou mais no treino com uso de 100% da CPU, enquanto a **Rede Neural MLP** foi bem rápida e balanceada. E por último o **Random Forest (96,69%)** teve o menor desempenho, no entanto com resposta aceitável.
    
    #### 3. Custo Computacional vs. Desempenho (Tempo de Treino vs. Acurácia)
    A comparação entre tempo e precisão nos traz a lição mais importante para aplicações no mundo real:
    * **O trade-off do SVM:** Embora tenha vencido por porcentagem nas métricas, o SVM foi o mais pesado de todos (quase **35 minutos** para terminar todo o processo de treino na CPU).
    * **A eficiência da Rede Neural:** A Rede Neural (MLP) alcançou 97,39% de acurácia treinando em apenas **36 segundos** (quase 7 vezes mais rápida que o SVM).
    * **A robustez do LightGBM:** Entregou 97,44% em tempo competitivo, mostrando excelente equilíbrio, mas com custo computacional alto.
    
    > **💡 Conclusão Prática de Engenharia:** Em um cenário de produção real (por exemplo, um aplicativo que precisa processar milhões de documentos ou rodar em dispositivos móveis), a **Rede Neural** ou o **LightGBM** seriam as escolhas mais inteligentes, pois entregam praticamente a mesma precisão do SVM com um consumo de tempo e processamento drasticamente menor.
    """)

# ------------------------------------------------------------------------------
# 🧪 TAB 5: FASE 5.1/5.2 - TESTES OOD & FALSA CERTEZA
# ------------------------------------------------------------------------------
with tab5:
    st.markdown("## 🧪 Fase 5.1 & 5.2: Desafios A & B — Testes OOD e Falsa Certeza (*Overconfidence*)")
    
    st.markdown("""
    ### 🎭 5.1 - Desafio (A): Treinamento Restrito com Classes Ocultadas (Class Masking)
    Para este experimento, criamos um modelo completamente novo que **nunca viu os dígitos 4 e 7** durante toda a sua fase de aprendizado.
    
    ### 🕵️‍♂️ 5.2 - Desafio (B): Teste Fora da Distribuição (OOD) e Falsa Certeza (*Overconfidence*)
    Nesta etapa, apresentamos **exclusivamente amostras reais dos dígitos 4 e 7** para os dois modelos que foram treinados sem nunca tê-los visto:
    * Como os modelos não possuem uma classe "Não Sei", eles são **forçados a classificar** essas imagens em uma das 8 classes conhecidas ($0, 1, 2, 3, 5, 6, 8, 9$).
    * Analisamos a distribuição dessas predições forçadas e o nível de confiança (probabilidade) atribuído a essas respostas erradas, revelando a perigosa tendência dos modelos de Machine Learning apresentarem **alta certeza em predições incorretas**.
    """)

    st.markdown("---")
    st.markdown("### 📊 Gráficos do Teste OOD (Fases 5.1 e 5.2):")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Matrizes de Confusão OOD (Dígitos 4 e 7):")
        if os.path.exists("src/assets/fase5_ood_matrizes_confusao.png"):
            st.image("src/assets/fase5_ood_matrizes_confusao.png", use_column_width=True)
    with col_g2:
        st.markdown("#### Distribuição de Probabilidades e Falsa Certeza:")
        if os.path.exists("src/assets/fase5_ood_distribuicao_probabilidades.png"):
            st.image("src/assets/fase5_ood_distribuicao_probabilidades.png", use_column_width=True)

    st.markdown("""
    ---
    ### 📌 Conclusão Técnica e Reflexão: Teste OOD e o Fenômeno da Falsa Certeza (*Overconfidence*)
    
    O experimento realizado nas Fases 5.1 e 5.2 nos permitiu ver o comportamento dos modelos quando confrontados com dados completamente **fora de sua distribuição de treino (Out-of-Distribution - OOD)**.
    
    #### 1. Como os modelos reagiram aos dígitos ocultados (4 e 7)?
    Observei como modelos de classificação tradicionais são fechados a responder *"não sei"*, e foram forçados a encaixar os números desconhecidos nas 8 classes que conhecem ($0, 1, 2, 3, 5, 6, 8, 9$):
    * **O Dígito 4:** Foi classificado em sua maioria no **dígito 9** pelos dois modelos.
    * **O Dígito 7:** Também foi classificado em sua maioria como **dígito 9** (e pontualmente como 1, 2 ou 3).
    
    #### 2. Comparação de Comportamento: Random Forest vs. LightGBM
    Os gráficos de probabilidade mostraram uma diferença entre os dois algoritmos ao lidar com a incerteza:
    * **Random Forest:** Foi mais cauteloso. Na dúvida, espalhou probabilidades menores entre classes parecidas (como 2, 3 e 9 na Amostra #0), gerando certezas mais moderadas (45% a 65%).
    * **LightGBM:** Apresentou uma **polarização agressiva e perigosa**. O modelo tomou decisões com certezas extremas e infundadas, atingindo **96,1% de confiança** em um 4 real (Amostra #10) e impressionantes **100,0% de certeza absoluta** de que um 7 real era um 9 (Amostra #15)!
    """)

    st.markdown("""
    <div class="alert-box">
        <h4>🤔 Reflexão Pessoal e Crítica sobre IA em Sistemas Reais:</h4>
        <p><i>"O que me deixa pensando: se for um veículo autônomo treinado para reconhecer pedestres, carros e entre outras coisas, mas aparece algo que nunca ele viu, ele não para? ele continua? o que acontece? ele vai dizer 'não sei'? Pelo observado não! Ao igual, como sempre é falado nas aulas, e em casos de medicina? Se achar uma doença rara ou desconhecida? Pode classificar como um simples resfriado ainda com certeza muito alta?</i></p>
        <p><i>Será que para aplicações do mundo real é essencial este tipo de filtros de detecção de OOD e limiares de confiança para fazer o sistema admitir a sua própria ignorância e solicitar intervenção ou supervisão humana? É isso possível? Achei super esclarecedor ver o trabalho por baixo do capô do funcionamento."</i></p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# ✍️ TAB 6: FASE 5.3 - INFERÊNCIA AO VIVO & COROAÇÃO DA REDE NEURAL
# ------------------------------------------------------------------------------
with tab6:
    st.markdown("## ✍️ Fase 5.3: Processamento de Imagens e Inferência com Imagens Manuscritas Próprias")
    
    st.markdown("""
    Nesta etapa final, testamos o modelo campeão com **imagens desenhadas por nós mesmos fora do laboratório**:
    * Criamos um pipeline completo de **Processamento Digital de Imagens (PDI) com OpenCV (`cv2`)** para transformar imagens de qualquer resolução no formato padrão estrito do MNIST ($28 \times 28$ pixels, escala de cinza invertida e normalizada entre $[0.0, 1.0]$).
    * Realizamos a inferência com o modelo campeão e plotamos a imagem original, a imagem processada e a distribuição de probabilidades preditivas.
    """)

    st.markdown("---")
    st.markdown("### 🎨 Laboratório Interativo de Testes (Ao Vivo):")

    modo_interativo = st.radio(
        "Selecione a Forma de Entrada:",
        ["✏️ Desenhar no Canvas Interativo", "🖼️ Testar com Amostras do Projeto (data/custom_digits)", "📁 Upload de Imagem Externa"],
        horizontal=True
    )

    if modo_interativo == "✏️ Desenhar no Canvas Interativo":
        col_c1, col_c2 = st.columns([1.1, 1.3])
        with col_c1:
            st.markdown("**Desenhe o dígito no quadro abaixo:**")
            grosor = st.slider("Espessura do Traço:", min_value=12, max_value=32, value=22, step=2, key="slider_grosor_direct")
            
            # Canvas visível diretamente no nível da aba
            canvas_res = st_canvas(
                fill_color="#000000",
                stroke_width=grosor,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                update_streamlit=True,
                key="canvas_live_main"
            )
            
            btn_classificar = st.button("🚀 Reconhecer Desenho", type="primary", use_container_width=True)

        with col_c2:
            tem_desenho = (canvas_res is not None and 
                           canvas_res.image_data is not None and 
                           np.max(canvas_res.image_data[:, :, :3]) > 20)
            
            if tem_desenho or btn_classificar:
                if canvas_res is not None and canvas_res.image_data is not None and np.max(canvas_res.image_data[:, :, :3]) > 20:
                    tela_proc, vetor_784 = processar_imagem_para_mnist(canvas_res.image_data)
                    
                    if modelo_mlp is not None:
                        probs = modelo_mlp.predict(vetor_784, verbose=0)[0]
                        pred_digito = int(np.argmax(probs))
                        conf = probs[pred_digito] * 100

                        col_res1, col_res2 = st.columns([1, 1.2])
                        with col_res1:
                            st.markdown("**Matriz 28x28 (OpenCV):**")
                            fig_pdi, ax_pdi = plt.subplots(figsize=(2.2, 2.2))
                            ax_pdi.imshow(tela_proc, cmap='gray')
                            ax_pdi.axis('off')
                            st.pyplot(fig_pdi)
                            plt.close(fig_pdi)
                        
                        with col_res2:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="metric-title">Dígito Previsto</div>
                                <div class="pred-digit">{pred_digito}</div>
                                <div class="pred-conf">Confiança: {conf:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Gráfico de Barras Softmax
                        fig_b, ax_b = plt.subplots(figsize=(6, 2.3))
                        fig_b.patch.set_facecolor('#0e1117')
                        ax_b.set_facecolor('#0e1117')
                        cores = ['#38bdf8' if i == pred_digito else '#334155' for i in range(10)]
                        ax_b.bar(range(10), probs * 100, color=cores, edgecolor='#1e293b')
                        ax_b.set_ylim(0, 105)
                        ax_b.set_ylabel('Certeza (%)', color='#94a3b8', fontsize=8)
                        ax_b.set_xticks(range(10))
                        ax_b.tick_params(colors='#94a3b8')
                        ax_b.spines['top'].set_visible(False)
                        ax_b.spines['right'].set_visible(False)
                        ax_b.spines['left'].set_color('#334155')
                        ax_b.spines['bottom'].set_color('#334155')
                        st.pyplot(fig_b)
                        plt.close(fig_b)
                else:
                    st.warning("⚠️ Desenhe um número no quadro antes de classificar.")
            else:
                st.info("✏️ Desenhe um número no quadro à esquerda com o mouse ou tela touch.")

    elif modo_interativo == "🖼️ Testar com Amostras do Projeto (data/custom_digits)":
        st.markdown("**Selecione uma das 5 imagens criadas no projeto:**")
        amostras_disponiveis = [
            "data/custom_digits/meu_digito_2.png",
            "data/custom_digits/meu_digito_3.png",
            "data/custom_digits/meu_digito_6.png",
            "data/custom_digits/meu_digito_7.png",
            "data/custom_digits/meu_digito_9.png"
        ]
        escolha = st.selectbox("Arquivo PNG:", amostras_disponiveis, format_func=lambda x: os.path.basename(x))
        
        if os.path.exists(escolha):
            img_escolhida = cv2.imread(escolha, cv2.IMREAD_GRAYSCALE)
            tela_proc_ex, vetor_784_ex = processar_imagem_para_mnist(img_escolhida)
            
            if modelo_mlp is not None:
                probs_ex = modelo_mlp.predict(vetor_784_ex, verbose=0)[0]
                pred_ex = int(np.argmax(probs_ex))
                conf_ex = probs_ex[pred_ex] * 100

                col_e1, col_e2, col_e3 = st.columns([1, 1, 1.5])
                with col_e1:
                    st.markdown("**Original:**")
                    st.image(img_escolhida, width=160)
                with col_e2:
                    st.markdown("**Processada (28x28):**")
                    st.image(tela_proc_ex, width=160)
                with col_e3:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="metric-title">Predição da Rede Neural</div>
                        <div class="pred-digit">{pred_ex}</div>
                        <div class="pred-conf">Confiança: {conf_ex:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        up_file = st.file_uploader("Envie uma foto de dígito (PNG/JPG):", type=["png", "jpg", "jpeg"])
        if up_file is not None:
            img_up = Image.open(up_file)
            tela_proc_up, vetor_784_up = processar_imagem_para_mnist(img_up)
            if modelo_mlp is not None:
                probs_up = modelo_mlp.predict(vetor_784_up, verbose=0)[0]
                pred_up = int(np.argmax(probs_up))
                conf_up = probs_up[pred_up] * 100

                col_u1, col_u2, col_u3 = st.columns([1, 1, 1.5])
                with col_u1:
                    st.markdown("**Original:**")
                    st.image(img_up, width=160)
                with col_u2:
                    st.markdown("**Processada 28x28:**")
                    st.image(tela_proc_up, width=160)
                with col_u3:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="metric-title">Dígito Predito</div>
                        <div class="pred-digit">{pred_up}</div>
                        <div class="pred-conf">Confiança: {conf_up:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🖼️ Painel Consolidado das 5 Amostras Reais (Notebook):")
    if os.path.exists("src/assets/fase5_inferencia_5_digitos_reais.png"):
        st.image("src/assets/fase5_inferencia_5_digitos_reais.png", use_column_width=True)

    st.markdown("""
    ---
    ### 5.3 Conclusão Técnica: Validação Cruzada no Mundo Real
    
    #### 🔍 Contexto e Motivação da Validação Multimodelo
    Durante o teste inicial de inferência em imagens manuscritas externas com o pipeline OpenCV, a **Rede Neural (MLP)** acertou 4 dos 5 dígitos, mas classificou o dígito `7` como `2`. 
    
    Na dúvida da boa performance da **Rede Neural**, submetemos exatamente as mesmas 5 amostras a dois dos outros modelos de alto desempenho do projeto: **LightGBM** e **SVM (Kernel RBF)**.
    """)
    
    dados_reais = {
        "Amostra Real": ["meu_digito_2.png", "meu_digito_3.png", "meu_digito_6.png", "meu_digito_7.png", "meu_digito_9.png"],
        "Rótulo Real": ["2", "3", "6", "7", "9"],
        "🧠 Rede Neural (MLP)": ["Dígito 2 (100.0%) ✅", "Dígito 3 (79.41%) ✅", "Dígito 6 (99.33%) ✅", "Dígito 2 (66.03%) ❌", "Dígito 9 (99.95%) ✅"],
        "⚡ LightGBM": ["Dígito 2 (99.69%) ✅", "Dígito 3 (99.33%) ✅", "Dígito 6 (41.84%) ✅", "Dígito 1 (40.42%) ❌", "Dígito 4 (36.73%) ❌"],
        "🔶 SVM (Kernel RBF)": ["Dígito 2 (70.41%) ✅", "Dígito 3 (77.38%) ✅", "Dígito 5 (63.95%) ❌", "Dígito 1 (63.57%) ❌", "Dígito 9 (64.75%) ✅"]
    }
    df_reais = pd.DataFrame(dados_reais)
    st.dataframe(df_reais, use_container_width=True, hide_index=True)
    
    st.markdown("""
    #### 📊 Análise dos Resultados:
    * **A MLP demonstrou a melhor capacidade de abstração espacial não-linear:** Nos dígitos que acertou (`2`, `3`, `6` e `9`), apresentou níveis de confiança altos, enquanto os outros modelos oscilaram ou erraram completamente (como o SVM errando o `6` e o LightGBM errando o `9`).
    * **O erro do Dígito `7`:** O dígito `7` falhou em **todos os três modelos**, comprovando que não se tratava de um defeito da Rede Neural, mas de um desafio de *Domain Shift* comum em dados reais.
    
    ---
    #### 👑 Veredito Final para Deploy da Aplicação
    A **Rede Neural (MLP)** é formalmente declarada como **Melhor Modelo do Projeto**, superando os demais modelos:
    * **Acurácia Prática em Produção:** 80% vs 60% dos concorrentes.
    * **Eficiência de Treinamento e Inferência:** Treinamento em ~36 segundos com inferência em milissegundos.
    * **Resiliência a Ruído e Variações de Traço:** Maior estabilidade na distribuição de probabilidades *Softmax*.
    """)
