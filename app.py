import streamlit as st
import torch
from PIL import Image
import os
import base64
import google.generativeai as genai
import google.api_core.exceptions
import time
import traceback

# --- Importar do modelo.py (Verifique se MAX_LENGTH=128 aqui também) ---
try:
    from modelo import (
        GFD_BERT_ViT, MACDM, MDCRAPN, tokenizer, image_transforms,
        D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, MAX_LENGTH, DEVICE
    )
    # Garante MAX_LENGTH correto
    if 'MAX_LENGTH' in globals() and MAX_LENGTH != 128:
         st.warning(f"AVISO: MAX_LENGTH em app.py ({MAX_LENGTH}) != 128 (treino). Ajustando.")
         MAX_LENGTH = 128
    elif 'MAX_LENGTH' not in globals():
         st.warning("MAX_LENGTH não definido globalmente. Definindo como 128.")
         MAX_LENGTH = 128

    MODELO_PY_LOADED = True
except ImportError:
    st.error("Erro: 'modelo.py' não encontrado.")
    MODELO_PY_LOADED = False
    st.stop()
except Exception as e:
    st.error(f"Erro ao importar 'modelo.py': {e}")
    MODELO_PY_LOADED = False
    st.stop()

# --- 1. Carregar SEU Modelo Local (MDCRAPN) ---
@st.cache_resource
def load_local_models():
    if not MODELO_PY_LOADED: return None, None, None
    try:
        gfd_path = 'gfd_model_weights.pth'
        macdm_path = 'macdm_model_weights.pth'
        mdcrapn_path = 'mdcrapn_model_weights.pth'

        if not all(os.path.exists(p) for p in [gfd_path, macdm_path, mdcrapn_path]):
             st.warning("Pesos do modelo local (.pth extraídos) não encontrados. Execute extract_weights.py. Saída MDCRAPN indisponível.")
             return None, None, None

        # Certifique-se que as constantes globais estão disponíveis aqui
        gfd_model = GFD_BERT_ViT(VOCAB_SIZE, D_MODEL, MAX_LENGTH).to(DEVICE)
        macdm_model = MACDM(D_MODEL).to(DEVICE)
        mdcrapn_model = MDCRAPN(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE).to(DEVICE)

        gfd_model.load_state_dict(torch.load(gfd_path, map_location=DEVICE))
        macdm_model.load_state_dict(torch.load(macdm_path, map_location=DEVICE))
        mdcrapn_model.load_state_dict(torch.load(mdcrapn_path, map_location=DEVICE))

        gfd_model.eval(); macdm_model.eval(); mdcrapn_model.eval()
        print("Modelos locais (MDCRAPN) carregados com sucesso do checkpoint!")
        return gfd_model, macdm_model, mdcrapn_model

    except Exception as e:
        st.error(f"Erro ao carregar modelos locais dos pesos extraídos: {e}")
        return None, None, None

gfd_model_inf, macdm_model_inf, mdcrapn_model_inf = load_local_models()
LOCAL_MODEL_LOADED = all(m is not None for m in [gfd_model_inf, macdm_model_inf, mdcrapn_model_inf])


# --- 2. FUNÇÃO DE GERAÇÃO (MDCRAPN - Usando seu modelo treinado) ---
# Usa a lógica de inferência do seu notebook (Argmax/Greedy)
def run_generation(input_text, pil_image, max_summary_length=MAX_LENGTH): # Nome original
    if not LOCAL_MODEL_LOADED:
        return "Erro: Modelos MDCRAPN locais não carregados."

    # Pré-processamento
    text_inputs = tokenizer(input_text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
    input_ids = text_inputs['input_ids']
    attention_mask = text_inputs['attention_mask']
    try:
        pixel_values = image_transforms(pil_image).unsqueeze(0).to(DEVICE)
    except Exception as e:
         return f"Erro ao processar imagem para MDCRAPN: {e}"

    # Inferência
    with torch.no_grad():
        try:
            # Fluxo Forward COMPLETO - Verificado e Corrigido
            disorder_output, GFD_features = gfd_model_inf(input_ids, attention_mask, pixel_values)

            # --- Cálculo correto de T_features_mean ---
            # 1. Obter saída do BERT
            bert_output = gfd_model_inf.bert_encoder.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 2. Aplicar a camada linear de projeção
            T_features_projected = gfd_model_inf.bert_encoder.linear(bert_output.last_hidden_state) # [B, Seq, D_MODEL]
            # 3. Calcular a média
            T_features_mean = T_features_projected.mean(dim=1) # [B, D_MODEL]
            # --- Fim da Correção ---

            GFD_features_mean = GFD_features.mean(dim=1)
            contextual_features = macdm_model_inf(T_features_mean, GFD_features_mean)
            attr_tokens = contextual_features.unsqueeze(1) # [B=1, 1, D]

            # Geração Gulosa (Argmax)
            decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(DEVICE)
            generated_token_ids = []
            for _ in range(max_summary_length):
                logits, _, _ = mdcrapn_model_inf(attr_tokens, decoder_input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).reshape(1, 1) # Correção shape argmax

                if next_token_id.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id]: break
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
                generated_token_ids.append(next_token_id.item())

            summary_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            # Adicione pós-processamento se necessário
            return summary_text.strip()
        except Exception as e:
            print("--- ERRO DURANTE INFERÊNCIA MDCRAPN ---")
            traceback.print_exc() # Imprime o traceback completo no terminal
            print("---------------------------------------")
            return f"Erro durante a execução do modelo MDCRAPN: {e}"


# --- 3. CONFIGURAÇÃO E FUNÇÃO DE GERAÇÃO (GEMINI) ---
# Coloque sua chave aqui (NÃO RECOMENDADO PARA PRODUÇÃO)
GEMINI_API_KEY = "Chave_da_API" # Substitua pela sua chave real

# Comente a linha abaixo se estiver usando a chave hardcoded
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_CONFIGURED = False
gemini_model = None

if not GEMINI_API_KEY:
    st.error("ERRO GRAVE: Chave da API Gemini não definida!")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CONFIGURED = True
        print("Gemini API configurada.")
    except Exception as e:
        st.error(f"Erro ao configurar API Gemini: {e}")

@st.cache_resource
def get_gemini_model():
    if GEMINI_CONFIGURED:
        try:
            model_name = 'gemini-2.5-flash' # Ou o nome que funcionou
            print(f"Carregando modelo Gemini: {model_name}")
            model = genai.GenerativeModel(model_name)
            return model
        except Exception as e:
            st.error(f"Erro ao inicializar modelo Gemini ('{model_name}'): {e}")
            return None
    return None

if GEMINI_CONFIGURED:
    gemini_model = get_gemini_model()

def generate_gemini_response(question_text: str, pil_image: Image.Image):
    # ... (Sua função generate_gemini_response com prompt curto, sem alterações) ...
    if not GEMINI_CONFIGURED or gemini_model is None: return "Erro: API/Modelo Gemini não pronto."
    try:
        prompt = [
            "Você é um assistente médico conciso. Sua tarefa é fornecer um breve resumo informativo, coerente e profissional para o paciente, usando o TEXTO e a IMAGEM como contexto (em 2 frases e bem direto, já diagnosticando de primeira)", # Seu prompt curto (3-5 linhas)
            "TEXTO do Paciente: " + question_text, "IMAGEM:", pil_image ]
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        safety_settings=[ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(contents=prompt, generation_config=generation_config, safety_settings=safety_settings)
        if not response.parts:
             try: return f"Erro: Resposta bloqueada (Razão: {response.prompt_feedback.block_reason})."
             except Exception: return "Erro: Resposta bloqueada pela segurança."
        return response.text.strip()
    except google.api_core.exceptions.PermissionDenied as e: return f"Erro 403: Chave API inválida/sem permissão. Detalhes: {e}"
    except google.api_core.exceptions.NotFound as e: return f"Erro 404: Modelo não encontrado/não suporta generateContent. Detalhes: {e}"
    except Exception as e: return f"Erro API Gemini: {type(e).__name__} - {e}"


# ==============================================================================
# 4. INTERFACE (UI) com Streamlit
# ==============================================================================
st.set_page_config(layout="wide", page_title="Synapse", page_icon="logo.png")

# --- Função logo ---
def get_image_as_base64(file):
    # ... (código logo) ...
    if not os.path.exists(file): return None
    try:
        with open(file, "rb") as f: return base64.b64encode(f.read()).decode()
    except Exception as e: print(f"Erro logo: {e}"); return None
logo_base64 = get_image_as_base64("logo-white.png") # Verifique nome
logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="navbar-logo">' if logo_base64 else '<span class="navbar-logo-text">Synapse</span>'

# --- CSS ---
# --- CSS (ATUALIZADO CONFORME SUAS 7 SUGESTÕES) ---
st.markdown(f"""
<style>
/* 4. Trocar tipografia */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* 3. Melhorar o contraste e cores (Variáveis) */
:root {{
    --cor-principal: #00bfa6;
    --cor-principal-hover: #009688; /* Mais escuro para hover */
    --cor-header: #00796b; /* Azul-esverdeado para cabeçalhos */
    --fundo-pagina: #f3f7f9; /* Cinza claro */
    --fundo-card: #FFFFFF;
    --cor-texto: #2E3A46;
    --border-gray: #DDE2E8;
    --gradiente-navbar: linear-gradient(90deg, #00bfa6, #00e1d4);
}}

* {{
    font-family: 'Inter', sans-serif;
}}

/* 1. Substituir o cabeçalho padrão do Streamlit */
header[data-testid="stHeader"], [data-testid="stToolbar"] {{
    display: none !important;
}}
footer {{
    display: none !important;
}}

/* Fundo da página */
html, body, .stApp {{
    background-color: var(--fundo-pagina) !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* 1. Barra superior fixa personalizada */
.navbar {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between; /* Para alinhar itens à esquerda e direita */
    padding: 0.8rem 1.5rem;
    background: var(--gradiente-navbar);
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}
.navbar-left {{
    display: flex;
    align-items: center;
}}
.navbar-logo {{
    height: 50px;
    margin-right: 1rem;
}}
.navbar-title {{
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--fundo-card); /* Branco para contraste */
}}
.navbar-right {{
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--fundo-card);
    opacity: 0.8;
}}


/* 5. Ajustar espaçamento (para compensar barra fixa) */
.main, [data-testid="block-container"] {{
    padding-top: 80px !important; /* ~80px de espaço */
    padding-bottom: 2rem !important;
}}

/* 3. Botões */
.stButton > button {{
    background-color: var(--cor-principal);
    color: var(--fundo-card);
    border: none;
    border-radius: 5px;
    padding: 10px 24px;
    font-weight: 600;
    width: 100%;
    margin-top: 1rem;
    transition: all 0.2s ease; /* 6. Microinterações */
}}
.stButton > button:hover {{
    background-color: var(--cor-principal-hover);
    transform: translateY(-2px); /* 6. Microinterações */
}}
.stButton > button:disabled {{
    background-color: #B0D9D3;
    color: #6c757d;
}}

/* 2. Reformular o layout com "cards" */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--fundo-card);
    border: 1px solid var(--border-gray);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    padding: 1.5rem !important;
    margin-bottom: 1.5rem;
    transition: all 0.2s ease; /* 6. Microinterações */
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 6px 16px rgba(0,0,0,0.07);
    transform: translateY(-2px); /* 6. Microinterações */
}}

/* Cabeçalho do Card (com ícones e cores) */
.card-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--cor-header); /* 3. Cor do cabeçalho */
    border-bottom: 1px solid var(--border-gray);
    padding-bottom: 0.75rem;
    margin-top: -0.5rem;
    margin-bottom: 1.25rem;
}}
.card-header span {{
    font-size: 1.75rem; /* 7. Ícones */
}}
.card-header h2 {{
    margin: 0;
    padding: 0;
    font-size: 1.4rem;
    font-weight: 600;
}}

/* File Uploader */
[data-testid="stFileUploader"] > div {{
    border: 2px dashed var(--border-gray);
    background-color: var(--fundo-pagina);
    border-radius: 5px;
}}
[data-testid="stFileUploader"] > div:hover {{
    border-color: var(--cor-principal);
    background-color: #E6F9F0;
}}

/* Card de Resultado (Gerado) */
.summary-card {{
    background-color: #E6F9F0;
    border: 1px solid #AAE0C6;
    border-left: 5px solid var(--cor-principal);
    border-radius: 5px;
    padding: 1.5rem;
}}
.summary-card h3 {{
    color: #005E30;
    margin-top: 0;
}}
.summary-card p {{
    color: var(--cor-texto);
    font-size: 1.05rem;
    line-height: 1.6;
}}
</style>
""", unsafe_allow_html=True)


# --- Navbar Fixa (ATUALIZADA) ---
st.markdown(f"""
<nav class="navbar">
    <div class="navbar-left">
        {logo_html}
        <span class="navbar-title">Sistema de Resumo Multimodal de Saúde</span>
    </div>
</nav>
""", unsafe_allow_html=True)

# ======================================================
# 5. LAYOUT DA APLICAÇÃO (Com Duas Saídas)
# ======================================================

# --- Bloco de Entrada de Dados ---
with st.container(border=True):
    st.markdown("<h2>⚕️ Entrada de Dados</h2>", unsafe_allow_html=True)
    col_input1, col_input2 = st.columns(2, gap="large")
    with col_input1:
        question_text = st.text_area("Pergunta / Descrição:", "My skin...", height=165)
    with col_input2:
        uploaded_file = st.file_uploader("Escolha imagem:", type=["jpg", "jpeg", "png"])
        # Botão desabilitado se Gemini não estiver pronto
        generate_button = st.button("Gerar Análise", disabled=(not GEMINI_CONFIGURED or gemini_model is None), use_container_width=True)

col_output1, col_output2 = st.columns(2, gap="large")

with col_output1:
    with st.container(border=True):
        st.markdown("<h2>🖼️ Visualização da Imagem</h2>", unsafe_allow_html=True)
        image_placeholder = st.empty()

with col_output2:
    with st.container(border=True):
        st.markdown("<h2>📄 Análise e Resultado (Assistente Virtual)</h2>", unsafe_allow_html=True)
        result_placeholder_gemini = st.empty() # Placeholder para Gemini

        # --- Área para MDCRAPN dentro do mesmo card ---
        st.markdown("<br><hr style='margin: 1.5rem 0;'><br>", unsafe_allow_html=True) # Separador visual mais espaçado
        st.markdown("<h3>🔬 Análise Técnica (Modelo MDCRAPN)</h3>", unsafe_allow_html=True)
        result_placeholder_mdcrapn = st.empty() # Placeholder para MDCRAPN
        # --- Fim ---

# --- Lógica Placeholders ---
image_to_process = None
if uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert('RGB')
        image_placeholder.image(image_to_process, caption='Imagem Carregada', use_container_width=True)
    except Exception as e:
        image_placeholder.error(f"Erro ao ler imagem: {e}")
        image_to_process = None
else:
    image_placeholder.info("Faça o upload de uma imagem.")

# Mensagens iniciais
if not generate_button:
    if not GEMINI_CONFIGURED or gemini_model is None: result_placeholder_gemini.warning("API Gemini não configurada.")
    else: result_placeholder_gemini.info("Resultado do assistente virtual aparecerá aqui.")
    if not LOCAL_MODEL_LOADED: result_placeholder_mdcrapn.warning("Modelo local MDCRAPN não carregado.")
    else: result_placeholder_mdcrapn.info("Resultado técnico do modelo MDCRAPN aparecerá aqui.")

# --- Lógica de Geração ---
if generate_button:
    if uploaded_file is None or question_text.strip() == "": st.error("Insira texto E imagem.")
    elif image_to_process is None: st.error("Erro ao carregar imagem.")
    else:
        # --- GERAÇÃO GEMINI ---
        with st.spinner("Gerando análise do assistente virtual..."):
            try:
                summary_gemini = generate_gemini_response(question_text, image_to_process)
                if "Erro" in summary_gemini: result_placeholder_gemini.error(summary_gemini)
                else:
                    result_placeholder_gemini.success("Assistente Virtual:")
                    result_placeholder_gemini.markdown(summary_gemini)
            except Exception as e:
                result_placeholder_gemini.error("Erro inesperado (Gemini):"); result_placeholder_gemini.exception(e)

        # --- GERAÇÃO MDCRAPN ---
        if LOCAL_MODEL_LOADED:
            with st.spinner("Gerando análise técnica (MDCRAPN)..."):
                try:
                    # Chama a função run_generation (que usa seu modelo local)
                    summary_mdcrapn = run_generation(question_text, image_to_process)
                    if "Erro" in summary_mdcrapn:
                         result_placeholder_mdcrapn.error(summary_mdcrapn)
                    else:
                         # Mostra o resultado do MDCRAPN em um card diferente (laranja)
                         result_html_mdcrapn = f"""
                         <div class="summary-card-mdcrapn">
                             <h3>Output MDCRAPN</h3>
                             <p>{summary_mdcrapn if summary_mdcrapn else "Nenhuma saída gerada."}</p>
                         </div>
                         """
                         result_placeholder_mdcrapn.markdown(result_html_mdcrapn, unsafe_allow_html=True)
                except Exception as e:
                    result_placeholder_mdcrapn.error("Erro inesperado (MDCRAPN):")
                    result_placeholder_mdcrapn.exception(e)
        else:
             result_placeholder_mdcrapn.warning("Modelo MDCRAPN não carregado.")


# --- Rodapé ---
st.markdown("---")
st.caption("© 2025 Synapse. Sistema de demonstração...")