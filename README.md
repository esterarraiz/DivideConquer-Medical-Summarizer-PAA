# Synapse 🩺 - Sistema de Resumo Multimodal Médico

Este repositório contém a implementação do projeto Synapse, um sistema web construído com Streamlit para gerar resumos de consultas médicas multimodais (texto + imagem). O sistema utiliza duas abordagens:

1.  **Modelo MDCRAPN:** Implementação baseada no artigo de Manonmani & Malathi (2025), utilizando a estratégia Dividir e Conquistar (D&C) com módulos GFD-BERT-ViT, MACDM e MDCRAPN treinados em PyTorch.
2.  **API Google Gemini:** Utiliza um modelo generativo multimodal do Google (como o Gemini 1.5 Flash/Pro) para fornecer uma análise mais coerente e profissional, usando o texto e a imagem como contexto.

O frontend permite ao usuário inserir uma pergunta/descrição textual e fazer upload de uma imagem, recebendo como resultado principal a análise do Gemini e, opcionalmente, a saída técnica do modelo MDCRAPN treinado.

## Funcionalidades ✨

* Interface web interativa criada com Streamlit.
* Upload de imagem (sintomas visuais, exames, etc.).
* Entrada de texto (descrição de sintomas, perguntas).
* Geração de resumo/análise concisa utilizando a API Google Gemini (multimodal).
* (Opcional) Geração de resumo utilizando o modelo MDCRAPN treinado localmente (demonstração da arquitetura D&C).
* Design customizado com paleta de cores definida.

## Arquitetura (Visão Geral) 🏛️

1.  **Frontend:** Aplicação Streamlit (`app.py`) que gerencia a interface do usuário, upload de arquivos e chamadas para as IAs.
2.  **Modelo MDCRAPN (Local):**
    * Definido em `modelo.py`.
    * Composto pelos módulos GFD-BERT-ViT (extração de features), MACDM (agregação contextual) e MDCRAPN (geração de texto D&C).
    * Requer arquivos de pesos (`.pth`) treinados (gerados por `train.py` ou `extract_weights.py`).
3.  **Modelo Gemini (API):**
    * Acessado via biblioteca `google-generativeai`.
    * Utiliza um modelo multimodal pré-treinado do Google (e.g., `gemini-2.5-flash`).
    * Requer uma chave de API válida configurada.

## Tecnologias Utilizadas 🚀

* Python 3.x
* Streamlit (Frontend)
* PyTorch (Modelo MDCRAPN)
* Hugging Face Transformers (BERT)
* Hugging Face Datasets (Carregamento de dados para treino)
* google-generativeai (API Gemini)
* Pillow (Processamento de imagem)
* NumPy
* Requests, tqdm (Para download de pesos, se implementado)

## Configuração e Instalação Local ⚙️

1.  **Clonar o Repositório:**
    ```bash
    git clone [https://docs.github.com/pt/repositories/creating-and-managing-repositories/quickstart-for-repositories](https://docs.github.com/pt/repositories/creating-and-managing-repositories/quickstart-for-repositories)
    cd [nome-do-repositorio]
    ```

2.  **Criar Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Obter Pesos do Modelo MDCRAPN (.pth):**
    * **Opção A (A partir do Checkpoint):**
        * Coloque o arquivo `checkpoint_epoch_X.pth` (ou o checkpoint final do seu treino) nesta pasta.
        * Execute o script para extrair os pesos individuais:
            ```bash
            python extract_weights.py
            ```
        * Isso deve criar os arquivos `gfd_model_weights.pth`, `macdm_model_weights.pth`, e `mdcrapn_model_weights.pth`.
    * **Opção B (Download Direto):** Se você hospedou os arquivos `.pth` extraídos online (Google Drive, etc.), certifique-se de que o `app.py` tem a lógica de download configurada com os IDs/links corretos.

5.  **Configurar Chave da API Gemini (OBRIGATÓRIO):** 🔑
    * **NUNCA coloque a chave diretamente no código `.py`!**
    * **Método 1 (Variável de Ambiente - Recomendado):** Antes de rodar o Streamlit, defina a variável de ambiente no seu terminal:
        * *Windows (CMD):* `set GEMINI_API_KEY=SUA_CHAVE_API_AQUI`
        * *Windows (PowerShell):* `$env:GEMINI_API_KEY = "SUA_CHAVE_API_AQUI"`
        * *Linux/macOS:* `export GEMINI_API_KEY=SUA_CHAVE_API_AQUI`
    * **Método 2 (Arquivo `.env` - Alternativa Local):**
        * Crie um arquivo chamado `.env` na pasta do projeto.
        * Dentro do `.env`, adicione a linha: `GEMINI_API_KEY=SUA_CHAVE_API_AQUI`
        * Instale `python-dotenv`: `pip install python-dotenv`
        * No início do `app.py`, adicione:
            ```python
            from dotenv import load_dotenv
            load_dotenv()
            # ... (o resto do código que usa os.getenv continua igual)
            ```
    * Obtenha sua chave no [Google AI Studio](https://aistudio.google.com/app/api-keys).

6.  **Logo:** Coloque seu arquivo de logo (ex: `logo.png` ou `logo-white.png`) na pasta do projeto, garantindo que o nome corresponda ao usado no `app.py`.

## Executando a Aplicação ▶️

Com o ambiente virtual ativado e as dependências instaladas, execute:

```bash
streamlit run app.py
```
## 🧠 Treinamento do Modelo MDCRAPN (Opcional)

O treinamento do modelo **MDCRAPN** é **computacionalmente intensivo** e **não faz parte da execução do `app.py`**.

O script `train.py` (ou o notebook Colab `modelo1.ipynb`) contém o código completo para:

- Carregar o dataset **ArkaAcharya/MMQSD_ClipSyntel**  
- Pré-processar os dados  
- Treinar os modelos **GFD-BERT-ViT**, **MACDM** e **MDCRAPN**

### 💡 Recomendação
Execute o treinamento em um ambiente com **GPU** (ex: Google Colab, Kaggle ou servidor dedicado).

O treinamento gera arquivos de **checkpoint** (ex: `checkpoint_epoch_X.pth`) que contêm os pesos do modelo.  
Use o script `extract_weights.py` para preparar esses pesos para o `app.py`.

---

## ✍️ Autores

- **Ana Júlia Campos Vieira**  
- **Dallyla de Moraes Sousa**  
- **Ester Arraiz de Matos**

*(Universidade Federal do Tocantins - UFT)*

---

**S. P. Manonmani** e **S. Malathi**  
> Manonmani, S. P., & Malathi, S. (2026). *Multi-head divide-and-conquer residual-attention mechanism with pointer network for multimodal question summarization in healthcare.*  
> *Information Processing & Management, 63*, 104348.  
> DOI Link
