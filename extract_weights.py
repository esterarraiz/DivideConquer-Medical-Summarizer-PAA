import torch
import os

# --- Configurações ---
CHECKPOINT_FILE = 'checkpoint_epoch_90.pth' # O arquivo que você baixou
OUTPUT_GFD = 'gfd_model_weights.pth'
OUTPUT_MACDM = 'macdm_model_weights.pth'
OUTPUT_MDCRAPN = 'mdcrapn_model_weights.pth'
# Use 'cpu' para garantir que funcione mesmo sem GPU local
DEVICE = torch.device('cpu')
# --- Fim Configurações ---

print(f"Tentando carregar o checkpoint: {CHECKPOINT_FILE}")

if not os.path.exists(CHECKPOINT_FILE):
    print(f"ERRO: Arquivo de checkpoint '{CHECKPOINT_FILE}' não encontrado.")
    print("Certifique-se de que ele está na mesma pasta que este script.")
else:
    try:
        # Carrega o checkpoint inteiro (permitindo carregar objetos não-tensor)
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        print("Checkpoint carregado com sucesso.")

        # Extrai os state_dicts de cada modelo
        gfd_state_dict = checkpoint.get('gfd_state_dict')
        macdm_state_dict = checkpoint.get('macdm_state_dict')
        mdcrapn_state_dict = checkpoint.get('mdcrapn_state_dict')

        if gfd_state_dict and macdm_state_dict and mdcrapn_state_dict:
            print("State dicts encontrados no checkpoint.")

            # Salva cada state_dict em um arquivo .pth separado
            torch.save(gfd_state_dict, OUTPUT_GFD)
            print(f"Pesos do GFD salvos em: {OUTPUT_GFD}")

            torch.save(macdm_state_dict, OUTPUT_MACDM)
            print(f"Pesos do MACDM salvos em: {OUTPUT_MACDM}")

            torch.save(mdcrapn_state_dict, OUTPUT_MDCRAPN)
            print(f"Pesos do MDCRAPN salvos em: {OUTPUT_MDCRAPN}")

            print("\nExtração concluída com sucesso!")
            print("Agora você pode usar esses arquivos .pth no seu app.py.")
        else:
            print("ERRO: Não foi possível encontrar os state_dicts esperados ('gfd_state_dict', 'macdm_state_dict', 'mdcrapn_state_dict') dentro do arquivo de checkpoint.")
            print("Verifique como o checkpoint foi salvo no script de treinamento.")

    except Exception as e:
        print(f"ERRO ao processar o checkpoint: {e}")
        print("Verifique se o arquivo não está corrompido e se foi salvo corretamente.")