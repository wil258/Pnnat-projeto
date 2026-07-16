import os
from ultralytics import YOLO

def main():
    # 1. Definição de caminhos e parâmetros
    # Certifique-se de que o arquivo 'data.yaml' está configurado apontando para as pastas de imagens
    DATASET_CONFIG = "data.yaml" 
    BASE_MODEL = "yolov8n.pt"      # Versão 'nano' (ideal para recursos limitados/Edge AI)
    EPOCHS = 30                    # Ajuste conforme a necessidade de convergência do dataset
    IMG_SIZE = 640                 # Resolução padrão de entrada para YOLOv8
    
    print("=== [1/4] Inicializando o Modelo YOLO ===")
    if not os.path.exists(DATASET_CONFIG):
        raise FileNotFoundError(
            f"O arquivo de configuração {DATASET_CONFIG} não foi encontrado! "
            f"Verifique se você está executando o script no diretório correto."
        )
        
    # Inicializa o modelo nano pré-treinado na COCO dataset
    m
    print("\n=== [2/4] Iniciando o Fine-Tuning do Modelo ===")
    # O treinamento gerará métricas automáticas na pasta 'runs/detect/train'
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device="auto",             # Seleciona GPU automaticamente se disponível, senão CPU
        workers=4,                 # Otimiza o carregamento de dados (data loading)
        project="runs",            # Pasta onde salvar os resultados
        name="mascaras_yolo"
    )
    
    print("\n=== [3/4] Avaliando Métricas de Validação (mAP) ===")
    # Executa a validação no conjunto de teste/validação oficial
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n=== [4/4] Exportando o Modelo para Formatos de Borda (Edge AI) ===")
    
    # Exportação 1: ONNX (Excelente para NPUs, CPUs e servidores embarcados)
    # Ativamos 'half=True' para quantizar os pesos de FP32 para FP16 (reduzindo o tamanho do modelo pela metade)
    print("-> Exportando para ONNX (FP16)...")
    onnx_path = model.export(format="onnx", half=True, imgsz=IMG_SIZE)
    print(f"Modelo ONNX salvo com sucesso em: {onnx_path}")
    
    # Exportação 2: TFLite (Ideal para dispositivos Android, Raspberry Pi e microcontroladores mais potentes)
    print("-> Exportando para TFLite (FP16)...")
    tflite_path = model.export(format="tflite", half=True, imgsz=IMG_SIZE)
    print(f"Modelo TFLite salvo com sucesso em: {tflite_path}")

    print("\n=======================================================")
    print(" Treinamento, validação e otimizações concluídos! 🚀")
    print("=======================================================")
    # Caminhos relativos ou absolutos para as pastas do seu dataset
path: ./dataset  # Pasta raiz do dataset (se houver)
train: train/images
val: val/images
test: test/images # Opcional

# Número de classes e nomes correspondentes
nc: 2
names:
  0: com_mascara
  1: sem_mascara
  # Caminhos relativos ou absolutos para as pastas do seu dataset
path: ./dataset  # Pasta raiz do dataset (se houver)
train: train/images
val: val/images
test: test/images # Opcional

# Número de classes e nomes correspondentes
nc: 2
names:
  0: com_mascara
  1: sem_mascara
  # Caminhos relativos ou absolutos para as pastas do seu dataset
path: ./dataset  # Pasta raiz do dataset (se houver)
train: train/images
val: val/images
test: test/images # Opcional

# Número de classes e nomes correspondentes
nc: 2
names:
  0: com_mascara
  1: sem_mascara

if __name__ == "__main__":
    main()
