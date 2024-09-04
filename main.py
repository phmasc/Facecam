import cv2
import os

# Diretório para salvar as imagens capturadas
SAVE_DIR = 'captured_faces'
os.makedirs(SAVE_DIR, exist_ok=True)

# Inicializa a captura de vídeo da webcam (0 indica a webcam padrão)
video_capture = cv2.VideoCapture(0)

# Verifica se a webcam foi aberta corretamente
if not video_capture.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Carrega o modelo Haar-Cascade para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Contador para nomear as imagens capturadas
image_count = 0

print("Pressione 'q' para sair.")

while True:
    # Captura frame por frame da webcam
    ret, frame = video_capture.read()

    if not ret:
        print("Erro ao capturar frame da webcam")
        break

    # Converte o frame para escala de cinza para melhorar a detecção
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Percorre os rostos detectados
    for (x, y, w, h) in faces:
        # Desenha um retângulo ao redor de cada rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extrai a imagem do rosto
        face_image = frame[y:y + h, x:x + w]

        # Salva a imagem do rosto
        face_filename = os.path.join(SAVE_DIR, f'face_{image_count}.jpg')
        cv2.imwrite(face_filename, face_image)
        image_count += 1

    # Exibe o frame com os retângulos desenhados
    cv2.imshow('Video', frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas abertas
video_capture.release()
cv2.destroyAllWindows()
