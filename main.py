import random
import tkinter as tk
import cv2
import numpy as np

# _______________________________________________________________________________PARTIE DONNEES ________________________________________________________________________________________________

# Charger le détecteur de visages pour la détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Charger le détecteur des yeux pour le filtre des lunettes
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
# Charger le détecteur de sourire pour le filtre de moustache
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# Charger l'image png des lunettes
sunglasses = cv2.imread('sunglasses.png', -1)
# Charger l'image png des moustaches
mustache = cv2.imread('moustache.png', -1)
# Charger l'image PNG des flocons de neige et modification de sa taille en 30x30 pour le filtre interactif.
snowflake_img = cv2.imread('snow.png', -1)
snowflake_img = cv2.resize(snowflake_img, (30, 30))  # Adjust the size as needed
# Initialiser la webcam
cap = cv2.VideoCapture(0)

# ____________________________________________________________________________ CREATION DU MENU __________________________________________________________________________
# Création d'une qui contient les élements du menu interactif
menu_items = ["Sepia", "Constract", "Lunette", "Moustache", "Snow flake","Background", "Annuler filtre", "Lancer Tous"]
# Initialiser la valeur sélection à None
selected_value = None


# Créer une fonction qui assigne la valeur sélectionnée dans le menu à la variable 'selected_value' pour récupérer la valeur sélectionnée
def button_click(item):
    global selected_value
    selected_value = item


# Création de la fenêtre Tkinter du menu en dehors de la boucle
root = tk.Tk()
root.title("Menu interactif")

# Créer les boutons une seule fois avec une fonction de rappel
buttons = [tk.Button(root, text=item, command=lambda i=item: button_click(i)) for item in menu_items]
for button in buttons:
    button.pack(pady=5)


# _________________________________________________________________________ 2.a CREATION DES FILTRES ____________________________________________________________________________

# Création de la fonction Sepia qui appliquera le filtre sepia sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def Sepia(frame):
    # Appliquer le filtre sépia à l'image entière
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    frame = cv2.transform(frame, sepia_filter)
    # Retourner la frame résultante
    return frame


# Création de la fonction Constract qui appliquera le filtre constract sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def constract(frame):
    # Appliquer le filtre de modification de contraste à l'image entière
    alpha = 1.5
    beta = 50

    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Retourner la frame résultante
    return frame


# _______________________________________________________________________ 2.b INCRUSTATION D'UNE IMAGE _______________________________________________________________________________

# Création de la fonction Lunette qui appliquera le filtre des lunettes sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def lunette(frame):
    # Convertir le frame en niveaux de gris.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectection du visage
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Iteration sur chaque visage détecté
    for (x, y, w, h) in faces:
        # Detectection des yeux dans la cadre de chaque visage détecté
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        # Si les yeux sont bien détectés, calcler à peut prés leur position (la moyenne)
        if len(eyes) >= 2:
            eye_centers = np.array([(x + ex + ew // 2, y + ey + eh // 2) for (ex, ey, ew, eh) in eyes])
            avg_eye_center = np.mean(eye_centers, axis=0).astype(int)

            # Calculer la taille de l'image des lunettes de soleil en fonction de la distance entre les yeux
            eye_distance = np.linalg.norm(eye_centers[0] - eye_centers[1])
            sunglasses_resized = cv2.resize(sunglasses, (int(2 * eye_distance), int(eye_distance)))

            # Calculer la position pour placer les lunettes de soleil
            x_sunglasses = avg_eye_center[0] - int(eye_distance)
            y_sunglasses = avg_eye_center[1] - int(eye_distance / 2)

            # S'assurer que les lunettes de soleil restent à l'intérieur de la frame
            if x_sunglasses >= 0 and y_sunglasses >= 0 and x_sunglasses + sunglasses_resized.shape[1] <= frame.shape[
                1] and y_sunglasses + sunglasses_resized.shape[0] <= frame.shape[0]:
                # Fusionner les lunettes de soleil avec le frame
                alpha_s = sunglasses_resized[:, :, 3] / 255.0
                alpha_frame = 1.0 - alpha_s

                for c in range(0, 3):
                    frame[y_sunglasses:y_sunglasses + sunglasses_resized.shape[0],
                    x_sunglasses:x_sunglasses + sunglasses_resized.shape[1], c] = (
                            alpha_s * sunglasses_resized[:, :, c] + alpha_frame * frame[y_sunglasses:y_sunglasses +
                                                                                                     sunglasses_resized.shape[
                                                                                                         0],
                                                                                  x_sunglasses:x_sunglasses +
                                                                                               sunglasses_resized.shape[
                                                                                                   1], c]
                    )
    # Retourner la frame résultante
    return frame


# Création de la fonction Moustache qui appliquera le filtre des moustaches sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def moustache(frame):
    # Convertir le frame en niveaux de gris.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectection du visage
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Iteration sur chaque visage détecté
    for (x, y, w, h) in faces:
        # Détecter les sourires dans la région du visage.
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # Iteration sur chaque sourire détectée
        for (sx, sy, sw, sh) in smiles:
            # Redimensionner l'image de la moustache pour qu'elle s'adapte à la région de la bouche.
            mustache_resized = cv2.resize(mustache, (sw, int(0.5 * sh)))

            # Coordonnées pour placer la moustache sur le frame
            mustache_x = x + sx
            mustache_y = y + int(-1.8 * sy)

            # Extraire le canal alpha de l'image de la moustache
            alpha_mask = mustache_resized[:, :, 3] / 255.0

            # Intégrer la moustache sur le cadre
            for c in range(0, 3):
                frame[mustache_y:mustache_y + mustache_resized.shape[0],
                mustache_x:mustache_x + mustache_resized.shape[1], c] = \
                    frame[mustache_y:mustache_y + mustache_resized.shape[0],
                    mustache_x:mustache_x + mustache_resized.shape[1], c] * (1 - alpha_mask) + \
                    mustache_resized[:, :, c] * alpha_mask
            # Ajouter le texte 'SMILE !' au-dessus du cadre puisque la vouche est détectée avec un sourire
            cv2.putText(frame, 'SMILE!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Retourner la frame résultante
    return frame


# _______________________________________________________________________ 2.c INCRUSTATION D'UNE IMAGE INTERACTIVE ______________________________________________________________________

# Initialisez la liste pour stocker les flocons de neige
snowflakes = []


# Fonction pour initialiser un nouveau flocon de neige
def initialize_snowflake():
    x = random.randint(0, cap.get(3) - 30)
    y = random.randint(0, cap.get(4) - 30)
    return {"x": x, "y": y}


# Fonction pour vérifier si un point se trouve dans la région du visage détectée
def is_point_inside_face(point, face_regions):
    for (x, y, w, h) in face_regions:
        if x - 30 < point["x"] < x + w + 30 and y - 50 < point["y"] < y + h:
            return True
    return False


# Ajouter les flocons de neige initiaux
num_initial_snowflakes = 100
for _ in range(num_initial_snowflakes):
    snowflakes.append(initialize_snowflake())


# Création de la fonction Snowflake qui appliquera le filtre des flocons de neige interactifs sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def snowflake(frame):
    # Convertir le frame en niveaux de gris.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectection du visage
    face_regions = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Mettre à jour et dessiner chaque flocon de neige
    for snowflake in snowflakes:
        alpha_channel = snowflake_img[:, :, 3] / 255.0

        # Vérifier si le flocon de neige est à l'intérieur de la région du visage détectée
        if is_point_inside_face(snowflake, face_regions):
            # If yes, reset its position to the top
            snowflake["y"] = 0
            snowflake["x"] = random.randint(0, cap.get(3) - 30)

        for c in range(0, 3):
            frame[snowflake["y"]:snowflake["y"] + 30, snowflake["x"]:snowflake["x"] + 30, c] = \
                (1 - alpha_channel) * frame[snowflake["y"]:snowflake["y"] + 30, snowflake["x"]:snowflake["x"] + 30, c] + \
                alpha_channel * snowflake_img[:, :, c]

        # Mettre à jour la position du flocon de neige pour la prochaine itération
        snowflake["y"] += 1
        # Si le flocon de neige atteint le bas, réinitialiser sa position en haut
        if snowflake["y"] > cap.get(4) - 30:
            snowflake["y"] = 0
            snowflake["x"] = random.randint(0, cap.get(3) - 30)
    return frame


# _______________________________________________________________________  2.d CHANGER LE FOND DE LA VIDEO ______________________________________________________________________
# Création de la fonction change_background_to_white qui appliquera le filtre du fond blanc sur les frames du vidéo et retourne la nouvelle frame aprés l'application du filtre
def change_background_to_white(frame):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Créer une image de fond blanche
    background = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255

    # Mettre à jour la région du fond avec la région du visage
    for (x, y, w, h) in faces:
        background[y:y + h, x:x + w, :] = frame[y:y + h, x:x + w, :]

    return background


# _______________________________________________________________________  3. LECTURE DE LA VIDEO CAMERA ET APPLICATION DES FILTRES ________________________________________________________________

while True:
    # Lire une image de la séquence vidéo
    ret, frame = cap.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dessiner des rectangles autour des qqqqqvisages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Vérifier l'effet que l'utilisateur veut ajouter sur la video
    if selected_value == "Sepia":
        frame = Sepia(
            frame)  # S'il sélectionne Sépia, alors nous appelons la fonction crée Sepia() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Lunette":
        frame = lunette(
            frame)  # S'il sélectionne Lunette, alors nous appelons la fonction crée lunette() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Constract":
        frame = constract(
            frame)  # S'il sélectionne Constract, alors nous appelons la fonction crée constract() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Moustache":
        frame = moustache(
            frame)  # S'il sélectionne Moustache, alors nous appelons la fonction crée moustache() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Snow flake":
        frame = snowflake(
            frame)  # S'il sélectionne Snow flake, alors nous appelons la fonction snowflake() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Background":
        frame = change_background_to_white(
            frame)# S'il sélectionne 'background', alors nous appelons la fonction change_background_to_white() et l'appliquons sur chaque image
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == "Lancer Tous":
        frame = snowflake(Sepia(lunette(moustache(
            frame))))  # S'il sélectionne 'Lancer Tous', alors nous appelons toutes les fonctions de filtres et les appliquons les unes après les autres
        cv2.imshow('Webcam Face Detection', frame)
    elif selected_value == None or selected_value == "Annuler filtre":
        cv2.imshow('Webcam Face Detection',
                   frame)  # S'il sélectionne 'Annuler filtre', alors on fait rien on affiche l'orignal frame

    # Mettre à jour la fenêtre Tkinter
    root.update()

    # Lire la touche pressée
    key = cv2.waitKey(1) & 0xFF

    # Quitter la boucle si la touche 'q' est pressée
    if key == ord('q'):
        # Détruire la fenêtre Tkinter
        root.destroy()
        break

# Libérer la webcam et détruire toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()

