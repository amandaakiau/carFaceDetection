import cv2 as cv

face_cascade = cv.CascadeClassifier('.\haarcascade\haarcascade_frontalface_default.xml')

cap = cv.VideoCapture('.\src\passageiros5.jpg')
org = (5, 30) #posição x,y do texto
color = (255, 0, 0) #cor rgb do texto

while True:

    #lê imagem/frames
    _, img = cap.read()

    #converte a imagem em escala cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detecção de faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) # img, minSize, maxSize


    cv.putText(img, "{0} faces detectadas!".format(len(faces)), org, 0, 1, color, 2) #0 font, 1 font scale, 2 linetype

    cv.imshow('img',img)

    print((len(faces))) #printa o valor de faces detectadas

    #Interrompe o código ao apertar ESC
    k = cv.waitKey(0)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()