import cv2
import numpy as np
import pytesseract
from PIL import Image
import textract

import matplotlib.pyplot as pl

def open_img (img_path):
    # Загружаем файл каскада для распознавания номеров автомобилей
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    # Загружаем изображение
    img = cv2.imread(img_path)

    # Преобразуем изображение в оттенки серого для улучшения производительности
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ищем номерные знаки на изображении
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    # Отрисовываем прямоугольник вокруг найденных номерных знаков
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Вырезаем номерной знак из изображения
        plate_img = gray[y:y + h, x:x + w]

        # Применяем пороговое преобразование для улучшения контрастности
        _, plate_img = cv2.threshold(plate_img, 120, 255, cv2.THRESH_BINARY)

        # Применяем бинаризацию Отцу для улучшения качества распознавания
        plate_img = cv2.adaptiveThreshold(plate_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Используем Tesseract для распознавания символов на номерном знаке
        #plate_text = pytesseract.image_to_string(plate_img, config=custom_config)
        # plate_text = textract.process(plate_img)
        # Выводим распознанный текст
        #print("Номерной знак:", plate_text)

    # Отображаем изображение с выделенными номерными знаками
    cv2.imshow('License plates', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def main ():
    img_path = open_img(img_path='images/BMW_num.jpg')

if __name__ == '__main__':
    main()

