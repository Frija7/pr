import telebot
from telebot import types
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

bot = telebot.TeleBot("7894841775:AAGqjDsk-L5eKdXxdxlxO1nCPGLW6g2gLYU")


def bot_classif(image_path):
        np.set_printoptions(suppress=True)

        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r", encoding='utf-8').readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open(image_path).convert("RGB")

        size = (224, 224)
    
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
        image_array = np.asarray(image)
    
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
        data[0] = normalized_image_array
    
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
    
        return class_name[2:-1], float(prediction[0][index])


@bot.message_handler(commands=['start'])
def send_hello(message):
    bot.send_message(message.chat.id,"Привет, я EcoVisionBot, отправь мне фото мусора и я определю его тип, и расскажу способ его утилизации")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split("/")[-1]
    downloaded_file = bot.download_file(file_info.file_path)

    with open(file_name, "wb") as new_file:
        new_file.write(downloaded_file)
        class_names, confidence_score = bot_classif(file_name)
        response = f"🔍 Результат классификации:\n📌 Тип отхода: {class_names}\n🎯 Точность: {confidence_score:.2%}"
        bot.reply_to(message, response)
    

bot.infinity_polling()