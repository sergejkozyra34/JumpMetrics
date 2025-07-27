import cv2
import mediapipe as mp
import numpy as np
import telebot
from telebot import types
import tempfile
import os
import logging
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot('YOUR_TELEGRAM_BOT_TOKEN')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

USER_DATA = {}

main_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
main_markup.row(types.KeyboardButton("📏 Измерить прыжок"))
main_markup.row(types.KeyboardButton("📏 Указать рост"), types.KeyboardButton("🔄 Сбросить"))

def calculate_pixel_to_meter_ratio(calibration_frames, user_height):
    if not calibration_frames or user_height <= 0:
        return None
    
    heights_px = []
    for frame_data in calibration_frames:
        landmarks, frame_height = frame_data
        try:
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            ankle_mid_y = (left_ankle.y + right_ankle.y) * 0.5
            height_px = abs(nose.y - ankle_mid_y) * frame_height
            
            if height_px > 0:
                heights_px.append(height_px)
        except:
            pass
    
    if not heights_px:
        return None
    
    return np.mean(heights_px) / user_height

def analyze_jump(video_path, user_data):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    pixel_per_meter = calculate_pixel_to_meter_ratio(user_data['calibration_frames'], user_data['height'])
    if pixel_per_meter is None:
        return None, None

    positions = []
    start_position = None
    calibration_complete = False
    jump_detected = False
    max_height = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ankle_mid = (
                (left_ankle.x + right_ankle.x) * 0.5 * w,
                (left_ankle.y + right_ankle.y) * 0.5 * h
            )
            
            if not calibration_complete:
                if time.time() - start_time > 3:
                    calibration_complete = True
                    start_position = ankle_mid
                continue
            
            positions.append(ankle_mid)
            
            current_height = ankle_mid[1]
            if not jump_detected and current_height < start_position[1] - 0.05 * h:
                jump_detected = True
            
            if jump_detected:
                max_height = max(max_height, start_position[1] - current_height)

    cap.release()
    
    if not positions:
        return None, None
    
    jump_height = min(max_height / pixel_per_meter, 2.0)
    
    start_x = start_position[0]
    end_x = positions[-1][0]
    jump_length = min(abs(end_x - start_x) / pixel_per_meter * 0.7, 4.0)
    
    return jump_height, jump_length

def create_results_plot(height, length):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(['Высота прыжка', 'Длина прыжка'], [height, length], color=['#4CAF50', '#2196F3'])
    ax.set_ylabel('Метры')
    ax.set_title('Результаты измерения прыжка')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f} м', ha='center', va='bottom')
    
    plot_path = 'jump_results.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    return plot_path

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    user_id = message.from_user.id
    if user_id not in USER_DATA:
        USER_DATA[user_id] = {'height': None, 'state': 'waiting', 'calibration_frames': []}
    
    bot.send_message(
        message.chat.id,
        "🏆 *Точный измеритель прыжков*\n\nОтправь видео прыжка после указания роста.",
        reply_markup=main_markup,
        parse_mode='Markdown'
    )

@bot.message_handler(func=lambda message: message.text == "📏 Указать рост")
def request_height(message):
    msg = bot.send_message(message.chat.id, "Введи свой рост в метрах (например: 1.75):", reply_markup=types.ForceReply())
    bot.register_next_step_handler(msg, process_height_step)

def process_height_step(message):
    user_id = message.from_user.id
    try:
        height = float(message.text.replace(',', '.'))
        if 1.0 <= height <= 2.5:
            USER_DATA[user_id]['height'] = height
            bot.send_message(message.chat.id, f"✅ Рост {height:.2f} м сохранен!", reply_markup=main_markup)
        else:
            raise ValueError()
    except:
        bot.send_message(message.chat.id, "❌ Неверный формат. Введи число от 1.0 до 2.5", reply_markup=main_markup)

@bot.message_handler(func=lambda message: message.text == "🔄 Сбросить")
def reset_user(message):
    user_id = message.from_user.id
    USER_DATA[user_id] = {'height': None, 'state': 'waiting', 'calibration_frames': []}
    bot.send_message(message.chat.id, "🔄 Данные сброшены", reply_markup=main_markup)

@bot.message_handler(func=lambda message: message.text == "📏 Измерить прыжок")
def request_video(message):
    user_id = message.from_user.id
    if user_id not in USER_DATA or USER_DATA[user_id]['height'] is None:
        bot.send_message(message.chat.id, "Сначала укажи рост", reply_markup=main_markup)
        return
    
    USER_DATA[user_id]['state'] = 'waiting_video'
    USER_DATA[user_id]['calibration_frames'] = []
    bot.send_message(message.chat.id, "Отправь видео прыжка (первые 3 сек стой неподвижно)", reply_markup=main_markup)

@bot.message_handler(content_types=['video', 'video_note'])
def handle_video(message):
    user_id = message.from_user.id
    if user_id not in USER_DATA or USER_DATA[user_id]['state'] != 'waiting_video':
        return

    try:
        file_id = message.video_note.file_id if message.content_type == 'video_note' else message.video.file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(downloaded_file)
            temp_video_path = temp_video.name
        
        bot.send_message(message.chat.id, "🔍 Анализирую...", reply_markup=main_markup)
        
        cap = cv2.VideoCapture(temp_video_path)
        calibration_frames = []
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 640) if message.content_type == 'video_note' else (640, 480))
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks and time.time() - start_time < 3.0:
                calibration_frames.append((results.pose_landmarks.landmark, frame.shape[0]))
        
        cap.release()
        
        if len(calibration_frames) < 10:
            raise ValueError("Нужно стоять неподвижно первые 3 секунды")
        
        USER_DATA[user_id]['calibration_frames'] = calibration_frames
        height, length = analyze_jump(temp_video_path, USER_DATA[user_id])
        os.unlink(temp_video_path)
        
        if height is not None and length is not None:
            plot = create_results_plot(height, length)
            with open(plot, 'rb') as p:
                bot.send_photo(
                    message.chat.id, 
                    p, 
                    caption=f"📊 Результаты:\nВысота: {height:.2f} м\nДлина: {length:.2f} м",
                    reply_markup=main_markup
                )
            os.unlink(plot)
        else:
            bot.send_message(message.chat.id, "❌ Не удалось измерить прыжок", reply_markup=main_markup)
    
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {str(e)}", reply_markup=main_markup)
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

if __name__ == "__main__":
    bot.infinity_polling()