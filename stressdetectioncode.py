import cv2
import speech_recognition as sr
from textblob import TextBlob
import numpy as np
import csv
import pyttsx3
from deepface import DeepFace
import threading
import time


# =========================================================
# VOICE ENGINE
# =========================================================

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
except Exception as e:
    print("Voice engine error:", e)
    engine = None


def speak(text):

    if engine is None:
        return

    def voice_thread():

        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)

    threading.Thread(
        target=voice_thread,
        daemon=True
    ).start()


# =========================================================
# FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    print("Face detector could not be loaded.")
    exit()


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")
    print("Try VideoCapture(1) if needed.")
    exit()

print("Camera started")
print()
print("Controls:")
print("V = Voice")
print("H = Heart Rate")
print("S = Save Report")
print("Q = Quit")
print()


# =========================================================
# VARIABLES
# =========================================================

facial_stress = 0
voice_stress = 0
pulse_stress = 0

heart_rate = 0

overall_stress = 0

voice_text = ""

facial_expression = "Detecting"

expression_percentage = 0

stress_level = "WAITING"

last_emotion_check = 0

emotion_interval = 0.7


# =========================================================
# INPUT STATUS
# =========================================================

facial_input_received = False
voice_input_received = False
heart_rate_received = False


# =========================================================
# EMOTION → STRESS
# =========================================================

emotion_stress = {
    "happy": 10,
    "neutral": 15,
    "surprise": 35,
    "sad": 60,
    "fear": 80,
    "disgust": 85,
    "angry": 90
}


# =========================================================
# SUGGESTION
# =========================================================

def get_suggestion(stress):

    if stress < 30:
        return "You are relaxed."

    elif stress < 60:
        return "Moderate stress. Take deep breaths."

    else:
        return "High stress detected. Please relax and take rest."


# =========================================================
# PRINT FINAL RESULT
# =========================================================

def print_final_result():

    global overall_stress
    global stress_level

    # Make sure all inputs are available

    if not (
        facial_input_received
        and voice_input_received
        and heart_rate_received
    ):

        return


    # Calculate overall stress

    overall_stress = int(
        (
            facial_stress
            + voice_stress
            + pulse_stress
        ) / 3
    )


    # Keep between 0 and 100

    overall_stress = max(
        0,
        min(overall_stress, 100)
    )


    # Determine level

    if overall_stress < 30:

        stress_level = "LOW"

    elif overall_stress < 60:

        stress_level = "MODERATE"

    else:

        stress_level = "HIGH"


    suggestion = get_suggestion(
        overall_stress
    )


    # =====================================================
    # PRINT RESULT
    # =====================================================

    print()
    print("=" * 60)
    print("                 FINAL STRESS RESULT")
    print("=" * 60)

    print(
        "Facial Expression       :",
        facial_expression.upper()
    )

    print(
        "Expression Percentage   :",
        expression_percentage,
        "%"
    )

    print(
        "Facial Stress           :",
        facial_stress,
        "%"
    )

    print(
        "Voice Input             :",
        voice_text
    )

    print(
        "Voice Stress            :",
        voice_stress,
        "%"
    )

    print(
        "Heart Rate              :",
        heart_rate,
        "BPM"
    )

    print(
        "Pulse Stress            :",
        pulse_stress,
        "%"
    )

    print("-" * 60)

    print(
        "OVERALL STRESS          :",
        overall_stress,
        "%"
    )

    print(
        "STRESS LEVEL            :",
        stress_level
    )

    print(
        "SUGGESTION              :",
        suggestion
    )

    print("=" * 60)
    print()


# =========================================================
# VOICE DETECTION
# =========================================================

def detect_voice():

    global voice_stress
    global voice_text
    global voice_input_received

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            print()
            print("Speak now...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=8
            )


        print("Processing voice...")


        text = recognizer.recognize_google(
            audio
        )


        voice_text = text


        print(
            "You said:",
            text
        )


        # =================================================
        # SENTIMENT ANALYSIS
        # =================================================

        analysis = TextBlob(text)

        polarity = analysis.sentiment.polarity


        # Convert sentiment to stress

        voice_stress = int(
            (1 - polarity) * 50
        )


        voice_stress = max(
            0,
            min(voice_stress, 100)
        )


        voice_input_received = True


        print(
            "Voice Stress:",
            voice_stress,
            "%"
        )


        # If heart rate already entered,
        # calculate final result

        if heart_rate_received:

            print_final_result()


    except sr.WaitTimeoutError:

        print(
            "No voice detected."
        )


    except sr.UnknownValueError:

        print(
            "Could not understand your voice."
        )

        voice_text = "Could not understand"


    except sr.RequestError:

        print(
            "Speech recognition service unavailable."
        )


    except Exception as e:

        print(
            "Voice error:",
            e
        )


# =========================================================
# MAIN LOOP
# =========================================================

while True:


    # =====================================================
    # READ CAMERA
    # =====================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "Cannot read camera frame."
        )

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    # =====================================================
    # FACIAL EXPRESSION DETECTION
    # =====================================================

    current_time = time.time()


    if (
        current_time
        - last_emotion_check
        > emotion_interval
    ):

        try:

            result = DeepFace.analyze(

                img_path=frame,

                actions=["emotion"],

                enforce_detection=False,

                detector_backend="opencv"

            )


            # DeepFace sometimes returns a list

            if isinstance(
                result,
                list
            ):

                result = result[0]


            # Get expression

            facial_expression = (
                result["dominant_emotion"]
            )


            # Get all emotion percentages

            emotions = result["emotion"]


            # Percentage of dominant expression

            expression_percentage = round(

                emotions.get(
                    facial_expression,
                    0
                ),

                2
            )


            # Convert expression to stress

            facial_stress = emotion_stress.get(

                facial_expression.lower(),

                0

            )


            facial_input_received = True


            last_emotion_check = current_time


        except Exception:

            pass


    # =====================================================
    # FACE RECTANGLE
    # =====================================================

    try:

        region = result["region"]

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]


        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (255, 0, 0),

            2

        )

    except:

        pass


    # =====================================================
    # HEART RATE → PULSE STRESS
    # =====================================================

    if heart_rate_received:

        if heart_rate < 70:

            pulse_stress = 20

        elif heart_rate < 90:

            pulse_stress = 50

        else:

            pulse_stress = 80

    else:

        pulse_stress = 0


    # =====================================================
    # OVERALL STRESS
    # =====================================================

    if (
        facial_input_received
        and voice_input_received
        and heart_rate_received
    ):

        overall_stress = int(

            (
                facial_stress
                + voice_stress
                + pulse_stress
            ) / 3

        )


        overall_stress = max(
            0,
            min(overall_stress, 100)
        )


        if overall_stress < 30:

            stress_level = "LOW"

        elif overall_stress < 60:

            stress_level = "MODERATE"

        else:

            stress_level = "HIGH"

    else:

        overall_stress = 0

        stress_level = "WAITING"


    # =====================================================
    # SUGGESTION
    # =====================================================

    suggestion = get_suggestion(
        overall_stress
    )


    # =====================================================
    # DISPLAY FACIAL EXPRESSION
    # =====================================================

    cv2.putText(

        frame,

        "Facial Expression: "
        + facial_expression.upper(),

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # EXPRESSION PERCENTAGE
    # =====================================================

    cv2.putText(

        frame,

        "Expression Percentage: "
        + str(expression_percentage)
        + "%",

        (10, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2

    )


    # =====================================================
    # FACIAL STRESS
    # =====================================================

    cv2.putText(

        frame,

        "Facial Stress: "
        + str(facial_stress)
        + "%",

        (10, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # VOICE STRESS
    # =====================================================

    cv2.putText(

        frame,

        "Voice Stress: "
        + str(voice_stress)
        + "%",

        (10, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # HEART RATE
    # =====================================================

    cv2.putText(

        frame,

        "Heart Rate: "
        + str(heart_rate)
        + " BPM",

        (10, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # PULSE STRESS
    # =====================================================

    cv2.putText(

        frame,

        "Pulse Stress: "
        + str(pulse_stress)
        + "%",

        (10, 180),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # INPUT STATUS
    # =====================================================

    if not facial_input_received:

        status = "Waiting for face..."

    elif not voice_input_received:

        status = "Press V for voice..."

    elif not heart_rate_received:

        status = "Press H for heart rate..."

    else:

        status = "All inputs received"


    cv2.putText(

        frame,

        status,

        (10, 215),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2

    )


    # =====================================================
    # OVERALL STRESS
    # =====================================================

    if (
        facial_input_received
        and voice_input_received
        and heart_rate_received
    ):

        cv2.putText(

            frame,

            "OVERALL STRESS: "
            + str(overall_stress)
            + "%",

            (10, 270),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 0, 255),

            3

        )


        cv2.putText(

            frame,

            "LEVEL: "
            + stress_level,

            (10, 305),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )


        cv2.putText(

            frame,

            suggestion,

            (10, 340),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1

        )

    else:

        cv2.putText(

            frame,

            "OVERALL STRESS: Waiting...",

            (10, 270),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2

        )


    # =====================================================
    # CONTROLS
    # =====================================================

    cv2.putText(

        frame,

        "V=Voice  H=Heart Rate  S=Save  Q=Quit",

        (10, 400),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1

    )


    # =====================================================
    # DISPLAY WINDOW
    # =====================================================

    cv2.imshow(

        "AI Stress Detection & Wellness System",

        frame

    )


    # =====================================================
    # KEYBOARD INPUT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # V = VOICE
    # =====================================================

    if key == ord("v"):

        detect_voice()


    # =====================================================
    # H = HEART RATE
    # =====================================================

    elif key == ord("h"):

        try:

            value = input(
                "Enter Heart Rate (BPM): "
            )

            heart_rate = int(
                value
            )


            if heart_rate <= 0:

                print(
                    "Enter a valid heart rate."
                )

                heart_rate_received = False


            else:

                heart_rate_received = True


                print(
                    "Heart Rate:",
                    heart_rate,
                    "BPM"
                )


                # Calculate pulse stress

                if heart_rate < 70:

                    pulse_stress = 20

                elif heart_rate < 90:

                    pulse_stress = 50

                else:

                    pulse_stress = 80


                # Print final result

                if (
                    facial_input_received
                    and voice_input_received
                ):

                    print_final_result()


        except ValueError:

            print(
                "Please enter a valid number."
            )


    # =====================================================
    # S = SAVE
    # =====================================================

    elif key == ord("s"):

        if (
            facial_input_received
            and voice_input_received
            and heart_rate_received
        ):

            try:

                with open(
                    "stress_report.csv",
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(
                        file
                    )


                    writer.writerow([

                        facial_expression,

                        expression_percentage,

                        facial_stress,

                        voice_text,

                        voice_stress,

                        heart_rate,

                        pulse_stress,

                        overall_stress,

                        stress_level

                    ])


                print()
                print(
                    "Stress report saved successfully."
                )


            except Exception as e:

                print(
                    "Error saving report:",
                    e
                )

        else:

            print()
            print(
                "Please provide all 3 inputs first."
            )


    # =====================================================
    # Q = QUIT
    # =====================================================

    elif key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Program closed.")