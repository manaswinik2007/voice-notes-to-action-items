import os
import re
import whisper

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".env"
    )
)


# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Upload folder
# --------------------------------------------------

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# Load Whisper locally
# --------------------------------------------------

print("Loading Whisper model...")

model = whisper.load_model("tiny")

print("Whisper model loaded successfully!")


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/")
def home():

    return jsonify({
        "message": "Voice Notes to Action Items API is running!"
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file uploaded"
        }), 400


    audio = request.files["audio"]


    if audio.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400


    filename = secure_filename(audio.filename)


    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    audio.save(path)


    try:

        result = model.transcribe(path)

        text = result["text"].strip()


        return jsonify({
            "text": text
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# PRIORITY DETECTION
# --------------------------------------------------

def detect_priority(text):

    text_lower = text.lower()


    high_priority_words = [

        "urgent",
        "urgently",
        "asap",
        "immediately",
        "critical",
        "high priority",
        "right away",
        "at once"
    ]


    if any(
        word in text_lower
        for word in high_priority_words
    ):

        return "High"


    medium_priority_words = [

        "important",
        "priority",
        "soon"
    ]


    if any(
        word in text_lower
        for word in medium_priority_words
    ):

        return "Medium"


    return "Low"


# --------------------------------------------------
# DEADLINE DETECTION
# --------------------------------------------------

def detect_deadline(text):

    text_lower = text.lower()


    patterns = [

        r"\b(immediately|urgent(?:ly)?|asap|right away|at once)\b",

        r"\bby (tomorrow|today|tonight)\b",

        r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",

        r"\bby (next week|next month)\b",

        r"\bon (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",

        r"\bbefore (tomorrow|today|tonight)\b",

        r"\bbefore (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",

        r"\bwithin (\d+ days?)\b"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text_lower
        )


        if match:

            return match.group(0).strip()


    return "Not specified"


# --------------------------------------------------
# PERSON EXTRACTION
# --------------------------------------------------

def extract_person(text):

    patterns = [

        r"\b([A-Z][a-z]+)\s+(?:needs to|need to|should|must|has to|have to|will)\b",

        r"\b([A-Z][a-z]+)\s+(?:needs|should|must|has|will)\b"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )


        if match:

            return match.group(1)


    # Case-insensitive version
    # Useful when Whisper converts names to lowercase

    pattern = r"\b([A-Za-z]+)\s+(?:needs to|need to|should|must|has to|have to|will)\b"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )


    if match:

        name = match.group(1)


        ignored_words = {

            "i",
            "we",
            "you",
            "they",
            "he",
            "she",
            "the",
            "someone",
            "please",
            "kindly"
        }


        if name.lower() not in ignored_words:

            return name.capitalize()


    return "Unassigned"


# --------------------------------------------------
# TASK EXTRACTION
# --------------------------------------------------

def extract_task(text):

    patterns = [

        r"(?:needs to)\s+(.+)",

        r"(?:need to)\s+(.+)",

        r"(?:should)\s+(.+)",

        r"(?:must)\s+(.+)",

        r"(?:has to)\s+(.+)",

        r"(?:have to)\s+(.+)",

        r"(?:please)\s+(.+)",

        r"(?:kindly)\s+(.+)",

        r"(?:will)\s+(.+)"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )


        if match:

            task = match.group(1).strip()


            # Remove deadline from task

            task = re.split(

                r"\s+(?:by|before|on|within)\s+",

                task,

                flags=re.IGNORECASE

            )[0].strip()


            # Remove urgency words

            task = re.sub(

                r"\s+(?:immediately|urgently|asap|right away|at once)$",

                "",

                task,

                flags=re.IGNORECASE

            ).strip()


            return task


    return text.strip()


# --------------------------------------------------
# ACTION ITEM EXTRACTION
# --------------------------------------------------

@app.route("/extract", methods=["POST"])
def extract():

    data = request.get_json()


    if not data or not data.get("text"):

        return jsonify({
            "error": "No transcript provided"
        }), 400


    text = data["text"].strip()


    # ------------------------------------------
    # Split transcript into sentences
    # ------------------------------------------

    sentences = re.split(
        r"[.!?]+",
        text
    )


    # ------------------------------------------
    # Split multiple actions connected with "and"
    # ------------------------------------------

    expanded_sentences = []


    for sentence in sentences:

        parts = re.split(

            r"\s+and\s+(?=[A-Z][a-z]+\s+(?:needs to|should|must|has to|will))",

            sentence

        )


        expanded_sentences.extend(parts)


    sentences = expanded_sentences


    action_items = []


    # ------------------------------------------
    # Action keywords
    # ------------------------------------------

    action_keywords = [

        "need to",
        "needs to",
        "should",
        "must",
        "has to",
        "have to",
        "will",
        "please",
        "kindly",

        "complete",
        "prepare",
        "submit",
        "send",
        "finish",
        "review",
        "create",
        "update",
        "check",
        "make",
        "write",
        "upload",
        "download",
        "attend",
        "call",
        "contact"
    ]


    # ------------------------------------------
    # Process every sentence
    # ------------------------------------------

    for sentence in sentences:

        sentence = sentence.strip()


        if not sentence:

            continue


        sentence_lower = sentence.lower()


        # ------------------------------------------
        # Check whether sentence contains action
        # ------------------------------------------

        is_action = any(

            keyword in sentence_lower

            for keyword in action_keywords

        )


        if is_action:

            person = extract_person(sentence)

            task = extract_task(sentence)

            deadline = detect_deadline(sentence)

            priority = detect_priority(sentence)


            action_items.append({

                "person": person,

                "task": task,

                "deadline": deadline,

                "priority": priority,

                "status": "Pending"
            })


    # ------------------------------------------
    # If no action item detected
    # ------------------------------------------

    if not action_items:

        action_items.append({

            "person": "Unassigned",

            "task": text,

            "deadline": "Not specified",

            "priority": "Low",

            "status": "Pending"
        })


    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return jsonify({

        "action_items": action_items
    })


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )