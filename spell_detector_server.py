#!/usr/bin/env python3
"""
Spell detector server for Unity integration
Usage: python spell_detector_server.py
Unity connects via HTTP to request spell detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import whisper
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)  # Allow Unity to connect

# -------- CONFIG --------
WHISPER_MODEL = "base"  # was "base" - faster for English-only
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2      # how long to record per attempt (shorter -> lower latency)
SILENCE_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.3

# Spells
SPELLS = [
    "wingardium leviosa",
    "alohomora",
    "incendio", 
    "lumos",
    "nox",
    "reparo",
    "start",
]

# Common misheard mappings
SPELL_MAPPINGS = {
    # wingardium leviosa common splits / truncations
    "wingardium": "wingardium leviosa",
    "leviosa": "wingardium leviosa",
    # existing mis-hearings for other spells
    "aloha mora": "alohomora",
    "aloha more": "alohomora",
    "allow mora": "alohomora",
    "hello": "alohomora",
    "in sendio": "incendio",
    "incentive": "incendio",
    "luminous": "lumos",
    "loom us": "lumos",
    "locks": "nox",
    "rocks": "nox",
    "repair o": "reparo",
    "re parrow": "reparo",
}

# Global state
current_detection = None
detection_lock = threading.Lock()
detection_running = threading.Event()   # indicates an active detection

# -------- Helper Functions --------
def similarity(a, b):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def token_best_match(text):
    """Return (best_spell, best_score) by scanning words/phrases in text."""
    text = text.lower().strip()
    if not text:
        return None, 0.0

    # direct mapping for common mis-hearings: check if any mapping key appears
    for k, v in SPELL_MAPPINGS.items():
        if k in text:
            return v, 0.99

    # check whole phrase contains full spell name
    for spell in SPELLS:
        if spell in text:
            return spell, 0.95

    # otherwise split into tokens and compute best token/subphrase similarity
    words = text.split()
    best_spell, best_score = None, 0.0
    # check single tokens
    for w in words:
        for spell in SPELLS:
            score = similarity(w, spell)
            if score > best_score:
                best_score, best_spell = score, spell
    # also check all contiguous subphrases up to length 3 (handles "wingardium leviosa")
    for L in (2, 3):
        for i in range(0, max(0, len(words) - L + 1)):
            phrase = " ".join(words[i:i+L])
            for spell in SPELLS:
                score = similarity(phrase, spell)
                if score > best_score:
                    best_score, best_spell = score, spell

    return best_spell, best_score

# -------- Audio Setup --------
print("Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)
print("✓ Whisper model loaded")
print("✓ Microphone ready (will use blocking sd.rec)")

# -------- Detection Thread --------
def detect_spell_worker(expected_spell, timeout_seconds):
    """Record short chunks (blocking) repeatedly up to timeout and transcribe."""
    global current_detection
    detection_running.set()
    start = time.time()
    found = False
    elapsed = 0.0

    print(f"🎯 Listening for '{expected_spell}' (timeout: {timeout_seconds}s)")

    while time.time() - start < timeout_seconds:
        # record a short chunk (blocking)
        try:
            rec = sd.rec(int(CHUNK_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = rec.flatten()
        except Exception as e:
            print(f"   ⚠️ Audio capture error: {e}")
            break

        # check energy/rms
        rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        if rms < SILENCE_THRESHOLD:
            # silent chunk -> skip
            continue

        print(f"   🔊 Audio detected (level: {rms:.4f})")

        # transcribe
        try:
            result = model.transcribe(audio, language="en", fp16=False)
            text = result.get("text", "").strip()
            
            if not text:
                continue
            
            print(f"   📝 Transcribed: '{text}'")

            best_spell, score = token_best_match(text)
            print(f"   🔮 Match -> '{best_spell}' (score: {score:.2f})")

            if best_spell == expected_spell and score >= CONFIDENCE_THRESHOLD:
                print(f"   ✅ SUCCESS! Detected '{expected_spell}'")
                found = True
                elapsed = time.time() - start
                break
            elif best_spell and score >= CONFIDENCE_THRESHOLD:
                print(f"   ❌ Wrong spell (detected '{best_spell}', expected '{expected_spell}')")
            # else continue listening until timeout or success
        except Exception as e:
            print(f"   ⚠️ Transcription error: {e}")
            # continue and retry
            continue

    # finalize result
    with detection_lock:
        current_detection = {
            "running": False,
            "success": bool(found),
            "expected_spell": expected_spell,
            "elapsed_time": round(elapsed if found else (time.time() - start), 2),
            "timestamp": time.time()
        }
    detection_running.clear()
    
    if not found:
        print(f"   ⏱️ TIMEOUT! Failed to detect '{expected_spell}' in {timeout_seconds}s")

# -------- API Endpoints --------
@app.route('/detect', methods=['POST'])
def detect_spell():
    """
    Request spell detection
    JSON body: {
        "spell": "lumos",
        "timeout": 10,
        "sync": false  # optional: if true, waits for result before returning
    }
    Returns: {"success": true/false, "expected_spell": "...", "elapsed_time": ...}
    """
    global current_detection
    
    data = request.json or {}
    spell = (data.get('spell') or '').lower()
    timeout = float(data.get('timeout') or 10.0)
    wait_sync = bool(data.get('sync') or False)  # if true: block until done and return result
    
    # Validate spell
    if spell not in SPELLS:
        return jsonify({
            "error": f"Unknown spell '{spell}'. Valid spells: {', '.join(SPELLS)}"
        }), 400
    
    # prevent overlapping requests
    if detection_running.is_set():
        return jsonify({
            "error": "Detection already in progress"
        }), 409
    
    with detection_lock:
        # mark as running
        current_detection = {"running": True, "started_at": time.time(), "expected_spell": spell}
    
    thread = threading.Thread(target=detect_spell_worker, args=(spell, timeout), daemon=True)
    thread.start()
    
    if wait_sync:
        # Wait for the worker to finish (but no longer than timeout + small buffer)
        thread.join(timeout + 1.0)
        with detection_lock:
            return jsonify(current_detection), 200
    else:
        return jsonify({
            "message": f"Started listening for '{spell}' (timeout: {timeout}s)",
            "spell": spell,
            "timeout": timeout
        }), 200

@app.route('/status', methods=['GET'])
def get_status():
    """
    Check detection status
    Returns: {"running": true/false, "result": {...}}
    """
    with detection_lock:
        if current_detection is None:
            return jsonify({"running": False, "result": None})
        
        if current_detection.get('running'):
            return jsonify({"running": True, "result": None})
        
        # Detection complete
        result = current_detection.copy()
        return jsonify({"running": False, "result": result})

@app.route('/reset', methods=['POST'])
def reset():
    """Reset detection state"""
    global current_detection
    with detection_lock:
        current_detection = None
    return jsonify({"message": "Reset complete"})

@app.route('/spells', methods=['GET'])
def list_spells():
    """Get list of available spells"""
    return jsonify({"spells": SPELLS})

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "model": WHISPER_MODEL})

# -------- Main --------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎮 SPELL DETECTOR SERVER FOR UNITY")
    print("="*60)
    print(f"Available spells: {', '.join(SPELLS)}")
    print("\n📡 Server starting on http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST /detect       - Start spell detection")
    print("  GET  /status       - Check detection status")
    print("  POST /reset        - Reset detection state")
    print("  GET  /spells       - List available spells")
    print("  GET  /health       - Health check")
    print("\n💡 Example Unity request:")
    print('  POST http://localhost:5000/detect')
    print('  Body: {"spell": "lumos", "timeout": 10}')
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
    finally:
        print("✅ Server stopped")
