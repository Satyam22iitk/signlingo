#!/usr/bin/env python3
"""
webcam_infer_ui_hybrid_final_adjusted.py

Same hybrid script as before, but improved bottom-suggestions rendering:

 - More aggressive downscaling (min_scale lowered).
 - If suggestions still don't fit, split into up to 2 rows (balanced).
 - Ellipsize items if still necessary.
 - NEW: When font scale gets small, automatically reduce stroke thickness
        so tiny text stays crisp (less "bold").

Other behavior preserved: Groq worker, non-blocking TTS, arrow nav,
accept with Enter/1..5, press 'a' for TTS, 'n' to clear and say "New sentence".
"""

import os
import json
import time
import datetime
import threading
import queue
import traceback
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from bisect import bisect_left
from typing import List

# ---------------- ROBUST TTS worker (create engine per utterance) ----------------
tts_queue = queue.Queue(maxsize=8)   # bounded queue to avoid huge backlog
tts_thread = None
tts_thread_lock = threading.Lock()
HAVE_TTS = False
try:
    import pyttsx3
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False

def tts_worker_fn():
    while True:
        try:
            txt = tts_queue.get()
        except Exception as e:
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [TTS ERROR] queue.get() failed: {e}")
            time.sleep(0.2)
            continue

        if txt is None:
            try:
                tts_queue.task_done()
            except Exception:
                pass
            break

        if not txt:
            try:
                tts_queue.task_done()
            except Exception:
                pass
            continue

        ts_start = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"[{ts_start}] [TTS START] len={len(txt)} preview='{txt[:120]}'")
        try:
            engine = None
            try:
                engine = pyttsx3.init()
                engine.say(txt)
                engine.runAndWait()
            finally:
                try:
                    if engine is not None and hasattr(engine, "stop"):
                        engine.stop()
                except Exception:
                    pass
            ts_done = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts_done}] [TTS DONE]")
        except Exception as e:
            ts_err = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts_err}] [TTS ERROR] speaking failed: {e}")
            traceback.print_exc()
            time.sleep(0.2)
        finally:
            try:
                tts_queue.task_done()
            except Exception:
                pass

def start_tts_thread_if_needed():
    global tts_thread
    if not HAVE_TTS:
        return
    with tts_thread_lock:
        if tts_thread is None or not tts_thread.is_alive():
            tts_thread = threading.Thread(target=tts_worker_fn, daemon=True)
            tts_thread.start()
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [TTS THREAD] started")

def enqueue_tts(text: str):
    if not HAVE_TTS:
        print("pyttsx3 not installed; install with: pip install pyttsx3")
        return
    txt = (text or "").strip()
    if not txt:
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"[{ts}] [TTS SKIP] empty text, nothing to speak")
        return
    start_tts_thread_if_needed()
    try:
        tts_queue.put_nowait(txt)
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"[{ts}] [TTS ENQ] queued text (len={len(txt)})")
    except queue.Full:
        try:
            _ = tts_queue.get_nowait()
            tts_queue.task_done()
        except Exception:
            pass
        try:
            tts_queue.put_nowait(txt)
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [TTS ENQ] queue was full — replaced oldest with new text")
        except Exception as e:
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [TTS ERROR] failed to enqueue after purge: {e}")

# ---------------- GROQ client import ----------------
try:
    from groq import Groq
except Exception as e:
    raise RuntimeError("Missing 'groq' package. Install with: pip install groq") from e

# optional libs (wordfreq / symspellpy)
try:
    from wordfreq import top_n_list, zipf_frequency
    HAVE_WORDFREQ = True
except Exception:
    HAVE_WORDFREQ = False

try:
    from symspellpy import SymSpell, Verbosity
    HAVE_SYMSPELL = True
except Exception:
    HAVE_SYMSPELL = False

# ---------------- CONFIG ----------------
MODEL_PATH = "logs/sign_model_2.h5"
LABELS_PATH = "logs/labels.json"
BACKBONE = "inception"

ROI = (300, 100, 600, 400)
PRED_INTERVAL_FRAMES = 5
STABILITY_SECONDS = 0.7
ACCEPT_COOLDOWN = 0.6

SUGGESTION_TOPK = 5
WORDLIST_SIZE = 50000
SYMSPELL_DICT_PATH = "frequency_dictionary_en_82_765.txt"

# --- GROQ ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY environment variable with your Groq API key.")

GROQ_MODEL = "llama-3.1-8b-instant"
SUGGESTION_TIMEOUT = 10.0
# --- END GROQ ---

WINDOW_NAME = "Webcam - Press ESC to quit"

# ---------------- letter model preprocess ----------------
if BACKBONE.lower().startswith("inception"):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    IMG_SIZE = (299, 299)
elif BACKBONE.lower().startswith("efficient"):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    IMG_SIZE = (224, 224)
else:
    raise ValueError("Unsupported BACKBONE; use 'inception' or 'efficientnet'")

print("Loading letter model:", MODEL_PATH)
letter_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loaded letter model.")
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
if isinstance(labels, dict):
    labels_list = [labels.get(str(i)) if str(i) in labels else labels.get(i) for i in range(len(labels))]
else:
    labels_list = list(labels)
print("Labels:", labels_list)

# --- GROQ client ---
client = Groq(api_key=GROQ_API_KEY)

# ---------------- build offline wordlist & symspell ----------------
WORDLIST: List[str] = []
SYMSPELL = None

def build_wordlist_and_symspell():
    global WORDLIST, SYMSPELL
    if HAVE_WORDFREQ:
        try:
            WORDLIST = top_n_list("en", n=WORDLIST_SIZE)
            seen = set()
            wl = []
            for w in WORDLIST:
                w2 = w.lower()
                if w2 not in seen:
                    seen.add(w2)
                    wl.append(w2)
            WORDLIST = wl
        except Exception:
            WORDLIST = []
    else:
        WORDLIST = ["hello","help","have","how","house","happy","here","hope","world","word","yes","you","your","please","thank","people","picture","press","play","person"]

    if HAVE_SYMSPELL:
        try:
            sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            p = Path(SYMSPELL_DICT_PATH)
            if p.exists():
                sym.load_dictionary(str(p), term_index=0, count_index=1)
            else:
                for w in WORDLIST:
                    try:
                        sym.create_dictionary_entry(w, 1)
                    except Exception:
                        pass
            SYMSPELL = sym
        except Exception:
            SYMSPELL = None
    else:
        SYMSPELL = None

build_wordlist_and_symspell()
WORDLIST_SORTED = sorted(set(WORDLIST))

def prefix_search(prefix: str, max_results: int = 50):
    if not prefix:
        return []
    p = prefix.lower()
    i = bisect_left(WORDLIST_SORTED, p)
    res = []
    n = len(WORDLIST_SORTED)
    while i < n and len(res) < max_results:
        w = WORDLIST_SORTED[i]
        if w.startswith(p):
            res.append(w)
        else:
            break
        i += 1
    return res

def rank_by_zipf(words, top_k=SUGGESTION_TOPK):
    if not HAVE_WORDFREQ:
        return words[:top_k]
    scored = []
    for w in words:
        try:
            score = zipf_frequency(w, "en")
        except Exception:
            score = 0.0
        scored.append((w, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w,_ in scored[:top_k]]

def symspell_lookup(prefix: str, max_results=SUGGESTION_TOPK):
    if SYMSPELL is None:
        return []
    try:
        suggestions = SYMSPELL.lookup(prefix, Verbosity.TOP, max_edit_distance=2, include_unknown=True)
        terms = [s.term for s in suggestions][:max_results]
        return terms
    except Exception:
        return []

def get_suggestions_for_prefix(prefix: str, top_k=SUGGESTION_TOPK):
    if not prefix:
        return []
    hits = prefix_search(prefix, max_results=200)
    if hits:
        return [w.upper() for w in rank_by_zipf(hits, top_k=top_k)]
    fuzzy = symspell_lookup(prefix, max_results=top_k)
    if fuzzy:
        return [w.upper() for w in fuzzy]
    small = [w.upper() for w in WORDLIST if w.startswith(prefix.lower())][:top_k]
    return small

# --- GROQ helpers (single continuation) ---
def groq_get_single_continuation(prompt: str,
                                max_new_tokens: int = 20,
                                timeout: float = SUGGESTION_TIMEOUT) -> str:
    if not prompt:
        return ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides short, predictive continuations to a sentence. Return only the most likely next words."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,  # single continuation
            temperature=0.0,
            top_p=0.95,
            timeout=timeout
        )
        if getattr(resp, "choices", None):
            try:
                text = resp.choices[0].message.content.strip()
            except Exception:
                text = str(resp.choices[0])
            if text.lower().startswith(prompt.lower()):
                text = text[len(prompt):].lstrip()
            return text
    except Exception as e:
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"[{ts}] [ERROR] Groq API request failed: {e}")
        return ""
    return ""

def get_suggestions_for_context_sync(prompt: str, top_k=SUGGESTION_TOPK) -> List[str]:
    if not prompt.strip():
        return []
    continuation = groq_get_single_continuation(prompt, max_new_tokens=24)
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f"[{ts}] [GROQ RAW] continuation='{continuation}'")
    if not continuation:
        return []
    words = continuation.split()
    cleaned_words = []
    for w in words:
        w_clean = w.strip().strip('.,;:!?()[]\"\'')
        if w_clean:
            cleaned_words.append(w_clean.upper())
    cumulative = []
    cur = ""
    for w in cleaned_words:
        cur = (cur + " " + w).strip()
        cumulative.append(cur)
        if len(cumulative) >= top_k:
            break
    return cumulative[:top_k]

# helpers
def safe_backspace(s: str) -> str:
    return s[:-1] if s else s

def wrap_text_to_lines(text: str, font, font_scale, thickness, max_width: int):
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def apply_chosen_to_sequence(chosen: str):
    global sequence
    if not chosen:
        return
    if sequence.strip() == "":
        sequence = chosen + " "
    elif sequence.endswith(" "):
        sequence = sequence + chosen + " "
    else:
        parts = sequence.rstrip().rsplit(" ", 1)
        if len(parts) == 1:
            sequence = chosen + " "
        else:
            sequence = parts[0] + " " + chosen + " "

# Groq worker
groq_queue = queue.Queue(maxsize=1)
groq_lock = threading.Lock()

def groq_worker():
    while True:
        prompt = groq_queue.get()
        if prompt is None:
            groq_queue.task_done()
            break
        ts_req = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"[{ts_req}] [GROQ REQ] prompt(len={len(prompt)}): '{prompt[-200:]}'")
        try:
            results = get_suggestions_for_context_sync(prompt, top_k=SUGGESTION_TOPK)
            ts_resp = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts_resp}] [GROQ RESP] suggestions={results}")
            with groq_lock:
                global suggestions, selection_index, current_prefix
                suggestions = results.copy() if results else []
                selection_index = 0
                current_prefix = ""
        except Exception as e:
            ts_err = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts_err}] [ERROR] Groq worker error: {e}")
        finally:
            groq_queue.task_done()

worker_thread = threading.Thread(target=groq_worker, daemon=True)
worker_thread.start()

def enqueue_groq_prompt(prompt: str):
    if not prompt:
        return
    try:
        groq_queue.put_nowait(prompt)
    except queue.Full:
        try:
            _ = groq_queue.get_nowait()
            groq_queue.task_done()
        except Exception:
            pass
        try:
            groq_queue.put_nowait(prompt)
        except Exception as e:
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [ERROR] Failed to enqueue Groq prompt: {e}")
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f"[{ts}] [GROQ ENQ] enqueued prompt (len={len(prompt)} chars)")

# Main loop & UI state
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except Exception:
    pass

frame_count = 0
sequence = ""
top3 = []

last_raw_pred = None
stable_candidate = None
stable_since = 0.0
last_accepted_pred = None
last_accept_time = 0.0

current_prefix = ""
suggestions: List[str] = []
selection_index = 0

NAV_LEFT_KEYS = {81, 2424832, ord('a'), ord('h')}
NAV_RIGHT_KEYS = {83, 2555904, ord('d'), ord('l')}

try:
    while True:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        x1, y1, x2, y2 = ROI
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)

        frame_count += 1
        if frame_count % PRED_INTERVAL_FRAMES == 0:
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            except Exception:
                roi_rgb = roi
            if roi_rgb.shape[0] == 0 or roi_rgb.shape[1] == 0:
                continue
            resized = cv2.resize(roi_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
            x = resized.astype("float32")
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)

            preds = letter_model.predict(x, verbose=0)
            if preds.ndim == 2:
                probs = preds[0]
            else:
                probs = np.array(preds).flatten()

            top_idx = probs.argsort()[-3:][::-1]
            top3 = [(labels_list[i] if i < len(labels_list) else str(i), float(probs[i])) for i in top_idx]
            pred_idx = int(np.argmax(probs))
            pred_label = labels_list[pred_idx] if pred_idx < len(labels_list) else str(pred_idx)
            now = time.time()

            if pred_label != last_raw_pred:
                stable_candidate = pred_label
                stable_since = now
                last_raw_pred = pred_label
            else:
                if stable_candidate is None:
                    stable_candidate = pred_label
                    stable_since = now
                if stable_candidate is not None and (now - stable_since) >= STABILITY_SECONDS:
                    same_as_last = (last_accepted_pred == stable_candidate)
                    cooldown_ok = (now - last_accept_time) >= ACCEPT_COOLDOWN
                    if (not same_as_last) or cooldown_ok:
                        lbl = stable_candidate.lower()
                        if lbl == "space":
                            sequence += " "
                            ts_accept = datetime.datetime.now().isoformat(timespec='seconds')
                            print(f"[{ts_accept}] [ACCEPT] 'space' -> sequence='{sequence}'")
                            enqueue_groq_prompt(sequence)
                            current_prefix = ""
                            with groq_lock:
                                selection_index = 0
                        elif lbl in ("del","delete","backspace"):
                            sequence = safe_backspace(sequence)
                            ts_accept = datetime.datetime.now().isoformat(timespec='seconds')
                            print(f"[{ts_accept}] [ACCEPT] 'delete' -> sequence='{sequence}'")
                            parts = sequence.rstrip().split(" ")
                            current_prefix = parts[-1] if parts else ""
                            with groq_lock:
                                suggestions = get_suggestions_for_prefix(current_prefix)
                                selection_index = 0
                        elif lbl in ("nothing","blank"):
                            pass
                        else:
                            sequence += stable_candidate
                            ts_accept = datetime.datetime.now().isoformat(timespec='seconds')
                            print(f"[{ts_accept}] [ACCEPT] '{stable_candidate}' -> sequence='{sequence}'")
                            parts = sequence.rstrip().split(" ")
                            current_prefix = parts[-1] if parts else ""
                            with groq_lock:
                                suggestions = get_suggestions_for_prefix(current_prefix)
                                selection_index = 0

                        last_accepted_pred = stable_candidate
                        last_accept_time = now
                        stable_candidate = None
                        stable_since = 0.0

        # top3 (left)
        if top3:
            y_off = 60
            for ch, sc in top3:
                cv2.putText(img, f"{ch}: {sc:.2f}", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                y_off += 28

        # -----------------------
        # Bottom suggestions: improved scaling + split into two rows if needed
        # and dynamic stroke thickness when scale is small.
        # -----------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.75    # starting font scale
        min_scale = 0.30     # aggressive lower bound for fitting

        def th_for_scale(sc: float) -> int:
            # Reduce boldness when text is small for clarity
            return 2 if sc >= 0.55 else 1

        with groq_lock:
            display_strings = [f"{i}) {s.upper()}" for i, s in enumerate(suggestions, start=1)]
            sel_index = selection_index

        rows = []  # list of rows, each row is list of (text, width, height)
        scale = base_scale

        if display_strings:
            avail_w = int(w * 0.94)

            def sizes_for_scale(sc, arr):
                local_th = th_for_scale(sc)
                sizes_w = [cv2.getTextSize(s, font, sc, local_th)[0][0] for s in arr]
                total_w = sum(sizes_w) + (20 * (len(sizes_w) - 1)) if sizes_w else 0
                return total_w, sizes_w, local_th

            total_w, sizes, local_th = sizes_for_scale(scale, display_strings)
            while total_w > avail_w and scale > min_scale:
                scale -= 0.05
                total_w, sizes, local_th = sizes_for_scale(scale, display_strings)

            if total_w <= avail_w:
                # single row fits
                row = []
                for i, s in enumerate(display_strings):
                    (tw, th_h), _ = cv2.getTextSize(s, font, scale, th_for_scale(scale))
                    row.append((s, tw, th_h))
                rows = [row]
            else:
                # split into two balanced rows
                n = len(display_strings)
                split_idx = max(1, n // 2)
                first = display_strings[:split_idx]
                second = display_strings[split_idx:]

                w1, s1, local_th = sizes_for_scale(scale, first)
                w2, s2, _ = sizes_for_scale(scale, second)

                if w1 > avail_w or w2 > avail_w:
                    scale2 = scale
                    while (w1 > avail_w or w2 > avail_w) and scale2 > min_scale:
                        scale2 -= 0.05
                        w1, s1, local_th = sizes_for_scale(scale2, first)
                        w2, s2, _ = sizes_for_scale(scale2, second)
                    scale = scale2

                    if w1 > avail_w or w2 > avail_w:
                        # ellipsize per item to target width
                        def ellipsize_list(arr, target_w, sc):
                            out = []
                            th_local = th_for_scale(sc)
                            for s in arr:
                                if cv2.getTextSize(s, font, sc, th_local)[0][0] <= target_w:
                                    out.append(s)
                                    continue
                                for L in range(len(s)-1, 0, -1):
                                    cand = s[:L].rstrip() + "..."
                                    if cv2.getTextSize(cand, font, sc, th_local)[0][0] <= target_w:
                                        out.append(cand)
                                        break
                                else:
                                    out.append("...")
                            return out

                        gap = 20
                        per_item_target1 = max(20, (avail_w - gap*(len(first)-1)) // max(1,len(first)))
                        per_item_target2 = max(20, (avail_w - gap*(len(second)-1)) // max(1,len(second)))
                        first = ellipsize_list(first, per_item_target1, scale)
                        second = ellipsize_list(second, per_item_target2, scale)
                        w1, s1, local_th = sizes_for_scale(scale, first)
                        w2, s2, _ = sizes_for_scale(scale, second)

                row1 = []
                row2 = []
                for i, s in enumerate(first):
                    (tw, th_h), _ = cv2.getTextSize(s, font, scale, th_for_scale(scale))
                    row1.append((s, tw, th_h))
                for i, s in enumerate(second):
                    (tw, th_h), _ = cv2.getTextSize(s, font, scale, th_for_scale(scale))
                    row2.append((s, tw, th_h))
                rows = [row1, row2]

        # draw rows centered at bottom (rows stacked, second above first)
        if rows:
            gap = 20
            base_y = h - 12  # bottom padding
            # draw from bottom-up
            y_cursor = base_y
            # Precompute flattened list for selection index mapping
            flat_list = [it[0] for r in rows for it in r]
            for r in reversed(rows):
                row_total_w = sum(item[1] for item in r) + gap * (len(r)-1) if r else 0
                sx = max(10, (w - row_total_w)//2)
                for display, tw, th_h in r:
                    bx1 = sx - 8
                    by1 = y_cursor - th_h - 8
                    bx2 = sx + tw + 8
                    by2 = y_cursor + 8
                    # th depends on scale:
                    draw_th = th_for_scale(scale)
                    # find flat index to compare selection
                    try:
                        flat_index = flat_list.index(display)
                    except ValueError:
                        flat_index = -1
                    if flat_index == sel_index:
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), (50,120,200), cv2.FILLED)
                        cv2.putText(img, display, (sx, y_cursor), font, scale, (255,255,255), draw_th, cv2.LINE_AA)
                    else:
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0,0,0), cv2.FILLED)
                        cv2.putText(img, display, (sx, y_cursor), font, scale, (200,200,255), draw_th, cv2.LINE_AA)
                    sx += tw + gap
                y_cursor -= (max(item[2] for item in r) + 6)

        # Sequence at top
        seq = sequence
        accepted_part = ""
        partial = ""
        if seq.strip() == "":
            pass
        else:
            parts = seq.rstrip().split(" ")
            if seq.endswith(" "):
                accepted_part = seq.strip()
                partial = ""
            else:
                partial = parts[-1]
                accepted_part = " ".join(parts[:-1])

        small_font = cv2.FONT_HERSHEY_SIMPLEX
        small_scale = 0.7
        small_th = 2
        max_width = int(w * 0.9)
        lines = wrap_text_to_lines(accepted_part, small_font, small_scale, small_th, max_width) if accepted_part else []
        y0 = 10 + 20
        for i, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, small_font, small_scale, small_th)
            tx = (w - tw)//2
            ty = y0 + i*(th+6)
            cv2.rectangle(img, (tx-6, ty-th-6), (tx+tw+6, ty+6), (0,0,0), cv2.FILLED)
            cv2.putText(img, line, (tx, ty), small_font, small_scale, (200,200,200), small_th, cv2.LINE_AA)

        large_font = cv2.FONT_HERSHEY_SIMPLEX
        large_scale = 1.0
        large_th = 3
        partial_display = partial if partial else "_"
        (pw, ph), _ = cv2.getTextSize(partial_display, large_font, large_scale, large_th)
        px = (w - pw)//2
        py = y0 + len(lines)*(int(ph+6)) + ph + 6
        cv2.rectangle(img, (px-12, py-ph-12), (px+pw+12, py+12), (0,0,0), cv2.FILLED)
        cv2.putText(img, partial_display, (px, py), large_font, large_scale, (255,255,255), large_th, cv2.LINE_AA)

        # Live raw pred at top-right of ROI
        if last_raw_pred:
            live_text = f"Live: {last_raw_pred}"
            nf, nscale, nth = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (tw_live, th_live), _ = cv2.getTextSize(live_text, nf, nscale, nth)
            pad = 8
            lx = x2 - tw_live - pad
            ly = y1 + th_live + pad
            lx = max(0, min(lx, w - tw_live - 1))
            ly = max(th_live + 1, min(ly, h - 1))
            cv2.rectangle(img, (lx - 6, ly - th_live - 6), (lx + tw_live + 6, ly + 6), (0,0,0), cv2.FILLED)
            cv2.putText(img, live_text, (lx, ly), nf, nscale, (0,255,255), nth, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)

        if key == -1:
            continue

        # 'a' for TTS
        if key == ord('a'):
            txt = sequence.strip()
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            if txt:
                print(f"[{ts}] [TTS REQ] speaking: '{txt}'")
                if HAVE_TTS:
                    enqueue_tts(txt)
                else:
                    print("pyttsx3 not installed; install with: pip install pyttsx3")
            else:
                print(f"[{ts}] [TTS REQ] nothing to speak (sequence empty)")

        # 'n' to clear and say "New sentence"
        if key in (ord('n'), ord('N')):
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts}] [ACTION] 'N' pressed -> clearing sequence and speaking 'New sentence'")
            sequence = ""
            with groq_lock:
                suggestions = []
                current_prefix = ""
                selection_index = 0
            if HAVE_TTS:
                enqueue_tts("New sentence")
            else:
                print("pyttsx3 not installed; install with: pip install pyttsx3")

        # navigation
        if key in NAV_LEFT_KEYS:
            with groq_lock:
                if suggestions:
                    selection_index = max(0, selection_index - 1)
        elif key in NAV_RIGHT_KEYS:
            with groq_lock:
                if suggestions:
                    selection_index = min(len(suggestions)-1, selection_index + 1)

        # accept number keys
        if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            sel = int(chr(key)) - 1
            with groq_lock:
                if sel < len(suggestions):
                    chosen = suggestions[sel]
                    ts_accept = datetime.datetime.now().isoformat(timespec='seconds')
                    print(f"[{ts_accept}] [ACCEPT] chosen='{chosen}' (via key {sel+1})")
                    apply_chosen_to_sequence(chosen)
                    enqueue_groq_prompt(sequence)

        # accept enter
        if key in (13, 10):
            with groq_lock:
                if suggestions and 0 <= selection_index < len(suggestions):
                    chosen = suggestions[selection_index]
                    ts_accept = datetime.datetime.now().isoformat(timespec='seconds')
                    print(f"[{ts_accept}] [ACCEPT] chosen='{chosen}' (via Enter)")
                    apply_chosen_to_sequence(chosen)
                    enqueue_groq_prompt(sequence)

        # delete/backspace
        if key in (8, ord('x')):
            sequence = safe_backspace(sequence)
            ts_del = datetime.datetime.now().isoformat(timespec='seconds')
            print(f"[{ts_del}] [ACTION] backspace -> sequence='{sequence}'")
            parts = sequence.rstrip().split(" ")
            current_prefix = parts[-1] if parts else ""
            with groq_lock:
                suggestions = get_suggestions_for_prefix(current_prefix)
                selection_index = 0

        # ESC quit
        if key == 27:
            break

finally:
    try:
        groq_queue.put_nowait(None)
    except Exception:
        pass
    worker_thread.join(timeout=1.0)

    if HAVE_TTS:
        try:
            tts_queue.put_nowait(None)
        except Exception:
            pass
        with tts_thread_lock:
            if tts_thread is not None:
                tts_thread.join(timeout=1.0)

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped. Final sequence:", sequence)
