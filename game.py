import cv2
import mediapipe as mp
import math
import numpy as np
import threading
import time
import os
from dotenv import load_dotenv

# ─── LangChain + Anthropic imports ───────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ─────────────────────────────────────────────────────────────────────────────
#  Enemy definitions
# ─────────────────────────────────────────────────────────────────────────────

ENEMIES = [
    {"name": "RADITZ",  "hp": 300,  "color": (0,   80,  200), "emoji": "R"},
    {"name": "VEGETA",  "hp": 500,  "color": (100, 0,   200), "emoji": "V"},
    {"name": "FRIEZA",  "hp": 700,  "color": (180, 0,   180), "emoji": "F"},
    {"name": "CELL",    "hp": 900,  "color": (0,   180, 0  ), "emoji": "C"},
    {"name": "BUU",     "hp": 1200, "color": (0,   100, 255), "emoji": "B"},
]

# ─────────────────────────────────────────────────────────────────────────────
#  RAG — Knowledge Base + Vector Store
# ─────────────────────────────────────────────────────────────────────────────

DBZ_MOVES = [
    {
        "move": "Kamehameha",
        "user": "Goku",
        "description": "Arms extended forward, hands cupped together, firing a blue energy beam.",
        "pose_cues": "Both wrists close together, arms fully extended forward, legs shoulder-width apart.",
        "power_level": 9000,
        "scoring_criteria": "Arms must be fully straight, wrists touching, body leaning forward slightly.",
    },
    {
        "move": "Final Flash",
        "user": "Vegeta",
        "description": "Arms spread wide then thrust forward, releasing a massive golden beam.",
        "pose_cues": "Arms wide first then pushed together forward, feet planted firmly.",
        "power_level": 10000,
        "scoring_criteria": "Arms fully extended, palms facing forward, strong wide stance.",
    },
    {
        "move": "Spirit Bomb",
        "user": "Goku",
        "description": "One or both arms raised above the head, gathering energy from all living things.",
        "pose_cues": "Arms raised high above head, palms facing up toward the sky.",
        "power_level": 15000,
        "scoring_criteria": "Arms fully raised, palms up, head tilted back.",
    },
    {
        "move": "Special Beam Cannon",
        "user": "Piccolo",
        "description": "Two fingers pointed forward from the forehead, charging a drilling beam.",
        "pose_cues": "Two fingers extended from forehead pointing outward, other hand gripping the arm.",
        "power_level": 8000,
        "scoring_criteria": "Fingers pointed, arm extended, focused stance.",
    },
    {
        "move": "Galick Gun",
        "user": "Vegeta",
        "description": "Hands brought to the side of the body then thrust forward releasing purple energy.",
        "pose_cues": "Hands brought to hip/side, then both palms thrust forward together.",
        "power_level": 9500,
        "scoring_criteria": "Hands together at side first, then thrust forward at chest height.",
    },
]

CHROMA_DIR = "./chroma_db"


def build_vector_store():
    """Build ChromaDB vector store from DBZ moves."""
    print("🔨 Building vector store...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = []
    for move in DBZ_MOVES:
        content = (
            f"Move: {move['move']}\n"
            f"User: {move['user']}\n"
            f"Description: {move['description']}\n"
            f"Pose Cues: {move['pose_cues']}\n"
            f"Scoring Criteria: {move['scoring_criteria']}\n"
            f"Power Level: {move['power_level']}\n"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={"move": move["move"], "power_level": move["power_level"]},
            )
        )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("✅ Vector store built!")
    return vectorstore


def load_vector_store():
    """Load existing vector store or build a new one."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return build_vector_store()


def retrieve_move_info(query: str, vectorstore, k: int = 1) -> str:
    """Retrieve the most relevant DBZ move for a given pose query."""
    results = vectorstore.similarity_search(query, k=k)
    return results[0].page_content if results else "No matching move found."


# ─────────────────────────────────────────────────────────────────────────────
#  LLM — Claude Judge via LangChain
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = PromptTemplate(
    input_variables=["move_info", "pose_state", "arm_angle", "wrist_distance"],
    template="""
You are a Dragon Ball Z battle commentator and pose judge.

The player is attempting a DBZ move. Here is the move reference:
{move_info}

Current player pose data:
- Pose State: {pose_state}
- Arm Angle: {arm_angle} degrees  (180 = fully extended, 90 = bent)
- Wrist Distance: {wrist_distance}  (lower = hands closer together)

Judge the pose based on the scoring criteria above.
1. Score out of 100
2. One short dramatic DBZ-style comment (max 10 words)
3. Rating: PERFECT (>80), GOOD (50-80), or WEAK (<50)

Reply ONLY in this exact format — no extra text:
SCORE: [number]
COMMENT: [short dramatic comment]
RATING: [PERFECT/GOOD/WEAK]
""",
)


def build_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file!")
    return ChatAnthropic(
        model="claude-opus-4-6",
        anthropic_api_key=api_key,
        max_tokens=150,
    )


def parse_llm_response(text: str) -> dict:
    result = {"score": 50, "comment": "KI DETECTED!", "rating": "GOOD"}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("COMMENT:"):
            result["comment"] = line.split(":", 1)[1].strip()
        elif line.startswith("RATING:"):
            result["rating"] = line.split(":", 1)[1].strip()
    return result


def judge_pose(llm, move_info: str, pose_state: str,
               arm_angle: float, wrist_distance: float) -> dict:
    """Ask Claude to judge the player's pose."""
    prompt = JUDGE_PROMPT.format(
        move_info=move_info,
        pose_state=pose_state,
        arm_angle=round(arm_angle, 1),
        wrist_distance=round(wrist_distance, 3),
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_llm_response(response.content)
    except Exception as e:
        print(f"⚠️  LLM error: {e}")
        return {"score": 50, "comment": "POWER LEVEL UNKNOWN!", "rating": "GOOD"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_angle_3d(a, b, c) -> float:
    v1 = [a.x - b.x, a.y - b.y, a.z - b.z]
    v2 = [c.x - b.x, c.y - b.y, c.z - b.z]
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(sum(x**2 for x in v1))
    mag2 = math.sqrt(sum(x**2 for x in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def get_looped_frame(cap):
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame if ret else None


def draw_hud(image, current_state, score_data, total_score, combo,
             enemy, enemy_hp, enemy_hp_max, flash_timer):
    """Draw all HUD elements on the frame."""
    h, w = image.shape[:2]

    # ── Accent line at top ────────────────────────────────────────────────────
    cv2.line(image, (0, 5), (w, 5), (0, 80, 200), 2)

    # ── Enemy HP bar (top centre) ─────────────────────────────────────────────
    bar_w   = int(w * 0.55)
    bar_h   = 28
    bar_x   = (w - bar_w) // 2
    bar_y   = 14
    hp_ratio = max(0.0, enemy_hp / enemy_hp_max)

    # background
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    # HP fill — green → orange → red
    if hp_ratio > 0.5:
        bar_color = (0, 200, 50)
    elif hp_ratio > 0.25:
        bar_color = (0, 140, 255)
    else:
        bar_color = (0, 0, 255)

    fill_w = int(bar_w * hp_ratio)
    if fill_w > 0:
        cv2.rectangle(image, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # border
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)

    # enemy name + HP numbers
    enemy_color = enemy["color"]
    cv2.putText(image, f"{enemy['name']}  {int(enemy_hp)}/{enemy_hp_max}",
                (bar_x + 6, bar_y + 21),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    # ── Damage flash (red tint when hit) ─────────────────────────────────────
    if flash_timer > 0:
        overlay = image.copy()
        cv2.rectangle(overlay, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

    # ── State banner (top right) ──────────────────────────────────────────────
    state_colors = {
        "IDLE":     (180, 180, 180),
        "CHARGING": (0, 200, 255),
        "FIRING":   (0, 80, 255),
    }
    s_color = state_colors.get(current_state, (255, 255, 255))
    cv2.putText(image, current_state, (w - 190, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, s_color, 2)

    # ── Score / rating (top left) ─────────────────────────────────────────────
    rating = score_data.get("rating", "IDLE")
    rating_color = (0, 255, 80)
    if rating == "GOOD":
        rating_color = (0, 165, 255)
    elif rating == "WEAK":
        rating_color = (0, 0, 255)

    score_val = score_data.get("score", 0)
    comment   = score_data.get("comment", "")

    if rating != "IDLE":
        cv2.putText(image, f"SCORE: {score_val}/100",
                    (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, rating_color, 3)
        cv2.putText(image, comment,
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
        cv2.putText(image, rating,
                    (20, 135), cv2.FONT_HERSHEY_DUPLEX, 0.95, rating_color, 2)

    # ── Total score & combo (bottom left) ────────────────────────────────────
    cv2.putText(image, f"TOTAL: {total_score}",
                (20, h - 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    if combo > 1:
        cv2.putText(image, f"COMBO x{combo}",
                    (20, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2)

    return image


def draw_result_screen(image, won: bool, enemy_name: str, total_score: int):
    """Full-screen win / lose overlay."""
    overlay = image.copy()
    color   = (0, 180, 0) if won else (0, 0, 200)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), color, -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    h, w = image.shape[:2]
    if won:
        lines = [
            ("YOU WIN!", (0, 255, 100), 2.2, 3),
            (f"{enemy_name} IS DEFEATED!", (255, 255, 0), 0.9, 2),
            (f"FINAL SCORE: {total_score}", (255, 255, 255), 1.0, 2),
            ("Press Q to quit / R to restart", (180, 180, 180), 0.65, 1),
        ]
    else:
        lines = [
            ("YOU LOSE!", (0, 80, 255), 2.2, 3),
            (f"{enemy_name} WINS!", (255, 100, 0), 0.9, 2),
            (f"FINAL SCORE: {total_score}", (255, 255, 255), 1.0, 2),
            ("Press Q to quit / R to restart", (180, 180, 180), 0.65, 1),
        ]

    y = h // 2 - 80
    for text, col, scale, thickness in lines:
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0], None
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0][0]
        cv2.putText(image, text, ((w - tw) // 2, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, thickness)
        y += int(scale * 60 + 20)

    return image


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("🐉 Loading RAG vector store...")
    vectorstore = load_vector_store()

    print("🤖 Initialising Claude judge...")
    llm = build_llm()

    # ── Video captures ────────────────────────────────────────────────────────
    cap        = cv2.VideoCapture(0)
    cap_energy = cv2.VideoCapture("assets/energy.mp4")
    cap_kame   = cv2.VideoCapture("assets/kamehameha.mp4")

    # ── MediaPipe pose landmarker ─────────────────────────────────────────────
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # ── Game state ────────────────────────────────────────────────────────────
    prev_state    = "IDLE"
    frame_idx     = 0
    smooth_hx     = -1
    smooth_hy     = -1
    smooth_factor = 0.5
    idle_frames   = 0

    score_data     = {"score": 0, "comment": "STRIKE A POSE!", "rating": "IDLE"}
    total_score    = 0
    combo          = 0
    last_fire_time = 0.0
    judging        = False

    last_arm_angle  = 90.0
    last_wrist_dist = 1.0

    # ── Enemy state ───────────────────────────────────────────────────────────
    enemy_idx   = 0
    enemy       = dict(ENEMIES[enemy_idx])
    enemy_hp    = float(enemy["hp"])
    enemy_hp_max = float(enemy["hp"])
    flash_timer = 0        # frames to show damage flash
    game_over   = False
    game_won    = False

    with landmarker as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
            height, width, _ = image.shape

            # ── Pose detection ────────────────────────────────────────────────
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            frame_idx += 1
            frame_ts   = frame_idx * 33
            result     = pose.detect_for_video(mp_image, frame_ts)

            current_state = "IDLE"
            hx, hy        = -1, -1
            left_angle    = 0.0
            right_angle   = 0.0
            wrist_dist    = 1.0

            if result.pose_landmarks:
                for landmarks in result.pose_landmarks:
                    ls  = landmarks[11]   # left shoulder
                    rs  = landmarks[12]   # right shoulder
                    le  = landmarks[13]   # left elbow
                    re  = landmarks[14]   # right elbow
                    lw  = landmarks[15]   # left wrist
                    rw  = landmarks[16]   # right wrist

                    if lw.visibility > 0.4 or rw.visibility > 0.4:
                        wrist_dist  = math.hypot(lw.x - rw.x, lw.y - rw.y)
                        dist_thresh = 0.50 if prev_state == "FIRING" else 0.35

                        if wrist_dist < dist_thresh:
                            left_angle  = get_angle_3d(ls, le, lw)
                            right_angle = get_angle_3d(rs, re, rw)
                            angle_thresh = 120 if prev_state == "FIRING" else 135

                            if max(left_angle, right_angle) > angle_thresh:
                                current_state = "FIRING"
                                hx = int((lw.x if lw.visibility > rw.visibility else rw.x) * width)
                                hy = int((lw.y if lw.visibility > rw.visibility else rw.y) * height)
                            else:
                                current_state = "CHARGING"
                                hx = int(((lw.x + rw.x) / 2) * width)
                                hy = int(((lw.y + rw.y) / 2) * height)

                        last_arm_angle  = (left_angle + right_angle) / 2
                        last_wrist_dist = wrist_dist

            # ── Idle debounce (5 frames) ──────────────────────────────────────
            if current_state == "IDLE" and prev_state != "IDLE":
                idle_frames += 1
                if idle_frames < 5:
                    current_state = prev_state
            else:
                idle_frames = 0

            # ── Smooth wrist position ─────────────────────────────────────────
            if hx != -1:
                if smooth_hx == -1:
                    smooth_hx, smooth_hy = hx, hy
                else:
                    smooth_hx = int((1 - smooth_factor) * smooth_hx + smooth_factor * hx)
                    smooth_hy = int((1 - smooth_factor) * smooth_hy + smooth_factor * hy)
            elif current_state == "IDLE":
                smooth_hx, smooth_hy = -1, -1

            # ── Reset effect videos on state change ───────────────────────────
            if current_state == "CHARGING" and prev_state != "CHARGING":
                cap_energy.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if current_state == "FIRING" and prev_state != "FIRING":
                cap_kame.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # ── 🔥 Trigger RAG + LLM judge (background thread) ────────────
                now = time.time()
                if not judging and (now - last_fire_time) > 3.0:
                    judging        = True
                    last_fire_time = now

                    # Capture current values for the thread
                    _angle = last_arm_angle
                    _dist  = last_wrist_dist

                    def run_judge(_angle=_angle, _dist=_dist):
                        nonlocal score_data, total_score, combo, judging
                        nonlocal enemy_hp, flash_timer, game_over, game_won
                        nonlocal enemy, enemy_idx, enemy_hp_max
                        move_info = retrieve_move_info(
                            "arms extended forward firing energy beam", vectorstore
                        )
                        result_data = judge_pose(
                            llm, move_info, "FIRING", _angle, _dist
                        )
                        # Combo
                        if result_data["rating"] in ("PERFECT", "GOOD"):
                            combo += 1
                        else:
                            combo = 0
                        bonus = combo * 10 if combo > 1 else 0
                        total_score += result_data["score"] + bonus
                        score_data   = result_data

                        # ── Deal damage to enemy ──────────────────────────────
                        damage = result_data["score"] * (1 + bonus // 50)
                        enemy_hp   -= damage
                        flash_timer = 8   # flash for 8 frames

                        if enemy_hp <= 0:
                            enemy_hp = 0
                            # next enemy or win
                            if enemy_idx + 1 < len(ENEMIES):
                                enemy_idx   += 1
                                enemy        = dict(ENEMIES[enemy_idx])
                                enemy_hp     = float(enemy["hp"])
                                enemy_hp_max = float(enemy["hp"])
                                score_data["comment"] = f"{enemy['name']} APPEARS!"
                            else:
                                game_over = True
                                game_won  = True

                        judging = False

                    threading.Thread(target=run_judge, daemon=True).start()

            prev_state = current_state

            # ── Draw effects ──────────────────────────────────────────────────
            if current_state == "CHARGING" and smooth_hx != -1:
                ef = get_looped_frame(cap_energy)
                if ef is not None:
                    size = int(height * 0.35)
                    ef_r = cv2.resize(ef, (size, size))
                    x1 = max(0, smooth_hx - size // 2)
                    y1 = max(0, smooth_hy - size // 2)
                    x2 = min(width,  smooth_hx + size // 2)
                    y2 = min(height, smooth_hy + size // 2)
                    ex1 = size // 2 - (smooth_hx - x1)
                    ey1 = size // 2 - (smooth_hy - y1)
                    ex2 = size // 2 + (x2 - smooth_hx)
                    ey2 = size // 2 + (y2 - smooth_hy)
                    if x2 > x1 and y2 > y1:
                        image[y1:y2, x1:x2] = cv2.add(
                            image[y1:y2, x1:x2], ef_r[ey1:ey2, ex1:ex2]
                        )

            elif current_state == "FIRING" and smooth_hx != -1:
                ef = get_looped_frame(cap_kame)
                if ef is not None:
                    oh, ow    = ef.shape[:2]
                    beam_h    = int(height * 0.75)
                    beam_w    = int(beam_h * (ow / oh))
                    ef_r      = cv2.resize(ef, (beam_w, beam_h))
                    target_x  = smooth_hx + beam_w // 2
                    target_y  = smooth_hy
                    x1 = max(0, target_x - beam_w // 2)
                    y1 = max(0, target_y - beam_h // 2)
                    x2 = min(width,  target_x + beam_w // 2)
                    y2 = min(height, target_y + beam_h // 2)
                    ex1 = beam_w // 2 - (target_x - x1)
                    ey1 = beam_h // 2 - (target_y - y1)
                    ex2 = beam_w // 2 + (x2 - target_x)
                    ey2 = beam_h // 2 + (y2 - target_y)
                    if x2 > x1 and y2 > y1:
                        image[y1:y2, x1:x2] = cv2.add(
                            image[y1:y2, x1:x2], ef_r[ey1:ey2, ex1:ex2]
                        )

            # ── Tick flash timer ──────────────────────────────────────────────
            if flash_timer > 0:
                flash_timer -= 1

            # ── Draw HUD ──────────────────────────────────────────────────────
            image = draw_hud(image, current_state, score_data,
                             total_score, combo,
                             enemy, enemy_hp, enemy_hp_max, flash_timer)

            # ── Show "Judging..." spinner while waiting for Claude ─────────────
            if judging:
                cv2.putText(image, "Judging...", (width // 2 - 70, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

            # ── Game over overlay ─────────────────────────────────────────────
            if game_over:
                image = draw_result_screen(image, game_won,
                                           enemy["name"], total_score)

            cv2.imshow("Dragon Ball Z — Pose Battle", image)

            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and game_over:
                # Restart
                enemy_idx    = 0
                enemy        = dict(ENEMIES[enemy_idx])
                enemy_hp     = float(enemy["hp"])
                enemy_hp_max = float(enemy["hp"])
                total_score  = 0
                combo        = 0
                score_data   = {"score": 0, "comment": "STRIKE A POSE!", "rating": "IDLE"}
                game_over    = False
                game_won     = False

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cap_energy.release()
    cap_kame.release()
    cv2.destroyAllWindows()
    print(f"\n🏆 Game Over!  Final Score: {total_score}")


if __name__ == "__main__":
    main()
