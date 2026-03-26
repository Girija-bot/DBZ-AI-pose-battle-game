# llm_judge.py

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Claude
llm = ChatAnthropic(
    model="claude-opus-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)

JUDGE_PROMPT = PromptTemplate(
    input_variables=["move_info", "pose_state", "arm_angle", "wrist_distance"],
    template="""
You are a Dragon Ball Z battle commentator and pose judge.

The player is attempting a DBZ move. Here is the move reference from the knowledge base:
{move_info}

Current player pose data:
- Pose State: {pose_state} (CHARGING or FIRING)
- Arm Angle: {arm_angle} degrees (180 = fully extended, 90 = bent)
- Wrist Distance: {wrist_distance} (lower = hands closer together)

Based on the pose data and the move's scoring criteria:
1. Give a score out of 100
2. Give ONE short DBZ-style comment (max 10 words, dramatic and fun)
3. State if this is PERFECT (>80), GOOD (50-80), or WEAK (<50)

Respond ONLY in this exact format:
SCORE: [number]
COMMENT: [short dramatic comment]
RATING: [PERFECT/GOOD/WEAK]
"""
)

def judge_pose(move_info: str, pose_state: str, arm_angle: float, wrist_distance: float):
    """Send pose data to Claude for scoring and commentary."""
    
    prompt = JUDGE_PROMPT.format(
        move_info=move_info,
        pose_state=pose_state,
        arm_angle=round(arm_angle, 1),
        wrist_distance=round(wrist_distance, 3)
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return parse_response(response.content)
    except Exception as e:
        print(f"LLM error: {e}")
        return {"score": 50, "comment": "POWER LEVEL UNKNOWN!", "rating": "GOOD"}


def parse_response(text: str):
    """Parse Claude's structured response."""
    result = {"score": 50, "comment": "KI DETECTED!", "rating": "GOOD"}
    
    for line in text.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.split(":")[1].strip())
            except:
                pass
        elif line.startswith("COMMENT:"):
            result["comment"] = line.split(":", 1)[1].strip()
        elif line.startswith("RATING:"):
            result["rating"] = line.split(":")[1].strip()
    
    return result