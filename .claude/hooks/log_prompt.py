"""UserPromptSubmit hook: appends every user prompt (with timestamp) to
claude_prompt_history.md in the project root, so it's tracked in git and
reviewable for weekly prof updates."""
import json
import sys
import datetime
import os

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "claude_prompt_history.md")

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    prompt = data.get("prompt", "")
    if not prompt:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n---\n**{ts}**\n\n{prompt}\n")

if __name__ == "__main__":
    main()
