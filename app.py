import os
import re

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """You are a concept checker for systems thinking concepts. Read the learner's explanation carefully.
First, detect which state the explanation is in. Read the full explanation before deciding.
STATE 1 — They wrote at least two sentences attempting to explain the concept, even if vague or using memorized phrases. This is the most common state.
STATE 2 — Their very first sentence says they do not know. They make no attempt to explain at all. Example: 'I honestly have no idea' or 'I am not sure about this one.'
STATE 3 — They STARTED explaining with at least one real sentence, then stopped mid-way or said they got lost. There is a clear starting point followed by a trail-off or admission of being stuck. Example: 'The backend exists because it handles... actually I am not sure how to explain this.'
The key difference: STATE 2 never attempts an explanation. STATE 3 starts one and then stops.
Always return in this exact format and nothing else:

STATE: [1 / 2 / 3]

GAP: [the exact sentence where understanding became a memorized phrase, or the last sentence before they trailed off, or none stated yet]

QUESTION: [one follow-up question that forces them to derive something specific, not just recall a phrase]"""

FOLLOWUP_SYSTEM_PROMPT = """You are an expert teacher reviewing a learner's answer to a follow-up question about a systems thinking concept. The follow-up question was designed to expose a gap in their understanding.
You will receive: the concept, the original gap sentence, the follow-up question, and their answer.
First judge whether they closed the gap. They closed it if their answer contains something specific and derived that was not in their original explanation. They did NOT close it if they just rephrased the same vague statement.
Return in this exact format and nothing else:

CLOSED: yes / no / partially

VERDICT: [one sentence explaining what they got right or what is still missing]

TEACHING: [two to three sentences explaining the concept correctly from first principles, as a good teacher would, without jargon]"""

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONCEPTS = [
    "Why does a backend exist?",
    "What is an API really?",
    "What is a frontend vs backend?",
    "Why do we need a database?",
    "Why can't we handle payments in the browser?",
    "What is an interface really?",
    "What are the different ways to store data?",
]


def parse_check_response(response):
    state = ""
    gap = ""
    question = ""

    state_match = re.search(r"STATE:\s*(.+)", response, re.IGNORECASE)
    gap_match = re.search(r"GAP:\s*(.+?)(?:\n\s*\n|\nQUESTION:|\Z)", response, re.IGNORECASE | re.DOTALL)
    question_match = re.search(r"QUESTION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

    if state_match:
        state = state_match.group(1).strip()
    if gap_match:
        gap = gap_match.group(1).strip()
    if question_match:
        question = question_match.group(1).strip()

    return state, gap, question


def parse_followup_response(response):
    closed = ""
    verdict = ""
    teaching = ""

    closed_match = re.search(r"CLOSED:\s*(.+)", response, re.IGNORECASE)
    verdict_match = re.search(r"VERDICT:\s*(.+?)(?:\n\s*\n|\nTEACHING:|\Z)", response, re.IGNORECASE | re.DOTALL)
    teaching_match = re.search(r"TEACHING:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

    if closed_match:
        closed = closed_match.group(1).strip()
    if verdict_match:
        verdict = verdict_match.group(1).strip()
    if teaching_match:
        teaching = teaching_match.group(1).strip()

    return closed, verdict, teaching


def check_understanding(name, concept, explanation):
    if not name.strip() or not explanation.strip():
        return "Please fill in your name and explanation before checking.", "", ""

    system_prompt = SYSTEM_PROMPT.replace("[concept]", concept)
    user_message = f"Name: {name}\nConcept: {concept}\nExplanation: {explanation}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return parse_check_response(response.choices[0].message.content)
    except Exception:
        return "Something went wrong. Please try again.", "", ""


def submit_answer(concept, gap, question, answer):
    if not answer.strip():
        return "Please type your answer before submitting.", "", ""
    if not gap.strip():
        return "Please check your understanding first before submitting an answer.", "", ""

    user_message = (
        f"Concept: {concept}\n"
        f"Original gap: {gap}\n"
        f"Follow-up question: {question}\n"
        f"Their answer: {answer}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return parse_followup_response(response.choices[0].message.content)
    except Exception:
        return "Something went wrong. Please try again.", "", ""


with gr.Blocks() as demo:
    name = gr.Textbox(label="Your name")
    concept = gr.Dropdown(choices=CONCEPTS, label="Concept")
    explanation = gr.Textbox(label="Your explanation", lines=10)
    check_btn = gr.Button("Check my understanding")
    state_output = gr.Textbox(label="State")
    gap_output = gr.Textbox(label="Gap found")
    question_output = gr.Textbox(label="Follow-up question")
    followup_answer = gr.Textbox(label="Your answer to the follow-up question", lines=5)
    submit_btn = gr.Button("Submit my answer")
    closed_output = gr.Textbox(label="Gap closed")
    verdict_output = gr.Textbox(label="Verdict")
    teaching_output = gr.Textbox(label="What you should know")

    check_btn.click(
        fn=check_understanding,
        inputs=[name, concept, explanation],
        outputs=[state_output, gap_output, question_output],
    )

    submit_btn.click(
        fn=submit_answer,
        inputs=[concept, gap_output, question_output, followup_answer],
        outputs=[closed_output, verdict_output, teaching_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
