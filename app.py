
import gradio as gr
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

# Previous email interactions
previous_emails = [
    {
        "email": "My package arrived damaged.",
        "reply": (
            "We apologize for the inconvenience. "
            "We will arrange a replacement after verification."
        )
    },
    {
        "email": "My order has not been delivered yet.",
        "reply": (
            "We are sorry for the delay. "
            "Our team will check your shipment and update you shortly."
        )
    },
    {
        "email": "I want to know the status of my refund.",
        "reply": (
            "Thank you for contacting us. "
            "We will check your refund status and update you shortly."
        )
    }
]


def analyze_email(email, tone):

    if not email or not email.strip():
        return (
            "Please enter a customer email.",
            "",
            ""
        )

    # Find relevant previous interactions
    past_interactions = "\n"

    for item in previous_emails:
        past_interactions += (
            "\nPrevious Email: " + item["email"]
            + "\nPrevious Reply: " + item["reply"]
            + "\n"
        )

    prompt = f"""
You are an intelligent email assistant.

Analyze the customer email below.

CUSTOMER EMAIL:
{email}

PREVIOUS EMAIL INTERACTIONS:
{past_interactions}

REQUIRED OUTPUT:

1. Customer Intention:
Identify what the customer wants.

2. Email Tone:
Identify whether the customer sounds angry,
polite, confused, worried, or neutral.

3. Suggested Reply:
Write a professional response in a {tone} tone.

RULES:
- Use previous interactions when relevant.
- Do not invent order numbers, refunds, or actions.
- Do not claim a replacement or refund is confirmed.
- Keep the reply clear and polite.
- Separate the three sections using headings.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()["response"]

        return result, "Analysis completed successfully.", result

    except requests.exceptions.ConnectionError:
        message = (
            "Cannot connect to Ollama. "
            "Please open the Ollama application "
            "and check that the model is installed."
        )
        return message, "Connection failed.", ""

    except requests.exceptions.Timeout:
        return (
            "The model took too long to respond. Please try again.",
            "Request timed out.",
            ""
        )

    except Exception as error:
        return (
            "Something went wrong: " + str(error),
            "Error occurred.",
            ""
        )


def clear_fields():
    return "", "", "", ""


custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    font-family: Inter, Arial, sans-serif;
}

#main-header {
    text-align: center;
    padding: 30px 15px;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca, #7c3aed, #2563eb);
    color: white;
    margin-bottom: 22px;
}

#main-header h1 {
    font-size: 34px;
    margin-bottom: 10px;
}

#main-header p {
    font-size: 15px;
    opacity: 0.95;
}

.panel {
    border: 1px solid #e0e7ff !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

.primary-btn {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

#footer {
    text-align: center;
    padding: 18px;
    color: #64748b;
    font-size: 13px;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet"
    ),
    css=custom_css,
    title="Intelligent Email Assistant"
) as demo:

    gr.HTML("""
    <div id="main-header">
        <h1>✉️ Intelligent Email Assistant</h1>
        <p>
            AI-powered email analysis and professional
            reply generation using Ollama
        </p>
    </div>
    """)

    gr.Markdown(
        "### Welcome!\n"
        "Enter a customer email to understand its intention, "
        "analyze its tone, and generate a suitable reply."
    )

    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group(elem_classes=["panel"]):

                gr.Markdown("### 📩 Customer Email")

                email_input = gr.Textbox(
                    label="Email Content",
                    placeholder=(
                        "Example: My order has not been delivered. "
                        "Could you please check its status?"
                    ),
                    lines=8
                )

                tone_input = gr.Dropdown(
                    choices=[
                        "Professional",
                        "Friendly",
                        "Formal",
                        "Empathetic"
                    ],
                    value="Professional",
                    label="Preferred Reply Tone"
                )

                generate_btn = gr.Button(
                    "✨ Analyze Email & Generate Reply",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )

                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):

            with gr.Group(elem_classes=["panel"]):

                gr.Markdown("### 🤖 AI Email Analysis")

                analysis_output = gr.Textbox(
                    label="Analysis and Suggested Reply",
                    lines=14,
                    interactive=False,
                    placeholder=(
                        "Your AI-generated analysis and reply "
                        "will appear here."
                    )
                )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                copy_output = gr.Textbox(
                    label="Generated Content to Copy",
                    lines=5,
                    interactive=False
                )

    gr.Markdown("### 🧠 Previous Email Interactions")

    gr.Markdown(
        "The assistant uses these sample interactions "
        "as context when generating replies."
    )

    for item in previous_emails:
        with gr.Accordion(item["email"], open=False):
            gr.Markdown("**Previous reply:**")
            gr.Markdown(item["reply"])

    generate_btn.click(
        fn=analyze_email,
        inputs=[email_input, tone_input],
        outputs=[
            analysis_output,
            status_output,
            copy_output
        ]
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[
            email_input,
            analysis_output,
            status_output,
            copy_output
        ]
    )

    gr.HTML("""
    <div id="footer">
        Experiment 3 | 24CS2506 – Building Applications using GPT-4
        <br>
        Intelligent Email Assistant using Ollama
    </div>
    """)


if __name__ == "__main__":
    demo.launch()