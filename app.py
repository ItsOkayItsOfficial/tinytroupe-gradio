import os
import re
import sys
import json
import tinytroupe
import gradio as gr
import tinytroupe.control as control
from typing import Union

from dotenv import load_dotenv
from tinytroupe.agent import TinyPerson
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.extraction import ResultsReducer
from tinytroupe.validation import TinyPersonValidator
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe import utils
from tinytroupe import openai_utils
import chevron
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

# Assuming tinytroupe is in the parent directory
sys.path.insert(0, "..")

# Load environment variables from .env file
load_dotenv()

KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

os.environ["AZURE_OPENAI_KEY"] = KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT

# Global variables to store the agent and factory (initialized once)
factory = None
agent = None
agent_expectations = ""


def create_factory(factory_description):
    """Creates the TinyPersonFactory."""
    global factory
    factory = TinyPersonFactory(factory_description)
    return f"Factory created with description: {factory_description}"


def create_agents(number_of_agents: int, agent_description: str = None):
    """Create one or more agents."""
    global factory
    if factory is None:
        return "Factory not created yet!"

    if number_of_agents > 1:
        agents_bios = []
        agents = factory.generate_people(number_of_agents, agent_description)

        for agent in agents:
            bio = agent.minibio()
            agents_bios.append(bio)

        return "\n\n".join(agents_bios)  # Join minibios with line breaks
    else:
        agent = factory.generate_person(agent_description)
        return agent.minibio()


def set_agent_expectations(expectations):
    """Sets the agent expectations."""
    global agent_expectations
    agent_expectations = expectations
    return "Agent expectations set."


def validate_agent():
    """Validates the agent agent."""
    global agent, agent_expectations
    if agent is None:
        return "Agent not created yet!"
    if not agent_expectations:
        return "Agent expectations not set yet!"

    agent_score, agent_justification = TinyPersonValidator.validate_person(
        agent,
        expectations=agent_expectations,
        include_agent_spec=False,
        max_content_length=None,
    )
    return f"Banker score: {agent_score}\nBanker justification: {agent_justification}"


def agent_think(thought):
    """Sets the agent's thought."""
    global agent
    if agent is None:
        return "Agent not created yet!"
    agent.think(thought)
    return f"Agent is now thinking: {thought}"


def agent_listen_and_act(question, max_length=1024, html_output: bool = True):
    """Asks the agent a question and gets their response."""
    global agent

    if agent is None:
        return "Agent not created yet!"

    agent.listen_and_act(question, max_content_length=max_length)

    responses = agent.pretty_current_interactions(max_content_length=max_length)

    if html_output:
        return format_text_to_html(responses)
    else:
        return responses


def agent_interactions(max_length=1024, html_output: bool = True):
    """
    Returns a pretty, readable, string with the current messages.
    """
    global agent
    if agent is None:
        return "Agent not created yet!"

    interactions = agent.pretty_current_interactions(max_content_length=max_length)

    if html_output:
        return format_text_to_html(interactions)
    else:
        return interactions


def save_agents():
    global agent

    for person in agent:
        file_name = f"agent_{person.name}"
        person.save_specification(
            f"/agents/{file_name}.json",
            include_mental_faculties=True,
            include_memory=True,
        )


def format_text_to_html(text):
    """
    Converts a custom-formatted text string into well-formatted HTML using string.replace().

    Args:
        text: The input string in the custom format.

    Returns:
        An HTML string representing the formatted text.
    """

    html_output = "<div class='chat-log'>"

    # Basic replacements for common patterns
    html_output += (
        text.replace(">>>>>>>>>", "")
        .replace("None", "")
        .replace("acts", "")
        .replace("USER", "")
        .replace("Date and time of events:", "<div class='section'><br>")
        .replace("[dim italic cyan1]", "<span class='bold'>")
        .replace("[bold italic cyan1]", "<span class='bold'>")
        .replace("[bold green3]", "<span class='bold'>")
        .replace("[green]", "<span class='bold'>")
        .replace("[grey82]", "<span class='bold'>")
        .replace("[purple]", "<span class='bold'>")
        .replace("[underline]", "")
        .replace("-->", "</span>")
        .replace(":", "</span>")
        .replace("[/]", "</div>")
    )

    # CSS for styling
    css = """
        <style>
        .chat-log { font-family: sans-serif; }
        .bold { font-weight: bold; }
        </style>
    """

    # Close all open tags, just in case
    html_output += css + "</div>"

    return html_output


def export_agent_interactions(
    artifact_name, content_type, content_format, target_format
):
    """
    Exports agent interactions to a file.
    """
    global agent
    if agent is None:
        return "Agent not created yet!"

    artifact_data = agent.current_interactions

    # dedent the strings in the dict:
    new_artifact_data = {}
    for key in artifact_data.keys():
        new_artifact_data[key] = utils.dedent(artifact_data[key])

    agent.export(
        artifact_name,
        new_artifact_data,
        content_type,
        content_format,
        target_format,
        verbose=False,
    )
    return f"Agent interactions exported to {artifact_name}.{target_format}"


def extract_and_show_results(extraction_objective, situation):
    """
    Extracts and displays the results from the agent's interactions.
    """
    global agent
    if agent is None:
        return "Agent not created yet!"

    # Extract results using the provided method
    results = agent.extract_results_from_agent(
        agent, extraction_objective, situation, verbose=False
    )

    if results:
        # Format results for display
        formatted_results = json.dumps(results, indent=4)
        return formatted_results
    else:
        return "No results found or error during extraction."


with gr.Blocks() as demo:
    gr.Markdown("# Synthetic Agent Interview")

    with gr.Tab("Factory Setup"):
        with gr.Row():
            with gr.Column():
                factory_description_input = gr.Textbox(
                    label="Factory Description",
                    placeholder="e.g., One of the largest banks in Brazil, full of bureaucracy and legacy systems.",
                )
                create_factory_btn = gr.Button("Create Factory")
            with gr.Column():
                factory_output = gr.Textbox(label="Factory Creation Result")
            create_factory_btn.click(
                create_factory, inputs=factory_description_input, outputs=factory_output
            )

    with gr.Tab("Agent Creation"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    agent_description_input = gr.Textbox(
                        label="Agent Description",
                        lines=3,
                        placeholder="e.g., The vice-president of product innovation...",
                    )
                    number_of_agents = gr.Number(
                        label="Number of Agents", value=1, precision=0, interactive=True
                    )

                create_agent_btn = gr.Button("Create Agent(s)")

            with gr.Column():
                agent_minibio_output = gr.Textbox(label="Agent Mini-Bio", lines=5)

            create_agent_btn.click(
                create_agents,
                inputs=[number_of_agents, agent_description_input],
                outputs=agent_minibio_output,
            )

        with gr.Accordion("Validate Agent", open=False):
            with gr.Row():
                with gr.Column():
                    agent_expectations_input = gr.Textbox(
                        label="Agent Expectations",
                        lines=5,
                        placeholder="e.g., He/she is: Wealthy, Intelligent...",
                    )
                    set_expectations_btn = gr.Button("Set Agent Expectations")
                with gr.Column():
                    expectations_output = gr.Textbox(label="Expectations Set Result")
                set_expectations_btn.click(
                    set_agent_expectations,
                    inputs=agent_expectations_input,
                    outputs=expectations_output,
                )

            validate_btn = gr.Button("Validate Agent")
            validation_output = gr.Textbox(label="Validation Results", lines=10)
            validate_btn.click(validate_agent, outputs=validation_output)

    with gr.Tab("Interview"):
        with gr.Row():
            with gr.Column():
                thought_input = gr.Textbox(
                    label="Agent's Thought",
                    placeholder="e.g., I am now talking to a business and technology consultant...",
                )
                think_btn = gr.Button("Set Agent's Thought")
            with gr.Column():
                thought_output = gr.Textbox(label="Agent's Thought Process", lines=2)
            think_btn.click(agent_think, inputs=thought_input, outputs=thought_output)

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question to Agent",
                    placeholder="e.g., What are your main problems today?",
                )
            with gr.Column(scale=1):
                max_length_input = gr.Number(
                    label="Max Response Length", value=3000, precision=0
                )
        ask_btn = gr.Button("Ask Agent")

        response_output = gr.HTML(label="Agent's Response")

        ask_btn.click(
            agent_listen_and_act,
            inputs=[question_input, max_length_input],
            outputs=response_output,
        )

    with gr.Tab("Interactions & Analysis"):
        with gr.Accordion("Agent Interactions", open=False):
            with gr.Row():
                with gr.Column():
                    artifact_name_input = gr.Textbox(
                        label="Artifact Name", placeholder="e.g., agent_interview"
                    )
                    content_type_input = gr.Textbox(
                        label="Content Type", placeholder="e.g., agent_interactions"
                    )
                    content_format_input = gr.Textbox(
                        label="Content Format", value="dict", placeholder="e.g., dict"
                    )
                    target_format_input = gr.Dropdown(
                        label="Target Format",
                        choices=["json", "txt", "docx"],
                        value="txt",
                    )
                    export_btn = gr.Button("Get Interactions")
                with gr.Column():
                    export_output = gr.Textbox(label="Interactions", lines=10)
                export_btn.click(
                    export_agent_interactions,
                    inputs=[
                        artifact_name_input,
                        content_type_input,
                        content_format_input,
                        target_format_input,
                    ],
                    outputs=export_output,
                )

        with gr.Accordion("Analyze Agent", open=False):
            with gr.Row():
                with gr.Column():
                    extraction_objective_input = gr.Textbox(
                        label="Analysis Objective",
                        lines=3,
                        placeholder="e.g., Extract the main challenges the agent is facing.",
                    )
                    situation_input = gr.Textbox(
                        label="Situation",
                        lines=3,
                        placeholder="e.g., The agent is speaking to a merchaniser.",
                    )
                    extract_results_btn = gr.Button("Analyze Agent")
                with gr.Column():
                    extraction_results_output = gr.Textbox(label="Analysis", lines=10)
                extract_results_btn.click(
                    extract_and_show_results,
                    inputs=[extraction_objective_input, situation_input],
                    outputs=extraction_results_output,
                )

demo.launch()
