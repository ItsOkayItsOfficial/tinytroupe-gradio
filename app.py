import gradio as gr
import json
import sys
import os
import re
from dotenv import load_dotenv # Import load_dotenv

# Assuming tinytroupe is in the parent directory
sys.path.insert(0, '..')

import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.extraction import ResultsReducer
from tinytroupe.validation import TinyPersonValidator
import tinytroupe.control as control

# Load environment variables from .env file
load_dotenv()

KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


os.environ["AZURE_OPENAI_KEY"] = KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT

# Global variables to store the agent and factory (initialized once)
factory = None
customer = None
customer_expectations = ""

def create_factory(factory_description):
    """Creates the TinyPersonFactory."""
    global factory
    factory = TinyPersonFactory(factory_description)
    return f"Factory created with description: {factory_description}"

def create_customer(customer_description):
    """Creates the customer agent."""
    global customer, factory
    if factory is None:
        return "Factory not created yet!"
    customer = factory.generate_person(customer_description)
    return customer.minibio()

def set_customer_expectations(expectations):
    """Sets the customer expectations."""
    global customer_expectations
    customer_expectations = expectations
    return "Customer expectations set."

def validate_customer():
    """Validates the customer agent."""
    global customer, customer_expectations
    if customer is None:
      return "Customer not created yet!"
    if not customer_expectations:
        return "Customer expectations not set yet!"

    customer_score, customer_justification = TinyPersonValidator.validate_person(customer, expectations=customer_expectations, include_agent_spec=False, max_content_length=None)
    return f"Banker score: {customer_score}\nBanker justification: {customer_justification}"

def customer_think(thought):
    """Sets the customer's thought."""
    global customer
    if customer is None:
        return "Customer not created yet!"
    customer.think(thought)
    return f"Customer is now thinking: {thought}"

def customer_listen_and_act(question, max_length, html_output: bool = True):
    """Asks the customer a question and gets their response."""
    global customer
    if customer is None:
        return "Customer not created yet!"
    
    customer.listen_and_act(question, max_content_length=max_length)

    responses = customer.pretty_current_interactions(max_content_length=max_length)
    
    if html_output:
        return format_text_to_html(responses)
    else:
        return responses

def customer_interactions(max_length=1024, html_output: bool= True):
    """
    Returns a pretty, readable, string with the current messages.
    """
    global customer
    if customer is None:
        return "Customer not created yet!"
    
    interactions = customer.pretty_current_interactions(max_content_length=max_length)

    if html_output:
        return format_text_to_html(interactions)
    else:
        return interactions

def format_interactions(interactions_string):
    """
    Formats the interactions string for better readability.
    """

    interactions_string.replace("[dim italic cyan1]", "<span style='color:blue'>").replace("Date and time of events: None", "<p>").replace("[/]", "</p>")

    # Use regular expressions to identify and format sections
    sections = interactions_string.split('Date and time of events: None')
    formatted_text = ""
    for section in sections:
        if "[underline]" in section:
          new_line+= ""
        elif "[/]" in section:
          new_line+= ""
        elif "[THOUGHT]" in section:
          new_line+= "[THOUGHT]"
        elif "[CONVERSATION]" in section:
          new_line+= "[CONVERSATION]"
        elif "[TALK]" in section:
          new_line+= "[TALK]"
        elif "[THINK]" in section:
          new_line+= "[THINK]"
        elif "[DONE]" in section:
          new_line+= "[DONE]"
        else:
          new_line+=section

    return formatted_text

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
    html_output += text.replace(">>>>>>>>>", "") \
        .replace("None", "") \
        .replace("acts", "") \
        .replace("USER", "") \
        .replace("Date and time of events:", "<div class='section'><br>") \
        .replace("[dim italic cyan1]", "<span class='bold'>") \
        .replace("[bold italic cyan1]", "<span class='bold'>") \
        .replace("[bold green3]", "<span class='bold'>") \
        .replace("[green]", "<span class='bold'>") \
        .replace("[grey82]", "<span class='bold'>") \
        .replace("[purple]", "<span class='bold'>") \
        .replace("[underline]", "") \
        .replace("-->", "</span>")\
        .replace(":", "</span>") \
        .replace("[/]", "</div>")

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

with gr.Blocks() as demo:
    gr.Markdown("# Synthetic Customer Interview")

    with gr.Tab("Factory Setup"):
        factory_description_input = gr.Textbox(label="Factory Description", placeholder="e.g., One of the largest banks in Brazil, full of bureaucracy and legacy systems.")
        create_factory_btn = gr.Button("Create Factory")
        factory_output = gr.Textbox(label="Factory Creation Result")
        create_factory_btn.click(create_factory, inputs=factory_description_input, outputs=factory_output)

    with gr.Tab("Agent Creation & Validation"):
        customer_description_input = gr.Textbox(label="Customer Description", lines=3, placeholder="e.g., The vice-president of product innovation...")
        create_customer_btn = gr.Button("Create Customer Agent")
        customer_minibio_output = gr.Textbox(label="Customer Mini-Bio", lines=5)
        create_customer_btn.click(create_customer, inputs=customer_description_input, outputs=customer_minibio_output)
        
        customer_expectations_input = gr.Textbox(label="Customer Expectations", lines=5, placeholder="e.g., He/she is: Wealthy, Intelligent...")
        set_expectations_btn = gr.Button("Set Customer Expectations")
        expectations_output = gr.Textbox(label="Expectations Set Result")
        set_expectations_btn.click(set_customer_expectations, inputs=customer_expectations_input, outputs=expectations_output)

        validate_btn = gr.Button("Validate Customer Agent")
        validation_output = gr.Textbox(label="Validation Results", lines=10)
        validate_btn.click(validate_customer, outputs=validation_output)

    with gr.Tab("Interview"):
        with gr.Row():
          thought_input = gr.Textbox(label="Customer's Thought", placeholder="e.g., I am now talking to a business and technology consultant...")
          think_btn = gr.Button("Set Customer's Thought")
        thought_output = gr.Textbox(label="Customer's Thought Process", lines=2)
        think_btn.click(customer_think, inputs=thought_input, outputs=thought_output)
        
        question_input = gr.Textbox(label="Question to Customer", placeholder="e.g., What are your main problems today?")
        max_length_input = gr.Number(label="Max Response Length", value=3000, precision=0)
        ask_btn = gr.Button("Ask Customer")
        response_output = gr.HTML(label="Customer's Response")
        
        #New button added:
        show_interactions_btn = gr.Button("Show Interactions")
        interactions_output = gr.HTML(label="All Customer Interactions")

        ask_btn.click(customer_listen_and_act, inputs=[question_input, max_length_input], outputs=response_output)
        show_interactions_btn.click(customer_interactions, inputs=[max_length_input], outputs=interactions_output)

demo.launch()
