import streamlit as st
import json
import os
from typing import List, Dict
import tempfile
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
import PyPDF2
import time
import shutil
import chardet
from crewai_tools import SerperDevTool
import time
from streamlit.components.v1 import html
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()
if not SERPER_API_KEY:
    st.error("Please set the SERPER_API_KEY environment variable.")
    st.stop()

KNOWLEDGE_BASE_DIR = "knowledge_base"
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)

# Custom CSS and JavaScript for the futuristic UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');

body {
    font-family: 'Orbitron', sans-serif;
    background-color: #0f0f1a;
    color: #ffffff;
    overflow-x: hidden;
}

.stApp {
    background-color: rgba(15, 15, 26, 0.8);
}

.sidebar {
    background-color: rgba(26, 26, 46, 0.8);
    padding: 20px;
    border-radius: 10px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.futuristic-title {
    font-size: 24px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
    text-shadow: 0 0 10px #4a4ae9;
    animation: glow 2s ease-in-out infinite alternate;
}

.message {
    margin-bottom: 15px;
    padding: 15px;
    border-radius: 20px;
    max-width: 80%;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    opacity: 0;
    transform: translateY(20px);
    background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255,255,255,0.1);
}

.manager { 
    background: linear-gradient(145deg, rgba(74, 74, 233, 0.8) 0%, rgba(74, 74, 233, 0.6) 100%);
    color: white; 
    margin-left: auto;
    animation: fadeInSlideLeft 0.5s ease-out forwards;
}

.researcher, .writer, .analyst, .financial-expert { 
    background: linear-gradient(145deg, rgba(233, 74, 74, 0.8) 0%, rgba(233, 74, 74, 0.6) 100%);
    color: white;
    animation: fadeInSlideRight 0.5s ease-out forwards;
}

.writer { 
    background: linear-gradient(145deg, rgba(74, 233, 74, 0.8) 0%, rgba(74, 233, 74, 0.6) 100%);
    color: black; 
}

.analyst { 
    background: linear-gradient(145deg, rgba(233, 233, 74, 0.8) 0%, rgba(233, 233, 74, 0.6) 100%);
    color: black; 
}

.financial-expert { 
    background: linear-gradient(145deg, rgba(74, 233, 233, 0.8) 0%, rgba(74, 233, 233, 0.6) 100%);
    color: black; 
}

.agent-name {
    font-weight: bold;
    margin-bottom: 5px;
}

.processing {
    text-align: center;
    font-style: italic;
    color: #aaaaaa;
    margin: 10px 0;
}

.futuristic-input {
    background-color: rgba(42, 42, 78, 0.6);
    border: none;
    border-radius: 5px;
    color: #ffffff;
    padding: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.futuristic-input:focus {
    background-color: rgba(42, 42, 78, 0.8);
    box-shadow: 0 0 10px rgba(74, 74, 233, 0.5);
}

.futuristic-button {
    background-color: #4a4ae9;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.futuristic-button:hover {
    background-color: #3a3ad9;
    box-shadow: 0 0 15px rgba(74, 74, 233, 0.7);
    transform: translateY(-2px);
}

@keyframes glow {
    from { text-shadow: 0 0 5px #4a4ae9, 0 0 10px #4a4ae9, 0 0 15px #4a4ae9; }
    to { text-shadow: 0 0 10px #4a4ae9, 0 0 20px #4a4ae9, 0 0 30px #4a4ae9; }
}

@keyframes fadeInSlideLeft {
    from { opacity: 0; transform: translateX(-50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes fadeInSlideRight {
    from { opacity: 0; transform: translateX(50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes typing {
    0% { content: ''; }
    25% { content: '.'; }
    50% { content: '..'; }
    75% { content: '...'; }
    100% { content: ''; }
}

.typing-animation {
    animation: typing 1.5s infinite;
    display: inline-block;
}

.stars {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.star {
    position: absolute;
    width: 2px;
    height: 2px;
    background-color: #ffffff;
    border-radius: 50%;
    animation: twinkle 2s infinite ease-in-out;
}

@keyframes twinkle {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
}
            

/* Target Streamlit's main container */
.main.st-emotion-cache-bm2z3a {
    background-color: transparent;
    background-image: var(--background-image);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Ensure our messages are on top */
.message {
    position: relative;
    z-index: 10;
}

/* Adjust the stars to be behind the content but visible */
.stars {
    z-index: 1;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', (event) => {
    function createStars() {
        const stars = document.createElement('div');
        stars.className = 'stars';
        for (let i = 0; i < 100; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = `${Math.random() * 100}%`;
            star.style.top = `${Math.random() * 100}%`;
            star.style.animationDelay = `${Math.random() * 2}s`;
            stars.appendChild(star);
        }
        document.body.appendChild(stars);
    }

    createStars();

    function scrollToBottom() {
        const messages = document.querySelector('.stApp');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    }

    function animateMessages() {
        const messages = document.querySelectorAll('.message');
        messages.forEach((msg, index) => {
            setTimeout(() => {
                msg.style.opacity = '1';
                msg.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                scrollToBottom();
                animateMessages();
            }
        });
    });

    const config = { childList: true, subtree: true };
    observer.observe(document.body, config);
});
</script>
""", unsafe_allow_html=True)

# Add custom background if it exists
if os.path.exists("background_image.png"):
    st.markdown("""
    <style>
    body {
        background-image: url('background_image.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

# Add semi-transparent overlay
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(15, 15, 26, 0.8); z-index: -1;"></div>
""", unsafe_allow_html=True)

if 'background_image' not in st.session_state:
    st.session_state.background_image = None


def inject_custom_js():
    html("""
    <script>
    function createStars() {
        const stars = document.createElement('div');
        stars.className = 'stars';
        for (let i = 0; i < 100; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = `${Math.random() * 100}%`;
            star.style.top = `${Math.random() * 100}%`;
            star.style.animationDelay = `${Math.random() * 2}s`;
            stars.appendChild(star);
        }
        document.body.appendChild(stars);
    }

    function scrollToBottom() {
        const mainContent = document.querySelector('.main.st-emotion-cache-bm2z3a');
        if (mainContent) {
            mainContent.scrollTop = mainContent.scrollHeight;
        }
        
        // Fallback: scroll the entire page if the main content isn't found
        window.scrollTo(0, document.body.scrollHeight);
    }

    createStars();
    
    const observer = new MutationObserver((mutations) => {
        scrollToBottom();
    });

    const config = { childList: true, subtree: true };
    observer.observe(document.body, config);

    // Additional scroll trigger
    setInterval(scrollToBottom, 1000);
    </script>
    """, height=0)

# Near the top of your script, add:
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Create a directory to store uploaded files if it doesn't exist
        if not os.path.exists("uploaded_files"):
            os.makedirs("uploaded_files")
        
        # Save the file
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None



# Updated display_message function
def display_message(agent_name, message, is_thinking=False):
    agent_class = agent_name.lower().replace(' ', '-')
    text_color = "black" if agent_class in ["writer", "analyst", "financial-expert"] else "white"
    
    message_html = f"""
    <div class="message {agent_class}">
        <div class="agent-name" style="color: {text_color};">{agent_name}</div>
        <div class="message-content" style="color: {text_color};">
            {"Thinking<span class='typing-animation'>...</span>" if is_thinking else message}
        </div>
    </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)
    
    if is_thinking:
        time.sleep(2)  # Simulate thinking time

# Function to show processing message
def show_processing(message):
    st.markdown(f'<div class="processing">{message}</div>', unsafe_allow_html=True)

# Updated Agent class
class Agent:
    def __init__(self, name: str, instructions: str, backstory: str):
        self.name = name
        self.instructions = instructions
        self.backstory = backstory
        self.llm = OpenAI(temperature=0.7, api_key=OPENAI_API_KEY)
        self.serper_tool = SerperDevTool()

    def display_message(self, message: str, is_thinking=False):
        display_message(self.name, message, is_thinking)

    def process(self, input_data: str, knowledge_base_used: bool = False, file_summary: str = "") -> str:
        self.display_message("Processing task...", is_thinking=True)
        
        prompt = PromptTemplate(
            input_variables=["instructions", "backstory", "input_data", "file_summary"],
            template="Instructions: {instructions}\nBackstory: {backstory}\nTask: {input_data}\nFile Summary: {file_summary}\nAnalyze the given information and provide insights. Use internet search if necessary.\nResponse:"
        )
        chain = prompt | self.llm
        result = chain.invoke({
            "instructions": self.instructions,
            "backstory": self.backstory,
            "input_data": input_data,
            "file_summary": file_summary
        })
        
        internet_used = False
        if "search the internet" in result.lower():
            internet_used = True
            self.display_message("Searching the internet...", is_thinking=True)
            search_query = result.split("search the internet for ")[-1].split(".")[0]
            search_result = self.serper_tool.search(search_query)
            result += f"\n\nInternet search results: {search_result}"
        
        prefix = []
        if knowledge_base_used:
            prefix.append("[Used knowledge base]")
        if internet_used:
            prefix.append("[Used internet search]")
        if file_summary:
            prefix.append("[Analyzed uploaded file]")
        
        prefix_str = " ".join(prefix) + " " if prefix else ""
        final_result = f"{prefix_str}Task completed. Response: {result}"
        self.display_message(final_result)
        return result

class Manager(Agent):
    def delegate(self, crew: List[Agent], input_data: str, knowledge_base_used: bool, file_summary: str) -> str:
        self.display_message(f"Delegating task: {input_data}")

        crew_output = []
        for agent in crew:
            self.display_message(f"Assigning task to {agent.name}...", is_thinking=True)
            self.display_message(f"Instructions for {agent.name}: Analyze the following task from your perspective and provide insights. Use internet search if necessary.\nTask: {input_data}")
            agent_output = agent.process(input_data, knowledge_base_used, file_summary)
            crew_output.append(f"{agent.name}: {agent_output}")
            self.display_message(f"Received output from {agent.name}")

        self.display_message("Reviewing crew outputs and providing final analysis...", is_thinking=True)
        final_prompt = f"""
        As the Manager, review the following crew outputs and provide a final analysis and recommendation:

        {' '.join(crew_output)}

        Original task: {input_data}
        File Summary: {file_summary}

        Provide a comprehensive summary and final recommendation based on the crew's output and the original task.
        Use internet search if additional information is needed.
        """
        return self.process(final_prompt, knowledge_base_used, file_summary)


class Crew:
    def __init__(self, manager: Manager, agents: List[Agent]):
        self.manager = manager
        self.agents = agents

    def process(self, input_data: str, knowledge_base_used: bool, file_summary: str) -> str:
        return self.manager.delegate(self.agents, input_data, knowledge_base_used, file_summary)
# Persistence functions
def save_agent_config(configs: Dict[str, Dict[str, str]]):
    with open("agent_configs.json", "w") as f:
        json.dump(configs, f)

def load_agent_config() -> Dict[str, Dict[str, str]]:
    try:
        with open("agent_configs.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def process_file(file):
    content = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    else:
        raw_content = file.getvalue()
        detected_encoding = chardet.detect(raw_content)['encoding'] or 'utf-8'
        try:
            content = raw_content.decode(detected_encoding)
        except UnicodeDecodeError:
            content = raw_content.decode('utf-8', errors='replace')
    
    # Create a simple summary using LLM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(content)
    
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    summary = chain.run(text=texts[0])  # Summarize the first chunk for brevity
    
    return content, summary
# Function to save uploaded file to knowledge base
def save_to_knowledge_base(uploaded_file):
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

# Function to get list of files in knowledge base
def get_knowledge_base_files():
    return [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isfile(os.path.join(KNOWLEDGE_BASE_DIR, f))]

# Function to delete file from knowledge base
def delete_from_knowledge_base(filename):
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

# Improved function to read file content
def read_file_content(file_path):
    with open(file_path, 'rb') as f:
        raw_content = f.read()
    detected_encoding = chardet.detect(raw_content)['encoding'] or 'utf-8'
    try:
        return raw_content.decode(detected_encoding)
    except UnicodeDecodeError:
        return raw_content.decode('utf-8', errors='replace')

# Streamlit app structure
st.sidebar.markdown('<div class="sidebar">', unsafe_allow_html=True)


# Sidebar with input components
st.sidebar.markdown('<h1 class="futuristic-title">Futuristic Auction Document Processor</h1>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="futuristic-input">Custom Background</p>', unsafe_allow_html=True)
background_image = st.sidebar.file_uploader("Upload a background image", type=["png", "jpg", "jpeg"])
if background_image:
    file_path = save_uploaded_file(background_image)
    st.session_state.background_image_path = file_path
    st.sidebar.success("Background image uploaded successfully!")
initial_prompt = st.sidebar.text_area("Enter your initial prompt about the auction:", height=150, key="initial_prompt")

uploaded_file = st.sidebar.file_uploader("Upload a file for this session (optional)", type=["txt", "pdf"])


# After the sidebar section, add:
if 'background_image_path' in st.session_state and os.path.exists(st.session_state.background_image_path):
    with open(st.session_state.background_image_path, "rb") as f:
        bg_image_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .main.st-emotion-cache-bm2z3a {{
            background-image: url(data:image/{background_image.type};base64,{bg_image_base64});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Knowledge Base Section
st.sidebar.markdown('<p class="futuristic-input">Knowledge Base</p>', unsafe_allow_html=True)
kb_files = st.sidebar.file_uploader("Upload files to Knowledge Base", accept_multiple_files=True)
if kb_files:
    for file in kb_files:
        save_to_knowledge_base(file)
        st.sidebar.success(f"Added {file.name} to Knowledge Base")

# Display and manage Knowledge Base files
st.sidebar.markdown("### Current Knowledge Base Files")
for file in get_knowledge_base_files():
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(file)
    if col2.button("Delete", key=f"delete_{file}"):
        delete_from_knowledge_base(file)
        st.sidebar.success(f"Deleted {file} from Knowledge Base")
        st.experimental_rerun()

# Load saved configurations
agent_configs = load_agent_config()

# Agent customization
st.sidebar.markdown('<p class="futuristic-input">Customize Agents</p>', unsafe_allow_html=True)
agent_names = ["Manager", "Researcher", "Writer", "Analyst", "Financial Expert"]
for agent_name in agent_names:
    with st.sidebar.expander(f"{agent_name} Configuration"):
        instructions = st.text_area(
            f"{agent_name} Instructions", 
            agent_configs.get(agent_name, {}).get("instructions", ""),
            key=f"{agent_name}_instructions"
        )
        backstory = st.text_area(
            f"{agent_name} Backstory", 
            agent_configs.get(agent_name, {}).get("backstory", ""),
            key=f"{agent_name}_backstory"
        )
        agent_configs[agent_name] = {"instructions": instructions, "backstory": backstory}

if st.sidebar.button("Save Agent Configurations", key="save_config"):
    save_agent_config(agent_configs)
    st.sidebar.success("Configurations saved successfully!")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
chat_placeholder = st.empty()

# Process button
if st.sidebar.button("Process", key="process"):
    if not initial_prompt:
        st.error("Please enter an initial prompt.")
    else:
        with st.spinner("Processing..."):
            try:
                # Create agents and crew
                manager = Manager("Manager", agent_configs["Manager"]["instructions"], agent_configs["Manager"]["backstory"])
                agents = [
                    Agent(name, config["instructions"], config["backstory"]) 
                    for name, config in agent_configs.items() 
                    if name != "Manager"
                ]
                crew = Crew(manager, agents)

                # Process input
                input_data = initial_prompt
                file_summary = ""
                if uploaded_file:
                    _, file_summary = process_file(uploaded_file)
                    input_data += f"\n\nA file has been uploaded. Here's a summary of its contents: {file_summary}"

                # Check if knowledge base is used
                knowledge_base_used = len(get_knowledge_base_files()) > 0

                # Add a note about knowledge base if it's used
                if knowledge_base_used:
                    input_data += "\n\nKnowledge base information is available for this task."

                result = crew.process(input_data, knowledge_base_used, file_summary)

                # Display result
                st.markdown('<p class="futuristic-title">Processing Result</p>', unsafe_allow_html=True)
                st.write(result)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

inject_custom_js()

# Add any additional UI elements or features as needed
st.markdown("---")
st.markdown("Developed with ❤️ by Vlad and AiPro")