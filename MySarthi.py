from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from openai import OpenAI
import gradio as gr
import time
import json
import os
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# === Gemini API Setup ===
gemini_api_key = "gen_key"  

gemini_llm_model = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# === Function to scrape website content ===
def scrape_page(url):
    try:
        print(f"[INFO] Scraping: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"[ERROR] Failed to fetch page content: {e}"

# === Function to scrape multiple URLs ===
def scrape_multiple_urls(urls):
    combined_content = ""
    for url in urls:
        url = url.strip()
        if url:  # Skip empty URLs
            content = scrape_page(url)
            if not content.startswith("[ERROR]"):
                combined_content += f"\n--- Content from {url} ---\n"
                combined_content += content
                combined_content += "\n\n"
    
    if not combined_content:
        return "[ERROR] Failed to fetch content from any of the provided URLs"
    
    return combined_content

# === Function to talk to Gemini using streamed output ===
def chat_about_website(page_text, prompt, custom_prompt, history):
    if not page_text or "Failed" in page_text:
        yield "[ERROR] Cannot respond due to scraping failure."
        return

    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        system_content = f"{custom_prompt}\n\nWebsite content:\n{page_text[:4000]}..."
    else:
        system_content = f"You are an AI assistant that answers questions based on this website's content:\n\n{page_text[:4000]}..."
    
    system_message = {
        "role": "system",
        "content": system_content
    }
    
    # Convert history from Gradio format to API format
    api_messages = []
    if history:
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] in ["user", "assistant"]:  # Only include user and assistant messages
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    user_message = {"role": "user", "content": prompt}
    full_messages = [system_message] + api_messages + [user_message]

    try:
        response = gemini_llm_model.chat.completions.create(
            messages=full_messages,
            model="gemini-1.5-flash",
            stream=True
        )

        output = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                token = chunk.choices[0].delta.content
                output += token
                yield output
    except Exception as e:
        yield f"[ERROR] Gemini API failed: {e}"

# === Chatbot Interface for Testing ===
def test_chat_interface(prompt, history, website_content, custom_prompt):
    # Create a copy of history that we can manipulate
    chat_history = history.copy() if history else []
    # Add the current user message to the history
    chat_history.append({"role": "user", "content": prompt})
    
    # Get the response generator
    response_generator = chat_about_website(website_content, prompt, custom_prompt, history)
    
    # For each update from the generator
    for response in response_generator:
        # Create or update the assistant message
        if len(chat_history) > 0 and chat_history[-1].get("role") == "assistant":
            # Update existing assistant message
            chat_history[-1]["content"] = response
        else:
            # Add new assistant message
            chat_history.append({"role": "assistant", "content": response})
        
        # Yield the entire conversation history for each update
        yield chat_history

# === Function to create chatbot endpoint ===
def create_chatbot_endpoint(bot_name, bot_description, urls, custom_prompt):
    # Generate a unique ID for this chatbot
    bot_id = str(uuid.uuid4())[:8]
    
    # Create a directory to store chatbot data
    if not os.path.exists("chatbots"):
        os.makedirs("chatbots")
    
    # Save the chatbot configuration
    config = {
        "id": bot_id,
        "name": bot_name,
        "description": bot_description,
        "urls": urls,
        "custom_prompt": custom_prompt,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"chatbots/{bot_id}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    # Return the bot ID for embedding
    return bot_id

# === Function to create HTML embed code ===
def create_embed_code(bot_name, bot_id, base_url):
    # Change this line to use the Flask server port
    api_url = base_url.replace(":7864", ":7865")
    
    embed_code = f"""
<!-- {bot_name} Chatbot Widget -->
<div id="chatbot-container" style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    <!-- Chat Button -->
    <div id="chat-button" style="background-color: #4a76fd; color: white; border-radius: 50%; width: 60px; height: 60px; 
                                 display: flex; justify-content: center; align-items: center; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8 15c4.418 0 8-3.134 8-7s-3.582-7-8-7-8 3.134-8 7c0 1.76.743 3.37 1.97 4.6-.097 1.016-.417 2.13-.771 2.966-.079.186.074.394.273.362 2.256-.37 3.597-.938 4.18-1.234A9.06 9.06 0 0 0 8 15z"/>
        </svg>
    </div>
    
    <!-- Chat Window (hidden by default) -->
    <div id="chat-window" style="display: none; position: absolute; bottom: 80px; right: 0; width: 350px; height: 450px; 
                                 background-color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
                                 overflow: hidden; flex-direction: column;">
        <!-- Header -->
        <div style="background-color: #4a76fd; color: white; padding: 15px; display: flex; justify-content: space-between; align-items: center;">
            <div>{bot_name}</div>
            <div id="close-chat" style="cursor: pointer;">✖</div>
        </div>
        
        <!-- Messages Area -->
        <div id="chat-messages" style="flex-grow: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column;"></div>
        
        <!-- Input Area -->
        <div style="padding: 15px; border-top: 1px solid #e0e0e0; display: flex;">
            <input id="chat-input" type="text" placeholder="Type your message..." 
                   style="flex-grow: 1; padding: 10px; border: 1px solid #e0e0e0; border-radius: 20px; margin-right: 10px;">
            <button id="send-button" style="background-color: #4a76fd; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11v-.001z"/>
                </svg>
            </button>
        </div>
    </div>
</div>

<script>
    // Chatbot functionality
    (function() {{
        // Elements
        const chatButton = document.getElementById('chat-button');
        const chatWindow = document.getElementById('chat-window');
        const closeChat = document.getElementById('close-chat');
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-button');
        
        // Bot configuration
        const botId = '{bot_id}';
        const botEndpoint = '{api_url}/bot/{bot_id}';  // Use the Flask server URL
        const botName = '{bot_name}';
        
        // Chat history
        let messageHistory = [];
        
        // Toggle chat window
        chatButton.addEventListener('click', () => {{
            chatWindow.style.display = 'flex';
            chatButton.style.display = 'none';
        }});
        
        closeChat.addEventListener('click', () => {{
            chatWindow.style.display = 'none';
            chatButton.style.display = 'flex';
        }});
        
        // Add a message to the chat
        function addMessage(content, isUser) {{
            const messageDiv = document.createElement('div');
            messageDiv.style.alignSelf = isUser ? 'flex-end' : 'flex-start';
            messageDiv.style.maxWidth = '70%';
            messageDiv.style.marginBottom = '10px';
            
            const bubble = document.createElement('div');
            bubble.style.padding = '10px 15px';
            bubble.style.borderRadius = '18px';
            bubble.style.backgroundColor = isUser ? '#dcf8c6' : '#f1f0f0';
            bubble.style.boxShadow = '0 1px 2px rgba(0,0,0,0.1)';
            bubble.textContent = content;
            
            messageDiv.appendChild(bubble);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Add to history
            if (isUser) {{
                messageHistory.push({{role: "user", content: content}});
            }} else {{
                messageHistory.push({{role: "assistant", content: content}});
            }}
        }}
        
        // Send message
        function sendMessage() {{
            const message = chatInput.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            chatInput.value = '';
            
            // Show typing indicator
            const typingDiv = document.createElement('div');
            typingDiv.style.alignSelf = 'flex-start';
            typingDiv.style.maxWidth = '70%';
            typingDiv.style.marginBottom = '10px';
            typingDiv.id = 'typing-indicator';
            
            const typingBubble = document.createElement('div');
            typingBubble.style.padding = '10px 15px';
            typingBubble.style.borderRadius = '18px';
            typingBubble.style.backgroundColor = '#f1f0f0';
            typingBubble.style.boxShadow = '0 1px 2px rgba(0,0,0,0.1)';
            typingBubble.textContent = 'Typing...';
            
            typingDiv.appendChild(typingBubble);
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // In a real implementation, you would call your API here
            fetch(botEndpoint, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    message: message,
                    history: messageHistory
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                // Remove typing indicator
                chatMessages.removeChild(document.getElementById('typing-indicator'));
                
                // Add bot response
                addMessage(data.response, false);
            }})
            .catch(error => {{
                // Remove typing indicator
                chatMessages.removeChild(document.getElementById('typing-indicator'));
                
                // Add error message
                addMessage("Sorry, there was an error connecting to the server. Please try again later.", false);
                console.error('Error:', error);
            }});
        }}
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') sendMessage();
        }});
        
        // Add initial message
        setTimeout(() => {{
            addMessage(`Hello! I'm your {bot_name} assistant. How can I help you today?`, false);
        }}, 500);
    }})();
</script>
"""
    return embed_code

# === Gradio Interface ===
with gr.Blocks(title="Your Sarthi") as demo:
    gr.Markdown("# 🤖 Your Sarthi")
    gr.Markdown("Create custom chatbots for your website in minutes!")
    
    with gr.Tab("Create Chatbot"):
        with gr.Row():
            with gr.Column(scale=1):
                bot_name = gr.Textbox(label="Chatbot Name", placeholder="My Website Assistant")
                bot_description = gr.Textbox(label="Chatbot Description", placeholder="AI assistant for my website")
            
        with gr.Row():
            with gr.Column(scale=1):
                urls_input = gr.Textbox(
                    label="Website URLs (one per line)", 
                    placeholder="https://example.com\nhttps://example.com/about",
                    lines=5
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                custom_prompt = gr.Textbox(
                    label="Custom Instructions (Optional)", 
                    placeholder="You are a helpful assistant for our company. Answer questions based on our website content. If you don't know the answer, suggest contacting support.",
                    lines=5
                )
        
        scrape_btn = gr.Button("Scrape Websites & Build Knowledge Base")
        scrape_status = gr.Markdown("") 
        website_content = gr.State()
        
        # Function to scrape websites and build knowledge base
        def build_knowledge_base(urls_text):
            urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
            if not urls:
                return "Please enter at least one valid URL", None
            
            content = scrape_multiple_urls(urls)
            if content.startswith("[ERROR]"):
                return f"Failed to build knowledge base: {content}", None
            
            return f"✅ Successfully scraped {len(urls)} website(s). Knowledge base built with {len(content)} characters.", content
        
        scrape_btn.click(
            fn=build_knowledge_base,
            inputs=[urls_input],
            outputs=[scrape_status, website_content]
        )
        
        gr.Markdown("## Test Your Chatbot")
        gr.Markdown("Try interacting with your chatbot to see how it performs")
        
        test_chatbot = gr.Chatbot(type="messages")
        test_prompt = gr.Textbox(label="Ask a question", placeholder="What does this website offer?")
        test_history = gr.State([])
        
        test_btn = gr.Button("Ask Question")
        
        test_btn.click(
            fn=test_chat_interface,
            inputs=[test_prompt, test_history, website_content, custom_prompt],
            outputs=[test_chatbot],
            concurrency_limit=1
        ).then(
            fn=lambda x, y: (x, y),
            inputs=[test_chatbot, test_prompt],
            outputs=[test_history, test_prompt]
        )
        
        gr.Markdown("## Get Embed Code")
        
        create_btn = gr.Button("Buy Now Your Sarthi")
        create_status = gr.Markdown("")
        
        # Instead of using a Group as output, use a Checkbox to control visibility
        embed_code_visibility = gr.Checkbox(visible=False, label="")  # Hidden checkbox to control visibility
        embed_code_output = gr.Code(label="Embed Code", language="html", lines=20, visible=False)
        chatbot_url = gr.Markdown(visible=False)

        # Modified function to return values for actual components
        def create_chatbot_handler(bot_name, bot_description, urls_text, custom_prompt, website_content):
            if not bot_name:
                return "Please enter a name for your chatbot", False, "", False
            
            urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
            if not urls:
                return "Please enter at least one valid URL", False, "", False
            
            if not website_content:
                return "Please click 'Scrape Websites & Build Knowledge Base' first", False, "", False
            
            try:
                # Create chatbot endpoint
                bot_id = create_chatbot_endpoint(bot_name, bot_description, urls, custom_prompt)
                
                # Get the base URL of the current Gradio instance
                server_port = demo.server_port if hasattr(demo, 'server_port') else 7864  # Changed from 7860
                base_url = f"http://127.0.0.1:{server_port}"
                
                # Generate embed code
                embed_code = create_embed_code(bot_name, bot_id, base_url)
                
                chatbot_direct_url = f"{base_url}/bot/{bot_id}"
                
                return (
                    f"✅ Chatbot created successfully! ID: {bot_id}\n\nEmbed the code below on your website:",
                    True,  # Set checkbox to True to show components
                    embed_code,  # The actual code content 
                    f"Direct URL: [{chatbot_direct_url}]({chatbot_direct_url})"
                )
            except Exception as e:
                return f"❌ Failed to create chatbot: {str(e)}", False, "", ""

        # Connect the button to update component visibility directly
        create_btn.click(
            fn=create_chatbot_handler,
            inputs=[bot_name, bot_description, urls_input, custom_prompt, website_content],
            outputs=[create_status, embed_code_visibility, embed_code_output, chatbot_url]
        )
    
    with gr.Tab("My Chatbots"):
        gr.Markdown("## Your Chatbots")
        
        refresh_btn = gr.Button("Refresh Chatbot List")
        chatbots_list = gr.Dataframe(
            headers=["ID", "Name", "Description", "Created At", "URL"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=10
        )
        
        # Function to refresh chatbot list
        def refresh_chatbots():
            if not os.path.exists("chatbots"):
                return []
            
            chatbots = []
            server_port = demo.server_port if hasattr(demo, 'server_port') else 7860
            base_url = f"http://127.0.0.1:{server_port}"
            
            for filename in os.listdir("chatbots"):
                if filename.endswith(".json"):
                    with open(os.path.join("chatbots", filename), "r", encoding="utf-8") as f:
                        config = json.load(f)
                        bot_id = config.get("id")
                        # Make the ID a clickable link for easy copying
                        bot_id_link = f"<a href='#' onclick='navigator.clipboard.writeText(\"{bot_id}\"); return false;'>{bot_id}</a>"
                        bot_url = f"{base_url}/bot/{bot_id}"
                        chatbots.append([
                            bot_id_link,  # Changed to HTML link
                            config.get("name"),
                            config.get("description"),
                            config.get("created_at"),
                            bot_url
                        ])
            
            return chatbots
        
        refresh_btn.click(
            fn=refresh_chatbots,
            inputs=[],
            outputs=[chatbots_list]
        )
        
        # Load chatbots on tab open
        demo.load(
            fn=refresh_chatbots,
            inputs=[],
            outputs=[chatbots_list]
        )

# Create necessary directories
os.makedirs("chatbots", exist_ok=True)

# Create a simple default icon for chatbots
if not os.path.exists("default_icon.png"):
    try:
        from PIL import Image
        img = Image.new('RGB', (128, 128), color = (73, 109, 137))
        img.save('default_icon.png')
    except:
        pass

# === Function to handle bot API requests ===
def bot_handler(request, bot_id: str):
    logger.debug(f"BOT HANDLER CALLED: bot_id={bot_id}, method={request.method}")
    try:
        # Print debug information
        print(f"[DEBUG] Received request for bot: {bot_id}")
        print(f"[DEBUG] Request method: {request.method}")
        
        # Load bot configuration
        config_path = os.path.join("chatbots", f"{bot_id}.json")
        
        if not os.path.exists(config_path):
            print(f"[ERROR] Bot file not found: {config_path}")
            return {"error": "Bot not found"}
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"[DEBUG] Loaded config for bot: {config.get('name')}")
        
        # Get request data
        if request.method == "GET":
            print(f"[DEBUG] Handling GET request for bot: {bot_id}")
            return {"id": bot_id, "name": config.get("name"), "description": config.get("description")}
        
        # Handle chat request
        if request.method == "POST":
            try:
                # Parse request body
                request_json = request.json()
                message = request_json.get("message", "")
                history = request_json.get("history", [])
                
                print(f"[DEBUG] Handling POST request. Message: {message[:30]}...")
                
                # Get website content from the config
                urls = config.get("urls", [])
                custom_prompt = config.get("custom_prompt", "")
                
                # Scrape website content
                print(f"[DEBUG] Scraping URLs: {urls}")
                website_content = scrape_multiple_urls(urls)
                
                # Process with Gemini
                print(f"[DEBUG] Processing with Gemini")
                response = ""
                for chunk in chat_about_website(website_content, message, custom_prompt, history):
                    response = chunk
                
                print(f"[DEBUG] Got response: {response[:30]}...")
                return {"response": response}
            except Exception as e:
                print(f"[ERROR] Error processing POST request: {str(e)}")
                return {"error": f"Error processing request: {str(e)}"}
        
        return {"error": "Method not allowed"}
    
    except Exception as e:
        print(f"[ERROR] Bot handler error: {str(e)}")
        return {"error": str(e)}

# Alternative approach using mount

if __name__ == "__main__":
    print("[INFO] Starting Your Sarthi...")
    logger.debug("Registering routes...")
    
    from fastapi import FastAPI
    api_app = FastAPI()
    
    @api_app.get("/test")
    async def test_route():
        return {"status": "API is working!"}
    
    @api_app.get("/simplebot/{bot_id}")
    async def simple_bot(bot_id: str):
        return {"message": f"Bot {bot_id} exists!"}
    
    @api_app.get("/bot/{bot_id}")
    async def get_bot(bot_id: str, request: Request):
        logger.debug(f"GET request received for bot: {bot_id}")
        return bot_handler(request, bot_id)
        
    @api_app.post("/bot/{bot_id}")
    async def post_bot(bot_id: str, request: Request):
        logger.debug(f"POST request received for bot: {bot_id}")
        return bot_handler(request, bot_id)
    
    # Add CORS middleware
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount our API app to the Gradio app
    demo.app.mount("/", api_app)
    
    # Launch Gradio
    logger.debug("Launching Gradio app...")
    demo.launch(debug=True, server_port=7864)
