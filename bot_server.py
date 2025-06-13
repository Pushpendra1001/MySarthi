from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import logging
from MySarthi import scrape_multiple_urls, chat_about_website


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)  


@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"status": "API is working!"})


@app.route('/simplebot/<bot_id>', methods=['GET'])
def simple_bot(bot_id):
    return jsonify({"message": f"Bot {bot_id} exists!"})


@app.route('/bot/<bot_id>', methods=['GET', 'POST'])
def bot_handler(bot_id):
    logger.debug(f"BOT HANDLER CALLED: bot_id={bot_id}, method={request.method}")
    try:
        
        print(f"[DEBUG] Received request for bot: {bot_id}")
        print(f"[DEBUG] Request method: {request.method}")
        
        
        config_path = os.path.join("chatbots", f"{bot_id}.json")
        
        if not os.path.exists(config_path):
            print(f"[ERROR] Bot file not found: {config_path}")
            return jsonify({"error": "Bot not found"}), 404
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"[DEBUG] Loaded config for bot: {config.get('name')}")
        
        
        if request.method == "GET":
            print(f"[DEBUG] Handling GET request for bot: {bot_id}")
            return jsonify({
                "id": bot_id, 
                "name": config.get("name"), 
                "description": config.get("description")
            })
        
        
        if request.method == "POST":
            try:
                
                request_json = request.get_json()
                message = request_json.get("message", "")
                history = request_json.get("history", [])
                
                print(f"[DEBUG] Handling POST request. Message: {message[:30]}...")
                
                
                urls = config.get("urls", [])
                custom_prompt = config.get("custom_prompt", "")
                
                
                print(f"[DEBUG] Scraping URLs: {urls}")
                website_content = scrape_multiple_urls(urls)
                
                
                print(f"[DEBUG] Processing with Gemini")
                response = ""
                for chunk in chat_about_website(website_content, message, custom_prompt, history):
                    response = chunk
                
                print(f"[DEBUG] Got response: {response[:30]}...")
                return jsonify({"response": response})
            except Exception as e:
                print(f"[ERROR] Error processing POST request: {str(e)}")
                return jsonify({"error": f"Error processing request: {str(e)}"}), 500
        
        return jsonify({"error": "Method not allowed"}), 405
    
    except Exception as e:
        print(f"[ERROR] Bot handler error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting Bot API Server...")
    app.run(host='0.0.0.0', port=7865, debug=True)