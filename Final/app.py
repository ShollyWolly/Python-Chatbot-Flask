import chatbot_web

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/test")
def home():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_web.chatbot_response(userText)


if __name__ == "__main__":
    app.run(host='0.0.0.0')