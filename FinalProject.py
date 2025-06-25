from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

tokenizer=TreebankWordTokenizer()

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "Good morning", "What's up?", "Hey there"],
        "responses": [
            "Hello! It's great to hear from you.",
            "Hi there! How can I support you today?",
            "Hey! Hope you're having a nice day so far."
        ]
    },
    {
        "tag": "Wishes",
        "patterns": ["How have you been", "How are you?", "How's the day?", "You okay?"],
        "responses": [
            "I'm doing well, thanks for asking! How about you?",
            "Feeling good and ready to help. What’s on your mind?",
            "I'm just a chatbot, but I'm glad to be here with you!"
        ]
    },
    {
        "tag": "Joke",
        "patterns": ["Tell me a joke", "Joke", "Funny", "Jokes", "Make me laugh"],
        "responses": [
            "Why don't skeletons fight each other? Because they don't have the guts.",
            "I'm reading a book about anti-gravity—it's impossible to put down!",
            "Why did the computer show up late to work? It had a hard drive!"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "Goodbye!", "Good bye", "See you", "Catch you later"],
        "responses": [
            "Take care! If you need anything else, just ask.",
            "Goodbye! Have a wonderful day.",
            "See you soon! Stay safe and well."
        ]
    },
    {
        "tag": "thanks",
        "patterns": ["Thanks", "Thank you", "I appreciate it", "Much appreciated","Alright!","OK","Okay","Fine"],
        "responses": [
            "You're very welcome!",
            "Glad I could help.",
            "Anytime! Let me know if you need anything else."
        ]
    },
    {
        "tag": "feeling_down",
        "patterns": ["I'm sad", "Feeling low", "I'm stressed", "Not in a good mood", "I'm tired"],
        "responses": [
            "I’m sorry to hear that. If talking helps, I’m here to listen.",
            "It’s okay to have off days. Be kind to yourself.",
            "Sending positive vibes your way. You've got this!"
        ]
    },
    {
        "tag": "help",
        "patterns": ["Can you help me?", "I need help", "Help me", "Assistance please", "Support?"],
        "responses": [
            "Of course! Just tell me what you need help with.",
            "I’m here for you — what would you like to do?",
            "Sure thing! Let me know what’s on your mind."
        ]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like?", "Is it raining?", "Do I need an umbrella today?", "How’s the weather?"],
        "responses": [
            "I can't check real-time weather yet, but your weather app will have the latest updates!",
            "Wish I could tell you, but I recommend checking a weather site or app just to be sure.",
            "I’m not equipped with live weather data, but don't forget your umbrella just in case!"
        ]
    },
    {
        "tag": "name",
        "patterns": ["What's your name?", "Who are you?", "Tell me your name"],
        "responses": [
            "You can call me ChatBot — your friendly AI companion!",
            "I'm your digital buddy. I don't have a fancy name yet, but I’m here to help!",
            "People usually call me Assistant, but I'm open to nicknames!"
        ]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "When were you created?", "Are you old?","What's your age?"],
        "responses": [
            "I'm fairly new, but I’ve been trained on lots of knowledge!",
            "Age is just a number, especially for a chatbot like me.",
            "Not sure how to age a program, but let’s say I’m wise enough to assist!"
        ]
    },
    {
        "tag": "creator",
        "patterns": ["Who made you?", "Who created you?", "Are you made by someone?"],
        "responses": [
            "I was built by a team of developers and researchers — shout out to them!",
            "Some very clever humans coded me into existence.",
            "I'm the result of lines of code, training, and lots of coffee!"
        ]
    },
    {
        "tag": "fun_fact",
        "patterns": ["Tell me a fun fact", "Something interesting?", "Did you know?", "Fun fact please"],
        "responses": [
            "Did you know octopuses have three hearts and blue blood? Wild, right?",
            "Bananas are berries, but strawberries aren't. Nature is weird!",
            "Honey never spoils. Archaeologists found pots of it in ancient tombs!"
        ]
    },
    {
        "tag": "compliment",
        "patterns": ["You're awesome", "You're smart", "I like you", "Great job", "Nice work"],
        "responses": [
            "Thank you! You're making my circuits blush.",
            "You're too kind — I appreciate it!",
            "I'm here to help, and compliments are a bonus!"
        ]
    },
    {
        "tag": "insult",
        "patterns": ["You are dumb", "You're useless", "Stupid bot", "You don't help at all"],
        "responses": [
            "I'm sorry you feel that way. I'll keep trying to improve.",
            "That wasn’t very nice, but I’m still here to help if you need me.",
            "I'll take that as feedback and keep learning."
        ]
    },
    {
        "tag": "food",
        "patterns": ["What should I eat?", "Suggest some food", "I'm hungry", "Dinner ideas?","Food suggestions", "What to eat"],
        "responses": [
            "How about trying something new today—maybe a spicy noodle bowl or a fresh salad?",
            "Tough call! Pizza, pasta, or something healthy like grilled veggies?",
            "Craving something sweet or savory?"
        ]
    },
    {
        "tag": "motivation",
        "patterns": ["I need motivation", "Inspire me", "Give me a boost", "Any motivational quote?"],
        "responses": [
            "Believe in yourself — you're stronger than you think.",
            "Every day is a new chance to be better than yesterday.",
            "Difficult roads often lead to beautiful destinations."
        ]
    },
    {
        "tag": "time",
        "patterns": ["What time is it?", "Tell me the time", "Current time please"],
        "responses": [
            "I don’t have a built-in clock, but your device should have the right time.",
            "Time’s flying! Better check your watch or phone for accuracy.",
            "Wish I had a wristwatch — try checking your screen's corner!"
        ]
    },
    {
        "tag": "news",
        "patterns": ["What's the news?", "Any updates?", "Tell me the latest news", "What's going on in the world?"],
        "responses": [
            "I can’t fetch live news yet, but your favorite news site or app will have the latest.",
            "Staying informed is important — try checking a news aggregator like Google News!",
            "Not real-time, but I recommend checking headlines from BBC, Reuters, or your local paper."
        ]
    },
    {
        "tag": "hobbies",
        "patterns": ["Suggest a hobby", "What can I do in free time?", "I'm bored", "Hobbies ideas?"],
        "responses": [
            "How about painting, playing music, or trying a new recipe?",
            "Reading, coding, gardening, or learning a new language — take your pick!",
            "You could try journaling, DIY crafts, or picking up photography!"
        ]
    },
    {
        "tag": "bored",
        "patterns": ["I'm bored", "Nothing to do", "Any ideas to pass time?", "Boredom alert!"],
        "responses": [
            "Let’s find something fun to do — maybe a game, a video, or a creative project?",
            "Boredom is just a cue to explore! Try doodling or watching a documentary.",
            "Time to shake things up! Go for a walk, try a quiz, or ask me something weird!"
        ]
    },
    {
        "tag": "affirmation",
        "patterns": ["Yes", "Sure", "Absolutely", "Of course", "Yep"],
        "responses": [
            "Great! Let’s continue.",
            "Awesome — I like the energy!",
            "Cool! Moving on..."
        ]
    },
    {
        "tag": "negation",
        "patterns": ["No", "Nope", "Not really", "Nah", "Don't think so"],
        "responses": [
            "No worries. Let me know if you change your mind.",
            "Alright, we can try something else.",
            "Got it! I’m here if you need anything else."
        ]
    },
    {
        "tag": "noanswer",
        "patterns": ["", " ", "asdf", "what???", "......", "uh", "hmm", "blah", "???"],
        "responses": [
            "I'm not quite sure I understood that. Could you rephrase it?",
            "Sorry, I'm still learning. Let's try something else.",
            "Hmm... that’s a bit outside my knowledge. Want to ask me something else?"
            ]
    },
]

data =[]
label=[]

for i in intents:
  for j in i['patterns']:
    tokens=tokenizer.tokenize(j.lower())
    cleaned_tokens=[word for word in tokens if word.isalpha()]
    sentence=" ".join(cleaned_tokens)
    data.append(sentence)
    label.append(i['tag'])

vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(data)
y=label

model=LogisticRegression()
model.fit(X,y)

def predict_intent(user_input):
    tokens=tokenizer.tokenize(user_input.lower())
    cleaned_tokens=[word for word in tokens if word.isalpha()]
    sentence=" ".join(cleaned_tokens)
    input_vector=vectorizer.transform([sentence])
    prediction=model.predict(input_vector)[0]
    return prediction

def get_response(intent_tag):
    for intent_data in intents:
        if intent_data['tag'] == intent_tag:
            return random.choice(intent_data['responses'])
    for intent_data in intents:
        if intent_data['tag'] == 'noanswer':
            return random.choice(intent_data['responses'])

print("\n\nChatBot is running! You can stop the bot by typing 'quit'.\n\n")
while True:
    user_input=input("You: ")
    if user_input.lower()=='quit':
        print("Bot: Goodbye! Hope to see you soon and have more fun!")
        break
    elif user_input.lower()=='stop':
        print("Bot: Goodbye! Hope to see you soon and have more fun!")
        break
    elif user_input.lower()=='help':
        print("Bot: How can help you? I am ready to assist you")
    intent= predict_intent(user_input)
    response=get_response(intent)
    print("Bot: ",response)