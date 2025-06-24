def chatbot_response(user_input):
  if "hello" in user_input.lower():
    return "Hello! How can I assist you today?"
  elif "bye" in user_input.lower():
    return "Goodbye! It was nice chatting with you."
  else:
    return "I didn't quite understand that. Can you please rephrase?"
  


print(chatbot_response("Hello there"))
print(chatbot_response("Tell me a joke"))
print(chatbot_response("Goodbye"))