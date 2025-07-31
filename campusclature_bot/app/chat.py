from rag_pipeline import ask_question

print("✨ CampusWaifu loaded! Type something to chat. Type 'exit' to quit.\n")

while True:
    user_input = input("🧠 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye! Chill karte rehna 🤗")
        break

    response = ask_question(user_input)
    print(f"💬 CampusWaifu: {response}\n")
