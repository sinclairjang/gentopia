from agents.linkedin_lookup_agent import lookup as Agent

if __name__ == "__main__":
    print("Agent activated.")
    res = Agent("AI 스토리텔러 김우정")
    print(res)