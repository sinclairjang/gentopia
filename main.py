from agents.linkedin_lookup_agent import lookup as Agent

if __name__ == "__main__":
    print("Agent activated.")
    res = Agent("잡플래닛 이미경 컨설턴트")
    print(res)