from src.multi_agent_ai.agents.agents import build_reader_agent, build_search_agent, writer_prompt, writer_chain,critic_chain

def run_search_pipeline(topic: str) -> dict:
    state = {"topic": topic}

    # --- Step 1: Search Agent (Find relevant URLs and snippets) ---
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent is finding sources...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Search the web for '{topic}'. Return a summary of key findings "
                        "along with the top 3-4 most relevant URLs to read in full."
                    ),
                }
            ]
        }
    )
    search_notes = search_result["messages"][-1].text
    state["search_notes"] = search_notes

    # --- Step 2: Reader Agent (Scrape full content from those URLs) ---
    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent is scraping key articles...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Based on the search results below for '{topic}', use scrape_url "
                        "on the top 2-3 most relevant URLs and extract detailed facts, data, "
                        f"and insights.\n\nSearch Results:\n{search_notes}"
                    ),
                }
            ]
        }
    )
    reader_notes = reader_result["messages"][-1].text
    state["reader_notes"] = reader_notes

    # Combine both search snippets and deep scraped content
    combined_research = (
        f"### Search Overview\n{search_notes}\n\n"
        f"### Deep Article Extractions\n{reader_notes}"
    )
    state["research"] = combined_research

    # --- Step 3: Writer Chain (Draft the structured report) ---
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    draft = writer_chain.invoke(
        {
            "topic": topic,
            "research": combined_research,
        }
    )
    state["draft"] = draft

    # --- Step 4: Critic Chain (Fact-check and review the draft) ---
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the draft...")
    print("=" * 50)

    critique = critic_chain.invoke(
        {
            "topic": topic,
            "research": combined_research,
            "draft": draft,
        }
    )
    state["critique"] = critique

    # --- Step 5: Revision Pass (Only runs if Critic flags issues) ---
    if "NEEDS REVISION" in critique.upper():
        print("\n" + "=" * 50)
        print("Step 5 - Writer is revising based on Critic feedback...")
        print("=" * 50)

        final_report = writer_chain.invoke(
            {
                "topic": topic,
                "research": (
                    f"{combined_research}\n\n"
                    f"--- Previous Draft ---\n{draft}\n\n"
                    f"--- Critic Feedback to Address ---\n{critique}"
                ),
            }
        )
        state["final_report"] = final_report
    else:
        state["final_report"] = draft

    return state


