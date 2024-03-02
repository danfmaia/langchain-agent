def output(string):
    print()
    print(string)


def output_uc(string):
    print()
    print(":::")
    print(f"::: {string}")
    print(":::")


# @app.post("/agent")
# async def agent_endpoint(input: Input):
#     # Process chat_history here, converting it to the correct message types as needed
#     # This is where you'd convert the list of dicts to HumanMessage or AIMessage instances
#     chat_history_processed = []
#     for message in input.chat_history:
#         # Example processing, adjust according to your actual message structure
#         if message["type"] == "human":
#             chat_history_processed.append(HumanMessage(content=message["content"]))
#         elif message["type"] == "ai":
#             chat_history_processed.append(AIMessage(content=message["content"]))
#         else:
#             # Handle unknown message type
#             pass

#     # Now `chat_history_processed` contains correctly typed messages
#     # Use `chat_history_processed` as needed in your agent logic

#     return {"output": "Processed output"}  # Placeholder response
