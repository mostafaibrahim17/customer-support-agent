import os
import json
import asyncio

import weave
from openai import OpenAI
from weave import Scorer
from datasets import load_dataset

# Initialize Weave
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your-api-key")  # Replace with your API key
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
weave.init("customer-support-agent")

# Set up the W&B Inference client
client = OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=WANDB_API_KEY,
)
AGENT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
print(f"✓ W&B Inference client ready | Model: {AGENT_MODEL}")

# Load the dataset
bitext_ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)
print(f"✓ Loaded {len(bitext_ds)} messages")
print(f"  Categories: {len(set(bitext_ds['category']))}")
print(f"  Intents: {len(set(bitext_ds['intent']))}")

# Knowledge base
KNOWLEDGE_BASE = {
    "refund_policy": (
       "Refunds are available within 30 days of delivery for undamaged items. "
      "For items still in transit, customers must wait until 10 business days "
        "past the expected delivery date before requesting a refund."
    ),
    "shipping_times": (
        "Standard shipping takes 5 to 7 business days. Express shipping takes "
      "2 to 3 business days. International orders take 10 to 15 business days."
    ),
    # Also includes: return_process, damaged_items, account_issues
}

@weave.op()
def search_knowledge_base(query: str) -> dict:
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if any(word in query_lower for word in topic.split("_")):
            results.append({"topic": topic, "content": content})
    if not results:
        results.append({
            "topic": "general",
            "content": "No specific policy found. Please escalate to a human agent."
        })
    return {"results": results, "query": query}

# Order database
ORDERS_DB = {
    "ORD-1001": {
        "status": "delivered", "item": "Wireless Headphones",
        "delivered_date": "2026-01-28", "amount": 79.99,
    },
    "ORD-1002": {
        "status": "in_transit", "item": "Running Shoes",
        "tracking": "1Z999AA10123456784", "eta": "2026-02-12", "amount": 129.99,
    },
    # Also includes: ORD-1003 (processing), ORD-1004 (delivered), ORD-1005 (cancelled)
}

@weave.op()
def lookup_order(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if order:
        return {"found": True, "order_id": order_id, **order}
    return {"found": False, "order_id": order_id, "error": "Order not found"}

# Tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search product FAQ and policy documents for answers "
                "to customer questions about refunds, shipping, returns, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about products or policies",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # lookup_order: takes order_id (str), returns order status and details
    # issue_refund: takes order_id (str), amount (number), reason (str)
    # escalate_ticket: takes ticket_id (str), reason (str), priority (low/medium/high)
]

# System prompt
SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company.\n\nYour responsibilities:\n1. Answer product and policy questions using the knowledge base.\n2. Look up order status when customers ask about their orders.\n3. Process refunds when eligible according to company policy.\n4. Escalate to human agents when a case is too complex or sensitive.\n\nRules:\n- Always look up the order before discussing its status.\n- Check the refund policy before processing any refund.\n- Never process a refund without first verifying the order exists.\n- If a customer is abusive or the situation is complex, escalate immediately.\n- Be concise, empathetic, and professional."""

TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "lookup_order": lookup_order,
    "issue_refund": issue_refund,
    "escalate_ticket": escalate_ticket,
}

@weave.op()
def run_support_agent(customer_message: str, ticket_id: str = "TKT-0000", system_prompt: str = None) -> dict:
    if system_prompt is None: system_prompt = SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": customer_message},
    ]
    tools_called = []
    max_iterations = 5  # Prevents infinite tool-calling loops

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "escalate_ticket" and "ticket_id" not in func_args:
                    func_args["ticket_id"] = ticket_id

                tool_func = TOOL_FUNCTIONS.get(func_name)
                result = tool_func(**func_args) if tool_func else {"error": f"Unknown tool: {func_name}"}

                tools_called.append({"tool": func_name, "args": func_args, "result": result})
                messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "name": func_name, "content": json.dumps(result),
                })
        else:
            return {"response": assistant_msg.content, "tools_called": tools_called, "iterations": iteration + 1}

# Example usage
result = run_support_agent(
    customer_message="My order ORD-1002 never arrived, and I want a refund.",
    ticket_id="TKT-2001",
)
print("Agent response:")
print(result["response"])
print(f"\nTools called: {[t['tool'] for t in result['tools_called']]}")
print(f"Iterations: {result['iterations']}")
