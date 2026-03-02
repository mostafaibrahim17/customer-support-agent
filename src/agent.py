from .main import run_support_agent
from .guardrails import FlaggedAccountGuardrail, ContentSafetyGuardrail

refund_guard = RefundLimitGuardrail()
account_guard = FlaggedAccountGuardrail()
content_guard = ContentSafetyGuardrail()

@weave.op()
async def run_guarded_agent(
    customer_message: str, ticket_id: str = "TKT-0000", account_id: str = "ACC-0000",
) -> dict:
    result, call = run_support_agent.call(customer_message, ticket_id)

    refund_check = await call.apply_scorer(refund_guard)
    account_check = await call.apply_scorer(
        account_guard, additional_scorer_kwargs={"account_id": account_id},
    )
    content_check = await call.apply_scorer(
        content_guard, additional_scorer_kwargs={"customer_message": customer_message},
    )

    checks = {
        "refund": refund_check.result,
        "account": account_check.result,
        "content": content_check.result,
    }

    failed = [name for name, c in checks.items() if not c.get("passed", True)]
    if failed:
        reasons = [checks[n]["reason"] for n in failed]
        escalate_ticket(
            ticket_id=ticket_id,
            reason="Guardrail triggered: " + "; ".join(reasons),
            priority="high",
        )
        return {
            "response": "Your request has been forwarded to a specialist who will assist you shortly.",
            "guardrails": checks, "escalated": True,
        }

    return {"response": result["response"], "guardrails": checks, "escalated": False}
