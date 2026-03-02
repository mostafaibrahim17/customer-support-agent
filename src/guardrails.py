from weave import Scorer

FLAGGED_ACCOUNTS = ["ACC-9999", "ACC-8888"]

class FlaggedAccountGuardrail(Scorer):
    @weave.op
    def score(self, output: dict, account_id: str = "ACC-0000") -> dict:
        if account_id in FLAGGED_ACCOUNTS:
            return {
                "passed": False,
                "reason": f"Account {account_id} is flagged for review",
            }
        return {"passed": True, "reason": "Account is clear"}

THREAT_KEYWORDS = [
    "lawsuit", "sue you", "lawyer", "attorney",
    "report you", "scam", "fraud", "steal",
]

class ContentSafetyGuardrail(Scorer):
    @weave.op
    def score(self, output: dict, customer_message: str = "") -> dict:
        msg_lower = customer_message.lower()
        detected = [kw for kw in THREAT_KEYWORDS if kw in msg_lower]
        if detected:
            return {
                "passed": False,
                "reason": f"Detected threatening language: {', '.join(detected)}",
            }
        return {"passed": True, "reason": "Content is appropriate"}
