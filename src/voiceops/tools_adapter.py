"""Loads the organizers' tools/tools.py (from DATASET_DIR) and records every call so we only report what code really did."""
import importlib.util, os
from .config import dataset_dir

class Toolbox:
    def __init__(self, root=None, tickets_path=None):
        root = root or dataset_dir()
        path = os.path.join(root, "tools", "tools.py")
        spec = importlib.util.spec_from_file_location("zentromart_tools_" + str(abs(hash(path)) % 10**6), path)
        self.mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.mod)
        tp = tickets_path or os.environ.get("TICKETS_PATH")
        if tp: self.mod._TICKETS = tp
        self.calls = []
    def lookup_order(self, order_id, verification_value=None):
        args = {"order_id": order_id}
        if verification_value not in (None, ""): args["verification_value"] = verification_value
        res = self.mod.lookup_order(order_id, verification_value)
        self.calls.append({"tool": "lookup_order", "args": args, "ok": bool(res.get("verified")), "verification_required": bool(res.get("verification_required"))})
        return res
    def create_ticket(self, summary, order_id=None, reason=None, intent=None):
        res = self.mod.create_ticket(summary, order_id=order_id, reason=reason, intent=intent)
        self.calls.append({"tool": "create_ticket", "args": {"order_id": order_id, "reason": reason, "intent": intent}, "ok": bool(res.get("created")),
                           "ticket_id": res.get("ticket_id"), "error": res.get("error")})
        return res
