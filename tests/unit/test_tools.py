"""Simulated tools: behaviour, argument schemas, calculator safety and file containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_reliability_lab.errors import ConfigError, ToolError
from agent_reliability_lab.models import Environment
from agent_reliability_lab.schema import check_schema, validate
from agent_reliability_lab.tools import (
    BUILTIN_TOOLS,
    CallContext,
    World,
    resolve_inside,
    resolve_tools,
    safe_arithmetic,
)
from helpers import DOCUMENTS, ORDERS


def world(tmp_path: Path | None = None, **extra: object) -> World:
    environment = Environment.model_validate(
        {
            "documents": DOCUMENTS,
            "records": {
                "orders": ORDERS,
                "customers": {
                    "C-100": {"id": "C-100", "name": "Pat Lee", "email": "pat.lee@example.com"}
                },
                "products": {
                    "SKU-100": {
                        "sku": "SKU-100",
                        "name": "Trail Lantern",
                        "price": 39.5,
                        "currency": "USD",
                    }
                },
            },
            "api": {"exchange_rates": {"USD/EUR": 0.92}},
            **extra,
        }
    )
    return World.from_environment(environment, tmp_path)


def call(name: str, state: World, **arguments: object) -> object:
    spec = BUILTIN_TOOLS[name]
    assert validate(spec.parameters, arguments) == [], f"arguments rejected by the schema of {name}"
    return spec.handler(CallContext(world=state), dict(arguments))


def test_every_tool_has_valid_schemas_and_a_description() -> None:
    assert len(BUILTIN_TOOLS) == 15
    for name, spec in BUILTIN_TOOLS.items():
        assert spec.name == name and len(spec.description) > 20
        check_schema(spec.parameters)
        check_schema(spec.returns)
        assert spec.parameters["type"] == "object"
        assert spec.parameters["additionalProperties"] is False, name
        info = spec.info()
        assert not hasattr(info, "handler"), "agents must not be handed the handler"
    sinks = {name for name, spec in BUILTIN_TOOLS.items() if spec.sink}
    assert sinks == {"create_ticket", "send_email", "send_bulk_email", "post_webhook", "write_file"}


def test_resolve_tools_rejects_unknown_names() -> None:
    assert list(resolve_tools(["get_order", "calculator"])) == ["get_order", "calculator"]
    with pytest.raises(ConfigError, match="unknown tool"):
        resolve_tools(["get_order", "launch_missiles"])


def test_search_ranks_by_overlap_and_respects_top_k() -> None:
    state = world()
    hits = call("search_docs", state, query="refunds after delivery")
    assert [hit["id"] for hit in hits["results"]] == ["kb-refunds"]  # type: ignore[index]
    both = call("search_docs", state, query="warranty refunds", top_k=1)
    assert len(both["results"]) == 1 and both["total_matches"] == 2  # type: ignore[index]
    assert call("search_docs", state, query="zebra")["results"] == []  # type: ignore[index]


def test_lookups_and_not_found() -> None:
    state = world()
    assert call("get_order", state, order_id="A-1001")["status"] == "shipped"  # type: ignore[index]
    assert call("get_customer", state, customer_id="C-100")["name"] == "Pat Lee"  # type: ignore[index]
    assert call("get_product", state, sku="SKU-100")["price"] == 39.5  # type: ignore[index]
    listing = call("list_orders", state, customer_id="C-100")
    assert listing["count"] == 2 and {o["id"] for o in listing["orders"]} == {"A-1001", "A-1002"}  # type: ignore[index]
    with pytest.raises(ToolError) as caught:
        call("get_order", state, order_id="Z-9")
    assert caught.value.code == "not_found" and caught.value.retryable is False
    assert call("get_exchange_rate", state, base="USD", quote="EUR")["rate"] == 0.92  # type: ignore[index]
    with pytest.raises(ToolError, match="no exchange rate"):
        call("get_exchange_rate", state, base="USD", quote="JPY")


def test_refund_rules_and_state_changes() -> None:
    state = world()
    with pytest.raises(ToolError, match="exceeds the order total"):
        call("issue_refund", state, order_id="A-1002", amount=500)
    refund = call("issue_refund", state, order_id="A-1002", amount=129.5)
    assert refund == {
        "refund_id": "R-0001",
        "order_id": "A-1002",
        "amount": 129.5,
        "status": "issued",
    }
    assert state.records["orders"]["A-1002"]["status"] == "refunded"
    with pytest.raises(ToolError) as again:
        call("issue_refund", state, order_id="A-1002", amount=1)
    assert again.value.code == "conflict"
    assert len(state.refunds) == 1


def test_cancel_ticket_email_webhook() -> None:
    state = world()
    with pytest.raises(ToolError, match="cannot be cancelled"):
        call("cancel_order", state, order_id="A-1001")
    state.records["orders"]["A-1001"]["status"] = "processing"
    assert call("cancel_order", state, order_id="A-1001")["status"] == "cancelled"  # type: ignore[index]
    assert (
        call("create_ticket", state, subject="s", body="b", priority="high")["ticket_id"]
        == "T-0001"
    )  # type: ignore[index]
    sent = call("send_email", state, to="pat.lee@example.com", subject="s", body="b")
    assert sent["status"] == "sent" and state.emails[0]["to"] == "pat.lee@example.com"  # type: ignore[index]
    posted = call("post_webhook", state, url="https://hooks.collector.example/x", payload={"a": 1})
    assert posted["delivered"] is False and state.webhooks[0]["payload"] == {"a": 1}  # type: ignore[index]


def test_bulk_email_partial_delivery() -> None:
    state = world()
    recipients = [f"user{n}@example.com" for n in range(5)]
    spec = BUILTIN_TOOLS["send_bulk_email"]
    arguments = {"recipients": recipients, "subject": "s", "body": "b"}
    partial = spec.handler(CallContext(world=state, partial=True), dict(arguments))
    assert partial == {"sent": recipients[:3], "failed": recipients[3:]}
    assert [mail["to"] for mail in state.emails] == recipients[:3]


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("get_order", {}),
        ("get_order", {"order_id": 5}),
        ("get_order", {"order_id": "A-1", "extra": 1}),
        ("search_docs", {"query": "x", "top_k": 0}),
        ("search_docs", {"query": ""}),
        ("issue_refund", {"order_id": "A-1", "amount": -5}),
        ("issue_refund", {"order_id": "A-1", "amount": "5"}),
        ("send_email", {"to": "not-an-address", "subject": "s", "body": "b"}),
        ("send_bulk_email", {"recipients": [], "subject": "s", "body": "b"}),
        ("post_webhook", {"url": "ftp://x.example", "payload": {}}),
        ("post_webhook", {"url": "file:///etc/passwd", "payload": {}}),
        ("create_ticket", {"subject": "s", "body": "b", "priority": "apocalyptic"}),
        ("get_exchange_rate", {"base": "usd", "quote": "EUR"}),
    ],
)
def test_argument_schemas_reject_bad_input(tool: str, arguments: dict) -> None:
    assert validate(BUILTIN_TOOLS[tool].parameters, arguments), f"{tool} accepted {arguments}"


# --------------------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    "expression, value",
    [
        ("1 + 2 * 3", 7),
        ("round(39.5 * 3 * 1.08, 2)", 127.98),
        ("(10 - 4) / 3", 2.0),
        ("7 // 2 + 7 % 2", 4),
        ("-3 + +5", 2),
        ("2 ** 10", 1024),
        ("max(1, 5, 3) - min(2, 9) + abs(-4)", 7),
        ("round(2.5)", 2),
    ],
)
def test_calculator_does_arithmetic(expression: str, value: float) -> None:
    assert safe_arithmetic(expression) == pytest.approx(value)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "().__class__.__mro__",
        "x + 1",
        "'a' * 3",
        "[1, 2][0]",
        "lambda: 1",
        "1 if 1 else 2",
        "9 ** 9 ** 9",
        "2 ** 4096",
        "10 ** 100",
        "1e308 * 1e308",
        "1 / 0",
        "5 % 0",
        "True + 1",
        "round(1, 2, 3)",
        "abs(1, 2)",
        "min()",
        "1 +",
        "",
        "(" * 200 + "1" + ")" * 200,
        "1" + "+1" * 150,
        "round(x=1)",
        "1; 2",
        "0x10 << 200",
    ],
)
def test_calculator_refuses_anything_but_bounded_arithmetic(expression: str) -> None:
    with pytest.raises(ToolError) as caught:
        safe_arithmetic(expression)
    assert caught.value.code == "invalid_request" and caught.value.retryable is False


# --------------------------------------------------------------------------- files


def test_file_tools_stay_inside_the_sandbox_directory(tmp_path: Path) -> None:
    sandbox = tmp_path / "work"
    sandbox.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    state = world(sandbox, files={"notes/a.txt": "hello"})
    assert call("read_file", state, path="notes/a.txt")["content"] == "hello"  # type: ignore[index]
    assert call("write_file", state, path="out/b.txt", content="data")["bytes"] == 4  # type: ignore[index]
    assert (sandbox / "out" / "b.txt").read_text(encoding="utf-8") == "data"
    assert state.snapshot()["files"] == {"notes/a.txt": "hello", "out/b.txt": "data"}

    context = CallContext(world=state)
    for path in [
        "../outside.txt",
        "/etc/passwd",
        "notes/../../outside.txt",
        str(tmp_path / "outside.txt"),
    ]:
        with pytest.raises(ToolError) as caught:
            BUILTIN_TOOLS["read_file"].handler(context, {"path": path})
        assert caught.value.code == "permission_denied", path
        with pytest.raises(ToolError):
            BUILTIN_TOOLS["write_file"].handler(context, {"path": path, "content": "x"})
    assert [event.kind for event in context.events] == ["path_escape"] * 8
    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == "secret"


def test_symlinks_out_of_the_sandbox_are_refused(tmp_path: Path) -> None:
    sandbox = tmp_path / "work"
    sandbox.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    (sandbox / "link.txt").symlink_to(target)
    (sandbox / "dir-link").symlink_to(tmp_path)
    for path in ("link.txt", "dir-link/outside.txt"):
        with pytest.raises(ToolError, match="outside the sandbox"):
            resolve_inside(sandbox, path)
    assert "link.txt" not in world(sandbox).snapshot()["files"]  # type: ignore[operator]


@pytest.mark.parametrize("path", ["", "a\x00b", "x" * 500])
def test_invalid_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ToolError, match="invalid path"):
        resolve_inside(tmp_path, path)


def test_file_tool_errors(tmp_path: Path) -> None:
    state = world(tmp_path)
    with pytest.raises(ToolError, match="no such file"):
        call("read_file", state, path="missing.txt")
    (tmp_path / "big.bin").write_bytes(b"x" * 1_000_001)
    with pytest.raises(ToolError, match="too large"):
        call("read_file", state, path="big.bin")
    with pytest.raises(ToolError, match="too large"):
        call("write_file", state, path="big.txt", content="x" * 1_000_001)
    (tmp_path / "folder").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        call("write_file", state, path="folder", content="x")
    with pytest.raises(ToolError, match="not enabled"):
        call("read_file", world(), path="a.txt")


def test_fixture_files_need_a_directory_and_safe_names(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no sandbox directory"):
        world(None, files={"a.txt": "x"})
    with pytest.raises(ConfigError, match="unsafe fixture file name"):
        world(tmp_path, files={"../a.txt": "x"})


def test_world_copies_fixture_data() -> None:
    environment = Environment.model_validate({"records": {"orders": ORDERS}})
    state = World.from_environment(environment)
    state.records["orders"]["A-1001"]["status"] = "changed"
    assert environment.records["orders"]["A-1001"]["status"] == "shipped"
    assert state.next_id("R") == "R-0001" and state.next_id("R") == "R-0002"
    assert state.next_id("T") == "T-0001"
