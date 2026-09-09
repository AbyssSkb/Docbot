import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pipeline


CHUNK = {
    "chunk_id": "chunk_" + "a" * 64,
    "source": "policy.txt",
    "section": "退款",
    "text": "签收后七天内可以退款。",
}
OTHER = {**CHUNK, "chunk_id": "chunk_" + "b" * 64, "text": "退款需提供订单号。"}
ANSWER = f"签收后七天内可以退款。[{CHUNK['chunk_id']}]"


def fake_client(*contents):
    client = Mock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        for text in contents
    ]
    return client


def reply(text=ANSWER, action="answer"):
    return json.dumps({"action": action, "query" if action == "search" else "answer": text}, ensure_ascii=False)


SEARCH = reply("退款", "search")


class AgentTest(unittest.TestCase):
    def run_agent(self, client, **kwargs):
        return pipeline.run_agent("退款期限？", [CHUNK, OTHER], {}, client, "test-model", **kwargs)

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_first_search_is_generated_and_progress_precedes_work(self, retrieve):
        client = fake_client(reply("签收 退款 天数", "search"), reply("七天。[chunk_1]"))
        events = self.run_agent(client)
        self.assertEqual(next(events)["status"], "generating")
        client.chat.completions.create.assert_not_called()
        self.assertEqual(next(events)["status"], "llm_refusal")
        search = next(events)
        self.assertEqual(search["retrieval_query"], "签收 退款 天数")
        retrieve.assert_not_called()
        self.assertEqual(next(events)["status"], "generating")
        result = next(events)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], f"七天。[{CHUNK['chunk_id']}]")
        self.assertEqual(result["cited_chunks"], [CHUNK])
        self.assertEqual(list(events), [])

    @patch("pipeline.retrieve", side_effect=[[CHUNK], [CHUNK, OTHER], [OTHER]])
    def test_all_failed_evidence_is_shared_by_search_and_answer_without_duplicates(self, retrieve):
        history = [{"role": "user", "content": f"历史消息{i}"} for i in range(25)]
        original_history = deepcopy(history)
        client = fake_client(reply("退款期限", "search"), reply("订单要求", "search"), reply("签收规则", "search"), reply())
        events = list(self.run_agent(client, history=history))
        results = [event for event in events if "answer" in event]
        self.assertEqual([item["new_chunk_count"] for item in results], [0, 1, 1, 0])
        self.assertEqual(results[-1]["context_chunk_ids"], [CHUNK["chunk_id"], OTHER["chunk_id"]])
        self.assertEqual(results[-1]["retrieved_chunk_ids"], [OTHER["chunk_id"]])
        self.assertEqual(results[-1]["answer"], ANSWER)
        self.assertEqual(results[-1]["cited_chunks"], [CHUNK])
        requests = client.chat.completions.create.call_args_list
        for request in requests:
            content = request.kwargs["messages"][1]["content"]
            self.assertIn("历史消息0", content)
            self.assertIn("历史消息24", content)
            self.assertIn("退款期限？", content)
        for index in (2, 3):
            content = requests[index].kwargs["messages"][1]["content"]
            self.assertEqual(content.count(CHUNK["text"]), 1)
            self.assertEqual(content.count(OTHER["text"]), 1)
        planner = requests[2].kwargs["messages"][1]["content"]
        self.assertIn("退款期限", planner)
        self.assertIn("订单要求", planner)
        self.assertIn("chunk_1", planner)
        self.assertEqual(history, original_history)

    @patch("pipeline.retrieve", return_value=[])
    def test_history_cited_evidence_answers_without_any_search(self, retrieve):
        history = [{"role": "assistant", "content": ANSWER, "cited_chunks": [CHUNK]}]
        client = fake_client(reply("七天。[chunk_1]"))
        result = list(self.run_agent(client, history=history))[-1]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["cited_chunk_ids"], [CHUNK["chunk_id"]])
        self.assertEqual(result["cited_chunks"], [CHUNK])
        self.assertEqual(result["attempt"], 0)
        retrieve.assert_not_called()
        content = client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
        self.assertIn(CHUNK["text"], content)

    @patch("pipeline.retrieve")
    def test_accumulated_context_is_not_truncated_to_ten(self, retrieve):
        first = [{**CHUNK, "chunk_id": f"chunk_{i:064x}", "text": f"线索{i}"} for i in range(10)]
        retrieve.side_effect = [first, [CHUNK]]
        client = fake_client(reply("第一条查询", "search"), reply("第二条查询", "search"), reply("七天。[chunk_11]"))
        result = list(self.run_agent(client))[-1]
        self.assertEqual(result["answer"], f"七天。[{CHUNK['chunk_id']}]")
        self.assertEqual(len(result["context_chunk_ids"]), 11)
        self.assertEqual(result["cited_chunks"], [CHUNK])
        content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("线索0", content)
        self.assertIn(CHUNK["text"], content)

    @patch("pipeline.retrieve")
    def test_history_citations_share_current_aliases_without_mutating_saved_history(self, retrieve):
        history = [{
            "role": "assistant",
            "content": f"期限七天。[{CHUNK['chunk_id']}] 还需订单号。[{OTHER['chunk_id']}]",
            "cited_chunks": [OTHER, CHUNK],
        }]
        original = deepcopy(history)
        client = fake_client(reply("七天。[chunk_2]"))
        result = list(self.run_agent(client, history=history))[-1]
        content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("期限七天。[chunk_2] 还需订单号。[chunk_1]", content)
        self.assertNotIn(CHUNK["chunk_id"], content)
        self.assertNotIn(OTHER["chunk_id"], content)
        self.assertEqual(result["cited_chunk_ids"], [CHUNK["chunk_id"]])
        self.assertEqual(history, original)
        retrieve.assert_not_called()

    @patch("pipeline.retrieve", side_effect=[[], [CHUNK]])
    def test_empty_retrieval_still_allows_model_to_decide_next_step(self, retrieve):
        client = fake_client(SEARCH, reply("签收天数", "search"), reply())
        events = list(self.run_agent(client))
        self.assertTrue(any(event["status"] == "llm_refusal" and event["attempt"] == 1 for event in events))
        self.assertEqual(events[-1]["answer"], ANSWER)
        self.assertEqual(client.chat.completions.create.call_count, 3)

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_invalid_citation_retries_without_repeating_search(self, retrieve):
        client = fake_client(SEARCH, reply("七天 [chunk_missing]"), reply())
        events = list(self.run_agent(client))
        self.assertEqual(events[-1]["answer"], ANSWER)
        self.assertEqual(events[-1]["attempt"], 2)
        self.assertEqual(events[-1]["search_count"], 1)
        content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("chunk_missing", content)
        self.assertIn("引用", content)
        self.assertEqual(retrieve.call_count, 1)

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_default_budget_is_five_searches_and_custom_budget_one_stops(self, retrieve):
        for budget in (None, 1):
            with self.subTest(budget=budget):
                retrieve.reset_mock()
                count = budget or 5
                client = fake_client(*([SEARCH] * (count + 1)))
                events = list(self.run_agent(client, **({"max_attempts": budget} if budget else {})))
                self.assertEqual(events[-1]["attempt"], count)
                self.assertEqual(events[-1]["answer"], "无答案")
                self.assertEqual(events[-1]["cited_chunks"], [])
                self.assertEqual(retrieve.call_count, count)
                self.assertEqual(client.chat.completions.create.call_count, count + 1)
                json.dumps(events)

    def test_invalid_budget_rejected(self):
        for budget in (0, -1, 1.5, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                list(self.run_agent(fake_client(), max_attempts=budget))

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_service_errors_and_empty_queries_do_not_trigger_another_search(self, retrieve):
        client = fake_client()
        client.chat.completions.create.side_effect = RuntimeError("service unavailable")
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            list(self.run_agent(client))
        retrieve.assert_not_called()
        result = list(self.run_agent(fake_client(SEARCH, reply("  ", "search"), reply())))[-1]
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(retrieve.call_count, 1)

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_each_single_query_uses_selected_retrieval_mode(self, retrieve):
        ranker = Mock()
        result = list(self.run_agent(fake_client(SEARCH, reply("签收天数", "search"), reply()), ranker=ranker))[-1]
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual([call.args[0] for call in retrieve.call_args_list], ["退款", "签收天数"])
        self.assertTrue(all(call.args[3] is ranker for call in retrieve.call_args_list))

    @patch("pipeline.retrieve", side_effect=[[OTHER], [CHUNK], []])
    def test_web_keeps_trace_but_passes_only_cited_evidence_to_next_turn(self, retrieve):
        from streamlit.testing.v1 import AppTest

        client = fake_client(reply("失败查询", "search"), reply("签收天数", "search"), reply(), reply())
        manifest = {"embedding_models": [], "counts": {"documents": 1, "chunks": 2}}
        with (
            patch("create_index.sha256_file", return_value="agent-memory-test"),
            patch("pipeline.load_index_bundle", return_value=(manifest, [CHUNK, OTHER])),
            patch("pipeline.load_models", return_value=({}, None)),
            patch("openai.OpenAI", return_value=client),
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py"))
            app.run()
            app.sidebar.number_input[0].set_value(2).run()
            app.chat_input[0].set_value("退款期限？").run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual([item.value for item in app.code], ["失败查询", "签收天数"])
            message = app.session_state["messages"][-1]
            self.assertEqual(message["cited_chunks"], [CHUNK])
            self.assertEqual(len(message["attempts"]), 3)
            app.run()
            self.assertEqual([item.value for item in app.code], ["失败查询", "签收天数"])
            app.chat_input[0].set_value("再确认一下期限？").run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state["messages"][-1]["content"], ANSWER)
            for request in client.chat.completions.create.call_args_list[3:]:
                content = request.kwargs["messages"][1]["content"]
                self.assertIn(CHUNK["text"], content)
                self.assertNotIn(OTHER["text"], content)
                self.assertNotIn("失败查询", content)
                self.assertNotIn("llm_refusal", content)
            self.assertEqual(retrieve.call_count, 2)


    @patch("pipeline.retrieve")
    def test_model_answer_needs_no_search_or_citation(self, retrieve):
        # No keyword routing: even the helper's document question obeys the LLM action.
        client = fake_client(reply("我是 Docbot，可以帮你查阅文档。"))
        result = list(self.run_agent(client))[-1]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["attempt"], 0)
        self.assertEqual(result["cited_chunks"], [])
        self.assertEqual(result["answer"], "我是 Docbot，可以帮你查阅文档。")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        retrieve.assert_not_called()

    @patch("pipeline.retrieve")
    def test_uncited_answer_is_accepted_without_semantic_classification(self, retrieve):
        result = list(self.run_agent(fake_client(reply("七天"))))[-1]
        self.assertEqual(result["answer"], "七天")
        self.assertEqual(result["cited_chunks"], [])
        retrieve.assert_not_called()

    @patch("pipeline.retrieve")
    def test_malformed_decision_is_retried_with_feedback(self, retrieve):
        for raw in ('', '{}', '[]', 'not json', '{"action":"answer","answer":""}',
                    '{"action":"search","answer":"invented"}',
                    '{"action":"unknown","answer":"hello"}'):
            with self.subTest(raw=raw):
                client = fake_client(raw, reply("已修正"))
                result = list(self.run_agent(client))[-1]
                self.assertEqual(result["answer"], "已修正")
                self.assertEqual(result["attempt"], 1)
                self.assertEqual(result["search_count"], 0)
                self.assertEqual(result["retry_count"], 1)
                content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
                self.assertIn("error", content)
                self.assertIn("raw_response", content)
        retrieve.assert_not_called()

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_search_and_repairs_share_one_budget(self, retrieve):
        client = fake_client(SEARCH, "bad json", SEARCH, reply())
        result = list(self.run_agent(client, max_attempts=3))[-1]
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual(result["attempt"], 3)
        self.assertEqual(result["search_count"], 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(client.chat.completions.create.call_count, 4)

    @patch("pipeline.retrieve")
    def test_broken_outputs_stop_at_shared_limit(self, retrieve):
        client = fake_client(*(["bad json"] * 6))
        result = list(self.run_agent(client))[-1]
        self.assertEqual(result["answer"], "无答案")
        self.assertEqual(result["attempt"], 5)
        self.assertEqual(result["retry_count"], 5)
        self.assertEqual(client.chat.completions.create.call_count, 6)
        retrieve.assert_not_called()

    @patch("pipeline.retrieve", return_value=[CHUNK])
    def test_repair_uses_last_slot_and_prevents_an_extra_search(self, retrieve):
        client = fake_client(SEARCH, "bad json", SEARCH)
        result = list(self.run_agent(client, max_attempts=2))[-1]
        self.assertEqual(result["answer"], "无答案")
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(client.chat.completions.create.call_count, 3)

    @patch("pipeline.retrieve")
    def test_web_zero_search_chat_is_saved_and_displayed_without_search_counter(self, retrieve):
        from streamlit.testing.v1 import AppTest

        client = fake_client("bad json", reply("我是 Docbot。"))
        manifest = {"embedding_models": [], "counts": {"documents": 1, "chunks": 2}}
        with (
            patch("create_index.sha256_file", return_value="agent-chat-test"),
            patch("pipeline.load_index_bundle", return_value=(manifest, [CHUNK, OTHER])),
            patch("pipeline.load_models", return_value=({}, None)),
            patch("openai.OpenAI", return_value=client),
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py"))
            app.run()
            app.chat_input[0].set_value("你是谁").run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(list(app.error), [])
            self.assertEqual(list(app.code), [])
            self.assertEqual(app.session_state["messages"][-1]["content"], "我是 Docbot。")
            self.assertEqual(app.session_state["messages"][-1]["cited_chunks"], [])
            self.assertTrue(any("未检索" in item.label for item in app.status))
            self.assertTrue(any("重试 1 次" in item.label for item in app.status))
            self.assertTrue(any("JSON" in item.value for item in app.caption))
            app.run()
            self.assertEqual(list(app.exception), [])
            retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
