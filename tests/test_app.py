"""Offline control-flow checks: python -m unittest discover -s tests -v.

These tests do not validate real Chroma retrieval, Streamlit rendering or Azure
answer quality. External dependencies are mocked; no credentials are needed.
"""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class StopRun(Exception):
    """Model Streamlit's stop-the-current-run behavior."""


class SessionState(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class EvidenceGuardTests(unittest.TestCase):
    def run_app(self, count=2, documents=None, question="What is RAG?"):
        collection = MagicMock()
        collection.count.return_value = count
        collection.query.return_value = {"documents": documents}
        chroma = MagicMock()
        chroma.PersistentClient.return_value.get_or_create_collection.return_value = collection
        streamlit = MagicMock()
        streamlit.session_state = SessionState()
        streamlit.chat_input.return_value = question
        streamlit.stop.side_effect = StopRun
        openai = MagicMock()
        openai.AzureOpenAI.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="An answer from evidence."))]
        )
        modules = {"chromadb": chroma, "streamlit": streamlit,
                   "openai": openai, "dotenv": MagicMock()}
        with patch.dict("sys.modules", modules):
            try:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"))
            except StopRun:
                pass
        return collection, streamlit, openai.AzureOpenAI

    def test_empty_collection_stops_before_query_or_azure(self):
        collection, ui, azure = self.run_app(count=0)
        collection.query.assert_not_called()
        azure.assert_not_called()
        ui.chat_input.assert_not_called()
        self.assertIn("python ingest.py", ui.info.call_args.args[0])

    def test_empty_or_blank_retrieval_never_calls_azure(self):
        for documents in (None, [], [[]], [None], [[None, "", " \n "]]):
            with self.subTest(documents=documents):
                _, ui, azure = self.run_app(documents=documents)
                azure.assert_not_called()
                self.assertEqual(ui.session_state.messages[-1]["role"], "assistant")
                self.assertIn("couldn't retrieve", ui.session_state.messages[-1]["content"])

    def test_one_record_requests_only_one_result(self):
        collection, ui, azure = self.run_app(count=1, documents=[["Evidence"]])
        collection.query.assert_called_once_with(query_texts=["What is RAG?"], n_results=1)
        azure.return_value.chat.completions.create.assert_called_once()
        self.assertEqual(ui.session_state.messages[-1]["content"], "An answer from evidence.")

    def test_evidence_reaches_prompt_and_blank_records_are_ignored(self):
        collection, _, azure = self.run_app(count=5, documents=[["First passage", None, " ", "Second passage"]])
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 2)
        prompt = azure.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(prompt, "Context:\nFirst passage\n\nSecond passage\n\nQuestion: What is RAG?")

    def test_idle_page_does_not_construct_azure_client(self):
        collection, _, azure = self.run_app(question=None)
        collection.query.assert_not_called()
        azure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
