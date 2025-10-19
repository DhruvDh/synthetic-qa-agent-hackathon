import unittest

from utils.harmony_prompt import extract_final


class HarmonyPromptExtractionTests(unittest.TestCase):
    def test_extract_final_without_terminal_token(self):
        text = '<|channel|>final<|message|>{"topic": "T"}'
        self.assertEqual(extract_final(text), '{"topic": "T"}')

    def test_extract_final_with_return_token(self):
        text = '<|channel|>final<|message|>{"topic": "T"}<|return|>'
        self.assertEqual(extract_final(text), '{"topic": "T"}')

    def test_extract_final_stops_before_next_channel(self):
        text = (
            '<|channel|>final<|message|>{"topic": "T"}'
            '<|channel|>analysis<|message|>extra'
        )
        self.assertEqual(extract_final(text), '{"topic": "T"}')

    def test_extract_final_allows_literal_channel_token(self):
        text = (
            '<|channel|>final<|message|>{"text": "mention <|channel|> token"}'
            '<|channel|>analysis<|message|>meta'
        )
        self.assertEqual(
            extract_final(text),
            '{"text": "mention <|channel|> token"}',
        )

    def test_extract_final_returns_none_when_missing(self):
        text = '<|channel|>analysis<|message|>{"topic": "T"}'
        self.assertIsNone(extract_final(text))


if __name__ == "__main__":
    unittest.main()
