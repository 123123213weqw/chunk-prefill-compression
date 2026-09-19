import copy
import itertools
import unittest
from methods import strict_score, random_tokens, random_spans, oracle_positions, semantic_positions, paired_summary, seed_for, compare_seed_family
from data import reparse, compute


class MethodsTest(unittest.TestCase):
    def test_json_order_and_fence(self):
        self.assertTrue(strict_score('```json\n{"b":2,"a":1}\n```', {'a': 1, 'b': 2})['correct'])

    def test_rejects_wrong_types_and_extra_text(self):
        for text in ('{"a":true}', '{"a":1.0}', '{"a":"1"}', '{"a":1,"b":2}',
                     '{"a":1,"a":2}', 'answer: {"a":1}', '{}', 'null', '[1]'):
            self.assertFalse(strict_score(text, {'a': 1})['correct'], text)

    def test_separates_format_and_content(self):
        s = strict_score('{"a":2}', {'a': 1})
        self.assertTrue(s['format_ok'])
        self.assertFalse(s['correct'])

    def test_random_is_paired(self):
        self.assertEqual(random_tokens(100, 25, seed_for('pair', 2)), random_tokens(100, 25, seed_for('pair', 2)))
        self.assertEqual(len(random_tokens(100, 25, 3)), 25)
        self.assertNotEqual(random_tokens(100, 25, 3), random_tokens(100, 25, 4))

    def test_span_subset_sum_optimal_budget_and_atomicity(self):
        spans = [{'a': 0, 'b': 3}, {'a': 3, 'b': 8}, {'a': 8, 'b': 15}, {'a': 15, 'b': 26}]
        for cap in range(27):
            for seed in range(5):
                selected = set(random_spans(spans, cap, seed))
                possible = [sum(s['b']-s['a'] for s, used in zip(spans, flags) if used)
                            for flags in itertools.product((0, 1), repeat=len(spans))]
                self.assertEqual(len(selected), max(x for x in possible if x <= cap))
                for s in spans:
                    part = selected & set(range(s['a'], s['b']))
                    self.assertTrue(not part or len(part) == s['b']-s['a'])

    def test_oracle_never_silently_truncates(self):
        self.assertTrue({2, 7} <= set(oracle_positions(20, [2, 7], 5, 42)))
        with self.assertRaises(ValueError):
            oracle_positions(20, [2, 7], 1, 42)

    def test_semantic_no_mutation_or_gold(self):
        spans = [{'a': 0, 'b': 4, 'text': 'x'}, {'a': 4, 'b': 6, 'text': 'y'}]
        original = copy.deepcopy(spans)
        selected = semantic_positions(spans, 4, lambda text, seen: (1, text))
        self.assertEqual(selected, list(range(4)))
        self.assertEqual(spans, original)

    def test_zero_full_denominator_is_na(self):
        rows = [{'pair_id': 'p', 'member': m, 'conditions': {
            'full': {'probes': [{'probe_id': 'p', 'correct': False}]},
            'compressed': {'probes': [{'probe_id': 'p', 'correct': True}]}}} for m in (0, 1)]
        result = paired_summary(rows, 'compressed', 'p', bootstrap=20)
        self.assertIsNone(result['retention'])
        self.assertEqual(result['pairs'], 1)
        self.assertEqual(result['paired_delta'], 1)
        self.assertEqual(paired_summary(rows[:1], 'compressed', 'p')['n'], 0)

    def test_reparse_not_state(self):
        row = {'target_job': 'ABC', 'segments': {'T': {'text': 'job=ABC field=shard_count value=11\njob=DEF field=shard_count value=99\n'}}}
        self.assertEqual(reparse(row), {'shard_count': 11})
        row['segments']['T']['text'] += 'job=ABC field=shard_count value=12\n'
        with self.assertRaises(ValueError):
            reparse(row)

    def test_math(self):
        self.assertEqual(compute('inventory', {'shard_count': 7, 'items_per_shard': 12, 'rejected_items': 5}), 79)
        self.assertEqual(compute('latency', {'base_latency_ms': 30, 'retry_count': 3, 'retry_penalty_ms': 7, 'parallel_discount_ms': 4}), 47)

    def test_seed_replications_do_not_inflate_sample_count(self):
        def cond(correct):
            return {'probes': [{'probe_id': 'readout', 'correct': correct}], 'metrics': {'T_kept': 10}}
        rows = [{'pair_id': 'p', 'member': m, 'conditions': {'auto': cond(True), **{f'r{s}': cond(False) for s in range(5)}}} for m in (0, 1)]
        summary = compare_seed_family(rows, 'auto', [f'r{s}' for s in range(5)], bootstrap=20)
        self.assertEqual((summary['pairs'], summary['members'], summary['random_seeds']), (1, 2, 5))
        self.assertEqual(summary['mean_delta'], 1)
        rows[0]['conditions']['r0']['metrics']['T_kept'] = 9
        with self.assertRaises(ValueError):
            compare_seed_family(rows, 'auto', [f'r{s}' for s in range(5)])


if __name__ == '__main__':
    unittest.main()
