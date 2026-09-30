// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

fn cfg(ngram_max: usize, ngram_min: usize, max_draft: usize) -> PromptLookupConfig {
    PromptLookupConfig {
        ngram_max,
        ngram_min,
        max_draft,
        adaptive: true,
    }
}

#[test]
fn proposes_the_tokens_after_an_earlier_occurrence() {
    // "... 1 2 3 4 5 6 ... 1 2" -> propose "3 4 5".
    let context = [9, 1, 2, 3, 4, 5, 6, 8, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 3)), vec![3, 4, 5]);
}

#[test]
fn draft_is_capped_by_max_draft_and_context_end() {
    let context = [1, 2, 3, 4, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 7)), vec![3, 4, 1, 2]);
    assert_eq!(find_draft(&context, &cfg(2, 1, 1)), vec![3]);
}

#[test]
fn prefers_the_longest_matching_suffix() {
    // Suffix [5, 2]: the 2-gram matches at index 3, the 1-gram [2] matches
    // later at index 6. The longer match must win.
    let context = [0, 0, 0, 5, 2, 7, 2, 9, 5, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 1)), vec![7]);
}

#[test]
fn prefers_the_most_recent_occurrence_of_a_given_length() {
    let context = [1, 2, 10, 1, 2, 20, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 2, 1)), vec![20]);
}

#[test]
fn never_matches_the_suffix_against_itself() {
    // The only occurrence of [4] is the suffix itself.
    assert!(find_draft(&[1, 2, 3, 4], &cfg(1, 1, 3)).is_empty());
}

#[test]
fn falls_back_to_shorter_ngrams() {
    // [3, 4] never occurred before, [4] did.
    let context = [4, 7, 8, 3, 4];
    assert_eq!(find_draft(&context, &cfg(2, 1, 2)), vec![7, 8]);
    assert!(find_draft(&context, &cfg(2, 2, 2)).is_empty());
}

#[test]
fn short_and_empty_contexts_propose_nothing() {
    let c = cfg(3, 1, 7);
    assert!(find_draft(&[], &c).is_empty());
    assert!(find_draft(&[1], &c).is_empty());
    assert_eq!(find_draft(&[1, 1], &c), vec![1]);
}

#[test]
fn repeated_run_proposes_the_run() {
    // A run of one token: the latest occurrence of [7, 7] is one back, so
    // exactly one token is available after it.
    let context = [7, 7, 7, 7];
    assert_eq!(find_draft(&context, &cfg(2, 1, 5)), vec![7]);
}

#[test]
fn config_validation() {
    assert!(PromptLookupConfig::default().validate().is_ok());
    assert!(cfg(3, 0, 7).validate().is_err());
    assert!(cfg(1, 2, 7).validate().is_err());
    assert!(cfg(3, 1, 0).validate().is_err());
}

#[test]
fn tokens_per_forward_excludes_the_prefill_token() {
    let stats = PromptLookupStats {
        rounds: 4,
        drafted_rounds: 4,
        paused_rounds: 0,
        proposed_draft_tokens: 28,
        accepted_draft_tokens: 20,
    };
    // 25 generated: 1 from prefill, 24 over 4 rounds.
    assert!((stats.tokens_per_forward(25) - 6.0).abs() < 1e-9);
    assert!((stats.acceptance_rate() - 20.0 / 28.0).abs() < 1e-9);
    assert_eq!(PromptLookupStats::default().tokens_per_forward(1), 0.0);
}

#[test]
fn governor_starts_with_the_full_block() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    assert_eq!(g.budget(), 7);
}

#[test]
fn governor_keeps_the_full_block_while_blocks_land() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..10 {
        g.record(6);
    }
    assert_eq!(g.budget(), 7);
}

#[test]
fn governor_shortens_the_block_when_few_tokens_land() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..10 {
        g.record(1);
    }
    // EMA converges on 1: twice that plus two.
    assert_eq!(g.budget(), 4);
}

/// Record one miss streak long enough to start a pause.
fn miss_streak(g: &mut DraftGovernor) {
    for _ in 0..MISSES_BEFORE_COOLDOWN {
        g.record(0);
    }
}

/// Count the paused rounds, ending on the first round that proposes again.
fn pause_len(g: &mut DraftGovernor) -> usize {
    let mut paused = 0;
    while g.budget() == 0 {
        paused += 1;
    }
    paused
}

#[test]
fn governor_pauses_after_repeated_misses_and_backs_off() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 1..MISSES_BEFORE_COOLDOWN {
        g.record(0);
    }
    assert!(g.budget() > 0, "a shorter run of misses is not a streak");
    g.record(0);
    assert_eq!(pause_len(&mut g), MIN_COOLDOWN);
    // A second streak pauses twice as long.
    miss_streak(&mut g);
    assert_eq!(pause_len(&mut g), 2 * MIN_COOLDOWN);
}

#[test]
fn governor_backoff_resets_once_a_round_lands() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..3 {
        miss_streak(&mut g);
        pause_len(&mut g);
    }
    g.record(2);
    miss_streak(&mut g);
    assert_eq!(pause_len(&mut g), MIN_COOLDOWN);
}

#[test]
fn governor_backoff_is_capped() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    let mut last = 0;
    for _ in 0..10 {
        miss_streak(&mut g);
        last = pause_len(&mut g);
    }
    assert_eq!(last, MAX_COOLDOWN);
}

#[test]
fn non_adaptive_governor_always_proposes_max_draft() {
    let mut c = cfg(3, 2, 7);
    c.adaptive = false;
    let mut g = DraftGovernor::new(&c);
    for _ in 0..10 {
        g.record(0);
        assert_eq!(g.budget(), 7);
    }
}

#[test]
fn default_ngram_min_skips_single_token_matches() {
    // Only a 1-gram ([4]) recurs; the default minimum of 2 proposes nothing.
    let context = [4, 7, 8, 3, 4];
    assert!(find_draft(&context, &PromptLookupConfig::default()).is_empty());
}

/// Deterministic pseudo-random tokens from a small alphabet, so n-grams recur.
fn lcg_tokens(seed: u64, len: usize, alphabet: u64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % alphabet) as i32
        })
        .collect()
}

#[test]
fn index_matches_the_scan_at_every_length() {
    for (seed, alphabet) in [(1, 3), (2, 5), (3, 11), (4, 40)] {
        for config in [cfg(3, 1, 7), cfg(3, 2, 7), cfg(5, 2, 4), cfg(1, 1, 3)] {
            let tokens = lcg_tokens(seed, 300, alphabet);
            let mut index = NgramIndex::new(&config);
            // Grow the context one token at a time, as decoding does.
            for len in 0..=tokens.len() {
                let context = &tokens[..len];
                index.extend(context);
                assert_eq!(
                    index.find(context, &config),
                    find_draft(context, &config),
                    "seed {seed} alphabet {alphabet} {config:?} len {len}"
                );
            }
        }
    }
}

#[test]
fn index_matches_the_scan_when_extended_in_blocks() {
    // A verify round emits several tokens before the next lookup.
    let config = cfg(3, 2, 7);
    let tokens = lcg_tokens(7, 400, 6);
    let mut index = NgramIndex::new(&config);
    let mut len = 50;
    while len <= tokens.len() {
        let context = &tokens[..len];
        index.extend(context);
        assert_eq!(index.find(context, &config), find_draft(context, &config));
        len += 1 + len % 7;
    }
}

/// Target that always predicts token 3 at every position, so a reply is a run
/// of 3s: lookup proposes full blocks and the loop guard has a loop to catch.
struct ConstantModel;

impl LanguageModel for ConstantModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        let seq_len = ffi::array_shape(input_ids)[1] as usize;
        let mut logits = Vec::with_capacity(seq_len * 8);
        for _ in 0..seq_len {
            logits.extend_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }
        ffi::from_slice_f32(&logits, &[1, seq_len as i32, 8])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }
}

#[test]
fn loop_guard_stops_mid_block_where_plain_decoding_stops() {
    let sampling = SamplingConfig {
        loop_detection: crate::loop_detection::LoopDetectionConfig::new(1, 1, 8),
        ..SamplingConfig::greedy()
    };
    let prompt = [1, 2, 3, 3];
    let mut plain = crate::generate::CxxGenerator::new(1);
    let expected = plain.generate(&ConstantModel, &prompt, 64, &sampling);
    assert_eq!(
        expected,
        vec![3; 8],
        "plain decoding stops on the 8th repeat"
    );

    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let (tokens, _) = generator.generate(&ConstantModel, &prompt, 64, &sampling);
    assert_eq!(tokens, expected);
    assert!(
        generator.stats().accepted_draft_tokens > 0,
        "the run verified lookup blocks"
    );
}

#[test]
fn without_the_loop_guard_lookup_runs_to_max_tokens() {
    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let (tokens, _) =
        generator.generate(&ConstantModel, &[1, 2, 3, 3], 64, &SamplingConfig::greedy());
    assert_eq!(tokens, vec![3; 64]);
}
