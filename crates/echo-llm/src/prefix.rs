//! KV-cache prefix reuse planning, independent of llama.cpp.
//!
//! The persistent context keeps the tokens of the previous request in its
//! KV cache. A new prompt that starts with the same tokens (the same system
//! prompt and transcript in a follow-up chat turn, or the whole previous
//! turn plus its answer) does not need those positions evaluated again:
//! only the part after the first difference is removed from the cache and
//! decoded.
//!
//! [`prepare_cache`] holds that policy behind the [`KvCache`] seam so it is
//! unit-tested with a fake cache; `LlamaEngine` implements the seam over a
//! real `LlamaContext`.

/// The slice of KV-cache operations the prefix policy needs.
pub(crate) trait KvCache {
    /// Remove every position `>= from` of sequence 0. `false` means the
    /// cache could not do a partial removal (the caller falls back to a
    /// full clear).
    fn remove_from(&mut self, from: usize) -> bool;

    /// Drop everything.
    fn clear(&mut self);
}

/// Length of the prefix of `prompt` whose evaluation can be reused from
/// `resident`: their longest common prefix, capped so at least the last
/// prompt token is decoded again (the sampler needs its logits).
pub(crate) fn reusable_prefix<T: PartialEq>(resident: &[T], prompt: &[T]) -> usize {
    let common = resident
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    common.min(prompt.len().saturating_sub(1))
}

/// Bring `cache` and the `resident` record in line with `prompt` and
/// return how many leading prompt tokens are already evaluated; the caller
/// decodes `prompt[returned..]`.
///
/// On return `resident` equals `prompt[..returned]`. If a partial removal
/// is refused the whole cache is cleared and `0` is returned.
pub(crate) fn prepare_cache<T: PartialEq + Clone>(
    cache: &mut impl KvCache,
    resident: &mut Vec<T>,
    prompt: &[T],
) -> usize {
    let keep = reusable_prefix(resident, prompt);
    if keep < resident.len() && !cache.remove_from(keep) {
        // The cache cannot drop a tail on its own: start over rather than
        // trust positions whose state is unknown.
        cache.clear();
        resident.clear();
        return 0;
    }
    resident.truncate(keep);
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake cache recording what the policy asked for.
    #[derive(Default)]
    struct FakeCache {
        removed_from: Vec<usize>,
        cleared: usize,
        refuse_partial: bool,
    }

    impl KvCache for FakeCache {
        fn remove_from(&mut self, from: usize) -> bool {
            self.removed_from.push(from);
            !self.refuse_partial
        }
        fn clear(&mut self) {
            self.cleared += 1;
        }
    }

    #[test]
    fn an_identical_prompt_redecodes_only_the_last_token() {
        assert_eq!(reusable_prefix(&[1, 2, 3, 4], &[1, 2, 3, 4]), 3);
    }

    #[test]
    fn an_empty_resident_record_reuses_nothing() {
        assert_eq!(reusable_prefix::<u32>(&[], &[1, 2, 3]), 0);
    }

    #[test]
    fn an_empty_prompt_reuses_nothing() {
        assert_eq!(reusable_prefix(&[1, 2], &[] as &[u32]), 0);
    }

    #[test]
    fn a_single_token_prompt_is_always_redecoded() {
        assert_eq!(reusable_prefix(&[1], &[1]), 0);
    }

    #[test]
    fn a_prompt_extending_the_resident_tokens_reuses_all_of_them() {
        // chat turn 2: previous prompt + answer, then the new user turn
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
    }

    #[test]
    fn a_diverging_prompt_reuses_up_to_the_divergence() {
        assert_eq!(reusable_prefix(&[1, 2, 3, 4], &[1, 2, 9, 4, 5]), 2);
    }

    #[test]
    fn a_prompt_shorter_than_the_resident_tokens_keeps_one_to_decode() {
        assert_eq!(reusable_prefix(&[1, 2, 3, 4, 5], &[1, 2, 3]), 2);
    }

    #[test]
    fn prepare_removes_the_stale_tail_and_records_the_kept_prefix() {
        let mut cache = FakeCache::default();
        let mut resident = vec![1, 2, 3, 4];
        let kept = prepare_cache(&mut cache, &mut resident, &[1, 2, 9, 5]);
        assert_eq!(kept, 2);
        assert_eq!(cache.removed_from, [2]);
        assert_eq!(cache.cleared, 0);
        assert_eq!(resident, [1, 2]);
    }

    #[test]
    fn prepare_skips_the_removal_when_nothing_is_stale() {
        let mut cache = FakeCache::default();
        let mut resident = vec![1, 2, 3];
        let kept = prepare_cache(&mut cache, &mut resident, &[1, 2, 3, 4]);
        assert_eq!(kept, 3);
        assert!(cache.removed_from.is_empty());
        assert_eq!(resident, [1, 2, 3]);
    }

    #[test]
    fn prepare_with_an_identical_prompt_drops_only_the_last_position() {
        let mut cache = FakeCache::default();
        let mut resident = vec![1, 2, 3];
        assert_eq!(prepare_cache(&mut cache, &mut resident, &[1, 2, 3]), 2);
        assert_eq!(cache.removed_from, [2]);
        assert_eq!(resident, [1, 2]);
    }

    #[test]
    fn prepare_with_nothing_resident_decodes_the_whole_prompt() {
        let mut cache = FakeCache::default();
        let mut resident: Vec<u32> = Vec::new();
        assert_eq!(prepare_cache(&mut cache, &mut resident, &[1, 2, 3]), 0);
        assert!(cache.removed_from.is_empty());
        assert_eq!(cache.cleared, 0);
        assert!(resident.is_empty());
    }

    #[test]
    fn prepare_falls_back_to_a_full_clear_when_partial_removal_is_refused() {
        let mut cache = FakeCache {
            refuse_partial: true,
            ..FakeCache::default()
        };
        let mut resident = vec![1, 2, 3, 4];
        let kept = prepare_cache(&mut cache, &mut resident, &[1, 2, 3, 9]);
        assert_eq!(kept, 0);
        assert_eq!(cache.cleared, 1);
        assert!(resident.is_empty());
    }
}
