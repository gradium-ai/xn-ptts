//! Bounded storage for immutable voice prefixes. Request KV buffers are owned separately.

use std::collections::VecDeque;

/// Limits on retained voice prefixes. Either limit set to zero disables retention.
///
/// The byte limit counts flow-LM K/V tensor payloads, including the guidance null branch.
/// It excludes allocator overhead, shared weights/conditions, and active request buffers;
/// it is not a bound on process RSS or device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct VoiceCacheConfig {
    pub max_entries: usize,
    pub max_kv_bytes: usize,
}

impl Default for VoiceCacheConfig {
    fn default() -> Self {
        Self { max_entries: 16, max_kv_bytes: 64 * 1024 * 1024 }
    }
}

/// A coherent snapshot of one Synth's prefix cache. Counters are cumulative since load.
///
/// Hits/misses count lookups, including simultaneous misses for one voice. Priming runs
/// outside the lock; duplicate completed primes occupy only one entry. Evictions count
/// entries removed for capacity; invalidations count entries removed after a voice or
/// conditioner changes. Bypasses count completed primes too large to retain or disabled
/// retention. Existing Sessions can keep copies alive after a prefix leaves this cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct VoiceCacheStats {
    pub entries: usize,
    pub kv_bytes: usize,
    pub max_entries: usize,
    pub max_kv_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub bypasses: u64,
}

type Key = (String, bool);

struct Entry<T> {
    key: Key,
    value: T,
    kv_bytes: usize,
}

/// The small bounded queue is ordered least to most recently used. No clock or timestamp
/// counter is needed; touching an entry cannot overflow or leave duplicate ordering records.
pub(crate) struct VoiceCache<T> {
    entries: VecDeque<Entry<T>>,
    stats: VoiceCacheStats,
}

impl<T> VoiceCache<T> {
    pub(crate) fn new(config: VoiceCacheConfig) -> Self {
        Self {
            entries: VecDeque::new(),
            stats: VoiceCacheStats {
                max_entries: config.max_entries,
                max_kv_bytes: config.max_kv_bytes,
                ..Default::default()
            },
        }
    }

    pub(crate) fn get(&mut self, key: &Key) -> Option<&T> {
        match self.entries.iter().position(|entry| &entry.key == key) {
            None => {
                self.stats.misses += 1;
                None
            }
            Some(index) => {
                self.stats.hits += 1;
                let entry = self.entries.remove(index).expect("position exists");
                self.entries.push_back(entry);
                self.entries.back().map(|entry| &entry.value)
            }
        }
    }

    pub(crate) fn insert(&mut self, key: Key, value: T, kv_bytes: usize) {
        // Another request may have completed this prime while our caller was outside the
        // lock. Retain the existing immutable prefix without evicting unrelated voices.
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.remove(index).expect("position exists");
            self.entries.push_back(entry);
            return;
        }
        if self.stats.max_entries == 0
            || self.stats.max_kv_bytes == 0
            || kv_bytes > self.stats.max_kv_bytes
        {
            self.stats.bypasses += 1;
            return;
        }
        while self.entries.len() >= self.stats.max_entries
            || self.stats.kv_bytes > self.stats.max_kv_bytes - kv_bytes
        {
            let entry = self.entries.pop_front().expect("capacity requires an eviction");
            self.stats.kv_bytes -= entry.kv_bytes;
            self.stats.evictions += 1;
        }
        self.entries.push_back(Entry { key, value, kv_bytes });
        self.stats.kv_bytes += kv_bytes;
        self.stats.insertions += 1;
    }

    pub(crate) fn invalidate_voice(&mut self, voice: &str) {
        self.entries.retain(|entry| {
            if entry.key.0 != voice {
                return true;
            }
            self.stats.kv_bytes -= entry.kv_bytes;
            self.stats.invalidations += 1;
            false
        });
    }

    pub(crate) fn clear(&mut self) {
        self.stats.invalidations += self.entries.len() as u64;
        self.entries.clear();
        self.stats.kv_bytes = 0;
    }

    pub(crate) fn stats(&self) -> VoiceCacheStats {
        VoiceCacheStats { entries: self.entries.len(), ..self.stats }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> Key {
        (name.into(), false)
    }

    fn cache(entries: usize, bytes: usize) -> VoiceCache<u8> {
        VoiceCache::new(VoiceCacheConfig { max_entries: entries, max_kv_bytes: bytes })
    }

    #[test]
    fn seventeenth_voice_keeps_recent_prefixes_instead_of_clearing_everything() {
        let mut c = cache(16, 1024);
        for i in 0..16 {
            c.insert(key(&i.to_string()), i, 8);
        }
        assert_eq!(c.get(&key("0")), Some(&0)); // protect the oldest inserted voice
        c.insert(key("16"), 16, 8);
        assert_eq!(c.stats().entries, 16);
        assert_eq!(c.stats().evictions, 1);
        assert_eq!(c.get(&key("1")), None);
        for i in [0, 2, 3, 15, 16] {
            assert_eq!(c.get(&key(&i.to_string())), Some(&i));
        }
        assert_eq!(c.stats().kv_bytes, 128);
    }

    #[test]
    fn byte_budget_can_evict_multiple_entries_and_exact_fit_is_retained() {
        let mut c = cache(10, 10);
        for name in ["a", "b", "c"] {
            c.insert(key(name), 1, 3);
        }
        c.insert(key("d"), 2, 7);
        assert_eq!(c.stats().entries, 2);
        assert_eq!(c.stats().kv_bytes, 10);
        assert_eq!(c.stats().evictions, 2);
        assert_eq!(c.get(&key("c")), Some(&1));
        assert_eq!(c.get(&key("d")), Some(&2));
    }

    #[test]
    fn oversized_prefix_does_not_evict_useful_entries() {
        let mut c = cache(2, 8);
        c.insert(key("hot"), 1, 8);
        c.insert(key("oversize"), 2, 9);
        assert_eq!(c.get(&key("hot")), Some(&1));
        assert_eq!(c.stats().bypasses, 1);
        assert_eq!(c.stats().evictions, 0);
        assert_eq!(c.stats().kv_bytes, 8);
    }

    #[test]
    fn either_zero_limit_disables_retention() {
        for (entries, bytes) in [(0, 10), (10, 0)] {
            let mut c = cache(entries, bytes);
            c.insert(key("a"), 1, 0);
            assert_eq!(c.get(&key("a")), None);
            assert_eq!(c.stats().entries, 0);
            assert_eq!(c.stats().bypasses, 1);
        }
    }

    #[test]
    fn racing_insertion_does_not_duplicate_or_evict() {
        let mut c = cache(2, 10);
        assert_eq!(c.get(&key("a")), None);
        assert_eq!(c.get(&key("a")), None);
        c.insert(key("a"), 1, 5);
        c.insert(key("b"), 2, 5);
        c.insert(key("a"), 3, 5);
        assert_eq!(c.get(&key("a")), Some(&1));
        assert_eq!(c.stats().insertions, 2);
        assert_eq!(c.stats().misses, 2);
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().evictions, 0);
        assert_eq!(c.stats().kv_bytes, 10);
    }

    #[test]
    fn voice_replacement_invalidates_both_guidance_keys_only() {
        let mut c = cache(3, 30);
        c.insert(key("a"), 1, 5);
        c.insert(("a".into(), true), 2, 10);
        c.insert(key("b"), 3, 5);
        c.invalidate_voice("a");
        assert_eq!(c.get(&key("a")), None);
        assert_eq!(c.get(&("a".into(), true)), None);
        assert_eq!(c.get(&key("b")), Some(&3));
        assert_eq!(c.stats().kv_bytes, 5);
        assert_eq!(c.stats().invalidations, 2);
        c.clear();
        assert_eq!(c.stats().entries, 0);
        assert_eq!(c.stats().kv_bytes, 0);
        assert_eq!(c.stats().invalidations, 3);
        assert_eq!(c.stats().hits, 1); // invalidation does not reset lifetime counters
    }
}
