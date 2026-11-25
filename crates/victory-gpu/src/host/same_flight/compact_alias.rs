//! Track distinct backing owners for each live title write range.

use super::super::{HostList, Step};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const MAX_FLIGHTS: usize = 256;
const MAX_RANGES: usize = 32_768;
const MAX_FLIGHT_RANGES: usize = 8_192;

#[derive(Clone, Copy)]
struct Range {
    from: usize,
    to: usize,
    shadow: usize,
}

#[derive(Clone, Copy, Default)]
struct End {
    to: usize,
    shadow: usize,
}

#[derive(Clone, Copy, Default)]
struct Ends([End; 2]);

impl Ends {
    fn include(&mut self, range: Range) {
        let end = End {
            to: range.to,
            shadow: range.shadow,
        };
        if let Some(at) = self.0.iter().position(|old| old.shadow == end.shadow) {
            self.0[at].to = self.0[at].to.max(end.to);
        } else if end.to > self.0[1].to {
            self.0[1] = end;
        }
        if self.0[1].to > self.0[0].to {
            self.0.swap(0, 1);
        }
    }

    fn other(self, shadow: usize) -> usize {
        self.0[usize::from(self.0[0].shadow == shadow)].to
    }
}

/// For every start-position prefix, keep the two greatest ends of
/// different owners. A query is one binary search even for nested ranges.
struct Scope {
    ranges: Vec<Range>,
    prefix: Vec<Ends>,
}

impl Scope {
    fn new(mut ranges: Vec<Range>) -> Self {
        ranges.sort_unstable_by_key(|r| r.from);
        let mut ends = Ends::default();
        let prefix = ranges
            .iter()
            .map(|r| {
                ends.include(*r);
                ends
            })
            .collect();
        Self { ranges, prefix }
    }

    fn from_lists(lists: &[&HostList]) -> Option<Self> {
        // Ordinary title writes do not name a Shadow. Their distinct owner
        // conservatively conflicts with every card-backed alias.
        const TITLE: usize = usize::MAX;
        let mut ranges = Vec::new();
        let mut add = |shadow, from, len| {
            if len == 0 {
                return Some(());
            }
            if ranges.len() == MAX_FLIGHT_RANGES {
                return None;
            }
            let to = usize::checked_add(from, len)?;
            ranges.push(Range { from, to, shadow });
            Some(())
        };
        for list in lists {
            let deferred = list.deferred();
            for step in &deferred.copies {
                match step {
                    Step::GpuWrite(shadow, from, len) => {
                        add(Arc::as_ptr(shadow) as usize, *from, *len)?
                    }
                    Step::Copy(from, _, len) => add(TITLE, *from, *len)?,
                    Step::Run(_) => return None,
                    _ => {}
                }
            }
            for (_, from, [row, rows, pitch]) in &deferred.backs {
                if *rows != 0 {
                    let extent = pitch.checked_mul(rows - 1)?.checked_add(*row)?;
                    add(TITLE, *from, extent)?;
                }
            }
            for &(from, _, len) in &deferred.writes {
                add(TITLE, from, len)?;
            }
        }
        Some(Self::new(ranges))
    }

    fn conflicts(&self, other: &Self) -> bool {
        other.ranges.iter().any(|range| {
            let before = self.ranges.partition_point(|old| old.from < range.to);
            before != 0 && self.prefix[before - 1].other(range.shadow) > range.from
        })
    }
}

struct Flight {
    scope: Scope,
    ambiguous: bool,
}

#[derive(Default)]
struct Tracker {
    flights: BTreeMap<u64, Flight>,
    ranges: usize,
    unknown: usize,
}

impl Tracker {
    fn record(&mut self, number: u64, scope: Option<Scope>) -> Option<u64> {
        let Some(scope) = scope.filter(|scope| {
            self.unknown == 0
                && self.flights.len() < MAX_FLIGHTS
                && self.ranges + scope.ranges.len() <= MAX_RANGES
        }) else {
            self.unknown += 1;
            for flight in self.flights.values_mut() {
                flight.ambiguous = true;
            }
            return None;
        };

        let mut ambiguous = scope.conflicts(&scope);
        for flight in self.flights.values_mut() {
            if flight.scope.conflicts(&scope) {
                flight.ambiguous = true;
                ambiguous = true;
            }
        }
        self.ranges += scope.ranges.len();
        self.flights.insert(number, Flight { scope, ambiguous });
        Some(number)
    }

    fn blocked(&self, id: Option<u64>) -> bool {
        id.and_then(|id| self.flights.get(&id))
            .is_none_or(|flight| flight.ambiguous)
    }

    fn retire(&mut self, id: Option<u64>) {
        if let Some(id) = id {
            if let Some(flight) = self.flights.remove(&id) {
                self.ranges -= flight.scope.ranges.len();
            }
        } else {
            self.unknown -= 1;
        }
    }
}

static ACTIVE: Mutex<Tracker> = Mutex::new(Tracker {
    flights: BTreeMap::new(),
    ranges: 0,
    unknown: 0,
});

pub(in crate::host) struct AliasGuard(Option<u64>);

impl AliasGuard {
    pub(in crate::host) fn register(number: u64, lists: &[&HostList]) -> Self {
        let scope = Scope::from_lists(lists);
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        Self(active.record(number, scope))
    }

    /// Keep new alias registrations out until this flight's validated
    /// snapshot is merged. Callers must not take the submission lock here.
    pub(super) fn with_all_clear<'a, T>(
        guards: impl Iterator<Item = Option<&'a AliasGuard>>,
        merge: impl FnOnce() -> T,
    ) -> Option<T> {
        let active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        guards
            .into_iter()
            .all(|g| g.is_some_and(|g| !active.blocked(g.0)))
            .then(merge)
    }

    pub(super) fn blocked(&self) -> bool {
        ACTIVE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .blocked(self.0)
    }
}

impl Drop for AliasGuard {
    fn drop(&mut self) {
        ACTIVE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retire(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(shadow: usize, from: usize, to: usize) -> Scope {
        Scope::new(vec![Range { shadow, from, to }])
    }

    #[test]
    fn rejects_only_actual_distinct_owner_overlap_or_unknown_flight() {
        let mut tracker = Tracker::default();
        let a = tracker.record(1, Some(range(1, 100, 200)));
        let touching = tracker.record(2, Some(range(2, 200, 240)));
        let same = tracker.record(3, Some(range(1, 110, 120)));
        assert!(!tracker.blocked(a));
        assert!(!tracker.blocked(touching));
        assert!(!tracker.blocked(same));
        let conflicting = tracker.record(4, Some(range(3, 119, 130)));
        assert!(tracker.blocked(a));
        assert!(tracker.blocked(same));
        assert!(tracker.blocked(conflicting));
        assert!(!tracker.blocked(touching));
        tracker.retire(conflicting);
        assert!(tracker.blocked(a));
        let unknown = tracker.record(5, None);
        assert!(unknown.is_none());
        assert!(tracker.blocked(touching));
        assert!(tracker.record(6, Some(range(2, 300, 320))).is_none());
        tracker.retire(unknown);
        tracker.retire(None);
        let clean = tracker.record(7, Some(range(4, 300, 320)));
        assert!(!tracker.blocked(clean));
    }
}
