// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Scan metrics and I/O counting for Parquet data file reads.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use crate::error::Result;
use crate::io::FileRead;
use crate::scan::ArrowRecordBatchStream;

/// Wraps a [`FileRead`] to count bytes read via a shared atomic counter.
pub(crate) struct CountingFileRead<F: FileRead> {
    inner: F,
    bytes_read: Arc<AtomicU64>,
}

impl<F: FileRead> CountingFileRead<F> {
    pub(crate) fn new(inner: F, bytes_read: Arc<AtomicU64>) -> Self {
        Self { inner, bytes_read }
    }
}

#[async_trait::async_trait]
impl<F: FileRead> FileRead for CountingFileRead<F> {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        debug_assert!(range.end >= range.start);
        self.bytes_read
            .fetch_add(range.end - range.start, Ordering::Relaxed);
        self.inner.read(range).await
    }
}

/// Metrics collected during an Iceberg scan.
///
/// Counters are shared by every clone and are updated while the scan's
/// record batch stream is polled, so read them after the stream is drained.
///
/// Row-group pruning counters are attributable per mechanism: a row group
/// is counted under exactly one of [`Self::row_groups_pruned_by_statistics`]
/// or [`Self::row_groups_pruned_by_bloom_filter`], because Bloom filters are
/// only probed for row groups that survived statistics pruning. The row
/// groups that were read are `considered - pruned_by_statistics -
/// pruned_by_bloom_filter`.
#[derive(Clone, Debug)]
pub struct ScanMetrics {
    bytes_read: Arc<AtomicU64>,
    pruning: Arc<PruningCounters>,
}

/// Per-mechanism predicate pruning counters shared by clones of
/// [`ScanMetrics`].
#[derive(Debug, Default)]
struct PruningCounters {
    row_groups_considered: AtomicU64,
    row_groups_pruned_by_statistics: AtomicU64,
    row_groups_pruned_by_bloom_filter: AtomicU64,
    rows_pruned_by_page_index: AtomicU64,
}

impl ScanMetrics {
    pub(crate) fn new() -> Self {
        Self {
            bytes_read: Arc::new(AtomicU64::new(0)),
            pruning: Arc::default(),
        }
    }

    pub(crate) fn bytes_read_counter(&self) -> &Arc<AtomicU64> {
        &self.bytes_read
    }

    /// Total bytes read from storage during this scan, including data files and delete files.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Row groups that predicate pruning evaluated: for every data file task
    /// that carries a predicate (scan filter and/or equality deletes), the
    /// row groups assigned to that task's byte range.
    pub fn row_groups_considered(&self) -> u64 {
        self.pruning.row_groups_considered.load(Ordering::Relaxed)
    }

    /// Considered row groups excluded by row-group min/max and null-count
    /// statistics.
    pub fn row_groups_pruned_by_statistics(&self) -> u64 {
        self.pruning
            .row_groups_pruned_by_statistics
            .load(Ordering::Relaxed)
    }

    /// Row groups that survived statistics pruning and were then excluded
    /// because a Parquet split-block Bloom filter proved every equality or
    /// `IN` probe value absent.
    pub fn row_groups_pruned_by_bloom_filter(&self) -> u64 {
        self.pruning
            .row_groups_pruned_by_bloom_filter
            .load(Ordering::Relaxed)
    }

    /// Rows in surviving row groups that the page index (column and offset
    /// index) row selection skipped. Rows skipped only by positional deletes
    /// are not counted.
    pub fn rows_pruned_by_page_index(&self) -> u64 {
        self.pruning
            .rows_pruned_by_page_index
            .load(Ordering::Relaxed)
    }

    /// Adds `count` row groups to [`Self::row_groups_considered`].
    pub(crate) fn record_row_groups_considered(&self, count: usize) {
        Self::add(&self.pruning.row_groups_considered, count);
    }

    /// Adds `count` row groups to [`Self::row_groups_pruned_by_statistics`].
    pub(crate) fn record_row_groups_pruned_by_statistics(&self, count: usize) {
        Self::add(&self.pruning.row_groups_pruned_by_statistics, count);
    }

    /// Adds `count` row groups to [`Self::row_groups_pruned_by_bloom_filter`].
    pub(crate) fn record_row_groups_pruned_by_bloom_filter(&self, count: usize) {
        Self::add(&self.pruning.row_groups_pruned_by_bloom_filter, count);
    }

    /// Adds `count` rows to [`Self::rows_pruned_by_page_index`].
    pub(crate) fn record_rows_pruned_by_page_index(&self, count: usize) {
        Self::add(&self.pruning.rows_pruned_by_page_index, count);
    }

    /// Adds a `usize` count to a relaxed atomic counter.
    fn add(counter: &AtomicU64, count: usize) {
        counter.fetch_add(count as u64, Ordering::Relaxed);
    }
}

/// Result of [`ArrowReader::read`](super::ArrowReader::read), containing the
/// record batch stream and metrics collected during the scan.
pub struct ScanResult {
    stream: ArrowRecordBatchStream,
    metrics: ScanMetrics,
}

impl ScanResult {
    pub(crate) fn new(stream: ArrowRecordBatchStream, metrics: ScanMetrics) -> Self {
        Self { stream, metrics }
    }

    /// Consumes the result, returning only the record batch stream.
    pub fn stream(self) -> ArrowRecordBatchStream {
        self.stream
    }

    /// Returns a reference to the scan metrics.
    pub fn metrics(&self) -> &ScanMetrics {
        &self.metrics
    }
}
