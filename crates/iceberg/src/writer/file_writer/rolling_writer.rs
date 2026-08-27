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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::future::{self, BoxFuture};

use crate::io::{FileIO, OutputFile};
use crate::runtime::{JoinHandle, Runtime};
use crate::spec::{DataFileBuilder, PartitionKey, TableProperties};
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};
use crate::writer::{CurrentFileStatus, PositionDeleteInput};
use crate::{Error, ErrorKind, Result};

type CloseFuture = BoxFuture<
    'static,
    (
        u64,
        String,
        RollingCloseReason,
        Result<Vec<DataFileBuilder>>,
    ),
>;

/// Why a [`RollingFileWriter`] stopped writing to one output object.
///
/// This is observation only. The variant records the decision the writer
/// already made; it never influences one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingCloseReason {
    /// The observed `current_written_size()` estimate exceeded the configured
    /// target, so the writer closed the output before accepting the next write.
    Threshold,
    /// The writer reached end of stream and closed the final, possibly small,
    /// residue for that partition. A residue closes regardless of the estimate.
    Final,
    /// The caller cancelled the writer through
    /// [`RollingFileWriter::cancel`], so this output was closed as cancellation
    /// evidence rather than as a completed residue.
    Cancel,
    /// The close itself failed. The output may or may not exist in storage.
    ///
    /// This is a settlement reason only: it never appears on
    /// [`RollingWriterEvent::CloseDecided`], because the writer must decide to
    /// close before it can discover that closing failed.
    Error,
}

/// A closed, non-semantic observation emitted by a [`RollingFileWriter`].
///
/// Every event is keyed by `logical_ordinal`: the order in which the writer
/// *opened* its outputs. That ordinal is assigned before any close can be
/// spawned, so it stays stable even when closes complete out of order under
/// `max_concurrent_closes`. `completion_ordinal` separately records the order in
/// which closes actually finished, so a caller can distinguish "the third file
/// this writer opened" from "the third close that landed" without inferring
/// either from file size or from the returned vector's order.
///
/// Emission is synchronous and infallible by construction: an observer cannot
/// fail, block, backpressure, reorder outputs, or change a roll decision.
#[derive(Debug, Clone)]
pub enum RollingWriterEvent {
    /// A new output object was opened.
    OutputOpened {
        /// Order in which this writer opened the output.
        logical_ordinal: u64,
        /// Path of the opened output object.
        path: String,
    },
    /// The writer decided to stop writing to an output, before the close runs.
    ///
    /// `written_size_estimate` is the writer's *anticipated encoded size* at the
    /// moment of the decision, not the final object length. For
    /// [`RollingCloseReason::Threshold`] it is strictly greater than
    /// `target_file_size`, and the decision is taken before the next write is
    /// accepted.
    CloseDecided {
        /// Order in which this writer opened the output.
        logical_ordinal: u64,
        /// Path of the output being closed.
        path: String,
        /// Why the output is being closed.
        reason: RollingCloseReason,
        /// Configured rolling target in bytes.
        target_file_size: usize,
        /// `current_written_size()` observed at the decision.
        written_size_estimate: usize,
    },
    /// A close finished, successfully or not. Exactly one per decided close.
    CloseSettled {
        /// Order in which this writer opened the output.
        logical_ordinal: u64,
        /// Order in which this close settled among this writer's closes.
        completion_ordinal: u64,
        /// Path of the output object.
        path: String,
        /// The decided reason, or [`RollingCloseReason::Error`] when the close
        /// failed.
        reason: RollingCloseReason,
        /// Number of data-file builders the close produced, or `None` when the
        /// close failed.
        output_files: Option<usize>,
    },
}

/// Receives [`RollingWriterEvent`]s from a [`RollingFileWriter`].
///
/// Implementations must be cheap, non-blocking, and infallible. The writer
/// calls this on its own execution path, so an implementation that blocks or
/// allocates unboundedly will slow physical writing. Update atomics or a
/// bounded map keyed by `logical_ordinal`; do not push into an unbounded
/// channel.
pub trait RollingWriterObserver: Debug + Send + Sync {
    /// Records one observation. Must not panic.
    fn on_event(&self, event: RollingWriterEvent);
}

/// Builder for [`RollingFileWriter`].
#[derive(Clone, Debug)]
pub struct RollingFileWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner_builder: B,
    target_file_size: usize,
    file_io: FileIO,
    location_generator: L,
    file_name_generator: F,
    max_concurrent_closes: usize,
    observer: Option<Arc<dyn RollingWriterObserver>>,
}

impl<B, L, F> RollingFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Creates a new `RollingFileWriterBuilder` with the specified target file size.
    ///
    /// # Parameters
    ///
    /// * `inner_builder` - The builder for the underlying file writer
    /// * `target_file_size` - The target file size in bytes that triggers rollover
    /// * `file_io` - The file IO interface for creating output files
    /// * `location_generator` - Generator for file locations
    /// * `file_name_generator` - Generator for file names
    ///
    /// # Returns
    ///
    /// A new `RollingFileWriterBuilder` instance
    pub fn new(
        inner_builder: B,
        target_file_size: usize,
        file_io: FileIO,
        location_generator: L,
        file_name_generator: F,
    ) -> Self {
        Self {
            inner_builder,
            target_file_size,
            file_io,
            location_generator,
            file_name_generator,
            max_concurrent_closes: 0,
            observer: None,
        }
    }

    /// Creates a new `RollingFileWriterBuilder` with the default target file size.
    ///
    /// # Parameters
    ///
    /// * `inner_builder` - The builder for the underlying file writer
    /// * `file_io` - The file IO interface for creating output files
    /// * `location_generator` - Generator for file locations
    /// * `file_name_generator` - Generator for file names
    ///
    /// # Returns
    ///
    /// A new `RollingFileWriterBuilder` instance with default target file size
    pub fn new_with_default_file_size(
        inner_builder: B,
        file_io: FileIO,
        location_generator: L,
        file_name_generator: F,
    ) -> Self {
        Self {
            inner_builder,
            target_file_size: TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT,
            file_io,
            location_generator,
            file_name_generator,
            max_concurrent_closes: 0,
            observer: None,
        }
    }

    /// Allow rolled files to be closed in the background.
    ///
    /// A value of `0` disables background close and preserves the historical
    /// synchronous behavior. Values greater than `0` allow up to that many
    /// outstanding close tasks before writes wait for one to finish.
    pub fn with_max_concurrent_closes(mut self, max_concurrent_closes: usize) -> Self {
        self.max_concurrent_closes = max_concurrent_closes;
        self
    }

    /// Observe this writer's open, roll-decision, and close events.
    ///
    /// The observer is strictly passive: it cannot change which files are
    /// written, when the writer rolls, what the close returns, or the order of
    /// the returned builders. It exists because the roll estimate and the reason
    /// an output closed cannot be recovered after the fact from the object's
    /// final size — a target-triggered close and a final residue are
    /// indistinguishable by bytes alone.
    pub fn with_observer(mut self, observer: Arc<dyn RollingWriterObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Build a new [`RollingFileWriter`].
    pub fn build(&self) -> RollingFileWriter<B, L, F> {
        RollingFileWriter {
            inner: None,
            inner_builder: self.inner_builder.clone(),
            target_file_size: self.target_file_size,
            data_file_builders: vec![],
            file_io: self.file_io.clone(),
            location_generator: self.location_generator.clone(),
            file_name_generator: self.file_name_generator.clone(),
            close_futures: vec![],
            max_concurrent_closes: self.max_concurrent_closes,
            observer: self.observer.clone(),
            current_output: None,
            next_logical_ordinal: 0,
            next_completion_ordinal: 0,
        }
    }
}

/// A writer that automatically rolls over to a new file when the data size
/// exceeds a target threshold.
///
/// This writer wraps another file writer that tracks the amount of data written.
/// When the data size exceeds the target size, it closes the current file and
/// starts writing to a new one.
pub struct RollingFileWriter<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> {
    inner: Option<B::R>,
    inner_builder: B,
    target_file_size: usize,
    data_file_builders: Vec<DataFileBuilder>,
    file_io: FileIO,
    location_generator: L,
    file_name_generator: F,
    close_futures: Vec<CloseFuture>,
    max_concurrent_closes: usize,
    observer: Option<Arc<dyn RollingWriterObserver>>,
    /// `(logical_ordinal, path)` of the output `inner` is currently writing.
    ///
    /// Captured when the output is opened so that a close spawned into the
    /// background can still be attributed to the output it belongs to.
    current_output: Option<(u64, String)>,
    /// Next `logical_ordinal` to assign when an output is opened.
    next_logical_ordinal: u64,
    /// Next `completion_ordinal` to assign when a close finishes.
    next_completion_ordinal: u64,
}

impl<B, L, F> Debug for RollingFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RollingFileWriter")
            .field("target_file_size", &self.target_file_size)
            .field("file_io", &self.file_io)
            .finish()
    }
}

impl<B, L, F> RollingFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Determines if the writer should roll over to a new file.
    ///
    /// # Returns
    ///
    /// `true` if a new file should be started, `false` otherwise
    fn should_roll(&self) -> bool {
        self.current_written_size() > self.target_file_size
    }

    fn new_output_file(&self, partition_key: &Option<PartitionKey>) -> Result<OutputFile> {
        self.file_io
            .new_output(self.location_generator.generate_location(
                partition_key.as_ref(),
                &self.file_name_generator.generate_file_name(),
            ))
    }

    /// Delivers one event to the observer, if any is installed.
    ///
    /// Kept in one place so that every emission site is trivially auditable as
    /// side-effect free with respect to the writer's own state.
    fn observe(&self, event: RollingWriterEvent) {
        if let Some(observer) = self.observer.as_ref() {
            observer.on_event(event);
        }
    }

    /// Opens a new output object and records its stable logical identity.
    ///
    /// The logical ordinal is assigned here — before any close for this output
    /// can be spawned — so background closes remain attributable in open order.
    ///
    /// # Errors
    ///
    /// Returns an error when the output location cannot be created or the inner
    /// writer cannot be built.
    async fn open_output(&mut self, partition_key: &Option<PartitionKey>) -> Result<()> {
        let inner = self
            .inner_builder
            .build(self.new_output_file(partition_key)?)
            .await?;
        let logical_ordinal = self.next_logical_ordinal;
        self.next_logical_ordinal += 1;
        let path = inner.current_file_path();
        self.current_output = Some((logical_ordinal, path.clone()));
        self.inner = Some(inner);
        self.observe(RollingWriterEvent::OutputOpened {
            logical_ordinal,
            path,
        });
        Ok(())
    }

    /// Records the decision to close the currently open output.
    ///
    /// Returns the `(logical_ordinal, path)` identity of that output so the
    /// caller can attribute the close completion, or `None` when no output is
    /// open. `written_size_estimate` must be sampled by the caller while the
    /// inner writer is still installed, so the event reports the value the roll
    /// decision was actually made on rather than the post-detach zero.
    fn decide_close(
        &mut self,
        reason: RollingCloseReason,
        written_size_estimate: usize,
    ) -> Option<(u64, String)> {
        let (logical_ordinal, path) = self.current_output.take()?;
        self.observe(RollingWriterEvent::CloseDecided {
            logical_ordinal,
            path: path.clone(),
            reason,
            target_file_size: self.target_file_size,
            written_size_estimate,
        });
        Some((logical_ordinal, path))
    }

    fn spawn_close(&mut self, inner: B::R, output: (u64, String), reason: RollingCloseReason) {
        let handle: JoinHandle<Result<Vec<DataFileBuilder>>> = Runtime::current()
            .io()
            .spawn(async move { inner.close().await });
        let (logical_ordinal, path) = output;
        self.close_futures.push(Box::pin(async move {
            let result = match handle.await {
                Ok(result) => result,
                Err(err) => Err(err),
            };
            (logical_ordinal, path, reason, result)
        }));
    }

    /// Records one finished close and folds its builders into the output.
    ///
    /// The completion ordinal is assigned here, so it reflects the real
    /// completion order rather than the open order.
    fn record_close(
        &mut self,
        logical_ordinal: u64,
        path: String,
        reason: RollingCloseReason,
        result: Result<Vec<DataFileBuilder>>,
    ) -> Result<()> {
        let completion_ordinal = self.next_completion_ordinal;
        self.next_completion_ordinal += 1;
        match result {
            Ok(files) => {
                self.observe(RollingWriterEvent::CloseSettled {
                    logical_ordinal,
                    completion_ordinal,
                    path,
                    reason,
                    output_files: Some(files.len()),
                });
                self.data_file_builders.extend(files);
                Ok(())
            }
            Err(err) => {
                self.observe(RollingWriterEvent::CloseSettled {
                    logical_ordinal,
                    completion_ordinal,
                    path,
                    reason: RollingCloseReason::Error,
                    output_files: None,
                });
                Err(err)
            }
        }
    }

    async fn wait_for_one_close(&mut self) -> Result<()> {
        if self.close_futures.is_empty() {
            return Ok(());
        }

        let ((logical_ordinal, path, reason, result), _index, remaining) =
            future::select_all(std::mem::take(&mut self.close_futures)).await;
        self.close_futures = remaining;

        self.record_close(logical_ordinal, path, reason, result)
    }

    async fn ensure_partition_writer(
        &mut self,
        partition_key: &Option<PartitionKey>,
    ) -> Result<&mut B::R> {
        if self.inner.is_none() {
            self.open_output(partition_key).await?;
        }

        if self.should_roll() {
            // Sampled while the inner writer is still installed; detaching it
            // below would make the estimate read as zero.
            let written_size_estimate = self.current_written_size();
            if self.max_concurrent_closes > 0
                && self.close_futures.len() >= self.max_concurrent_closes
            {
                self.wait_for_one_close().await?;
            }

            if let Some(inner) = self.inner.take() {
                let output =
                    self.decide_close(RollingCloseReason::Threshold, written_size_estimate);
                if self.max_concurrent_closes == 0 {
                    let result = inner.close().await;
                    match output {
                        Some((logical_ordinal, path)) => self.record_close(
                            logical_ordinal,
                            path,
                            RollingCloseReason::Threshold,
                            result,
                        )?,
                        None => self.data_file_builders.extend(result?),
                    }
                } else if let Some(output) = output {
                    self.spawn_close(inner, output, RollingCloseReason::Threshold);
                } else {
                    self.data_file_builders.extend(inner.close().await?);
                }

                // start a new writer
                self.open_output(partition_key).await?;
            }
        }

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Writer is not initialized!"))?;
        Ok(writer)
    }

    /// Writes a record batch to the current file, rolling over to a new file if necessary.
    ///
    /// # Parameters
    ///
    /// * `partition_key` - Optional partition key for the data
    /// * `input` - The record batch to write
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is not initialized or if writing fails
    pub async fn write(
        &mut self,
        partition_key: &Option<PartitionKey>,
        input: &RecordBatch,
    ) -> Result<()> {
        let writer = self.ensure_partition_writer(partition_key).await?;
        writer.write(input).await
    }

    /// Writes a record batch and returns the `(file_path, pos)` position for each record.
    ///
    /// # Parameters
    ///
    /// * `partition_key` - Optional partition key for the data
    /// * `input` - The record batch to write
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `PositionDeleteInput` with the file path and position for each record
    ///
    /// # Errors
    ///
    /// Returns an error for the following cases:
    /// - If the writer is not initialized
    /// - If writing fails
    /// - If the position exceeds `i64::MAX`, which would cause overflow in position deletes
    pub async fn write_with_position(
        &mut self,
        partition_key: &Option<PartitionKey>,
        input: &RecordBatch,
    ) -> Result<Vec<PositionDeleteInput>> {
        let writer = self.ensure_partition_writer(partition_key).await?;

        let num_rows = input.num_rows();
        let file_path: Arc<str> = Arc::from(writer.current_file_path());
        let start_pos = writer.current_row_num();
        if start_pos.saturating_add(num_rows) > i64::MAX as usize {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "position overflow: file has more than 2^63 - 1 rows",
            ));
        }

        writer.write(input).await?;

        // Generate position delete inputs for the entire batch
        let positions = (0..num_rows)
            .map(|i| PositionDeleteInput::new(file_path.clone(), (start_pos + i) as i64))
            .collect();
        Ok(positions)
    }

    /// Closes the writer and returns all data file builders.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `DataFileBuilder` instances representing
    /// all files that were written, including any that were created due to rollover
    pub async fn close(self) -> Result<Vec<DataFileBuilder>> {
        self.settle(RollingCloseReason::Final).await
    }

    /// Closes the writer as cancellation and returns everything it produced.
    ///
    /// Behaviourally identical to [`Self::close`] — the currently open output is
    /// closed and every outstanding background close is drained — so a cancelled
    /// attempt still surfaces every object it may have created rather than
    /// abandoning them untracked. The only difference is the reason recorded on
    /// the settlement events, which lets a caller distinguish "this residue is
    /// the natural end of a partition" from "this residue exists because I was
    /// cancelled and must be treated as unpublished evidence".
    ///
    /// # Errors
    ///
    /// Returns the first close error encountered, after draining every
    /// outstanding close.
    pub async fn cancel(self) -> Result<Vec<DataFileBuilder>> {
        self.settle(RollingCloseReason::Cancel).await
    }

    /// Shared terminal path for [`Self::close`] and [`Self::cancel`].
    ///
    /// # Errors
    ///
    /// Returns the first close error encountered. Every outstanding close is
    /// still awaited before returning, so no writer task outlives this call.
    async fn settle(mut self, reason: RollingCloseReason) -> Result<Vec<DataFileBuilder>> {
        let mut first_error = None;

        // close the current writer and merge the output
        let written_size_estimate = self.current_written_size();
        if let Some(current_writer) = self.inner.take() {
            let output = self.decide_close(reason, written_size_estimate);
            let result = current_writer.close().await;
            match output {
                Some((logical_ordinal, path)) => {
                    if let Err(err) = self.record_close(logical_ordinal, path, reason, result) {
                        first_error = Some(err);
                    }
                }
                None => match result {
                    Ok(files) => self.data_file_builders.extend(files),
                    Err(err) => first_error = Some(err),
                },
            }
        }

        while !self.close_futures.is_empty() {
            if let Err(err) = self.wait_for_one_close().await
                && first_error.is_none()
            {
                first_error = Some(err);
            }
        }

        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(self.data_file_builders)
        }
    }
}

impl<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> CurrentFileStatus
    for RollingFileWriter<B, L, F>
{
    fn current_file_path(&self) -> String {
        self.inner.as_ref().unwrap().current_file_path()
    }

    fn current_row_num(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |inner| inner.current_row_num())
    }

    fn current_written_size(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |inner| inner.current_written_size())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};

    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::properties::WriterProperties;
    use rand::prelude::IteratorRandom;
    use tempfile::TempDir;
    use tokio::sync::oneshot;
    use tokio::time::{Duration, timeout};

    use super::*;
    use crate::io::{FileIO, OutputFile};
    use crate::spec::{DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, Type};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::tests::check_parquet_data_file;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder, RecordBatch};

    fn make_test_schema() -> Result<Schema> {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
    }

    fn make_test_arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                1.to_string(),
            )])),
            Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                2.to_string(),
            )])),
        ])
    }

    #[tokio::test]
    async fn test_rolling_writer_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Create schema
        let schema = make_test_schema()?;

        // Create writer builders
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema));

        // Set a large target size so no rolling occurs
        let rolling_file_writer_builder = RollingFileWriterBuilder::new(
            parquet_writer_builder,
            1024 * 1024,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_file_writer_builder);

        // Create writer
        let mut writer = data_file_writer_builder.build(None).await?;

        // Create test data
        let arrow_schema = make_test_arrow_schema();

        let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])?;

        // Write data
        writer.write(batch.clone()).await?;

        // Close writer and get data files
        let data_files = writer.close().await?;

        // Verify only one file was created
        assert_eq!(
            data_files.len(),
            1,
            "Expected only one data file to be created"
        );

        // Verify file content
        check_parquet_data_file(&file_io, &data_files[0], &batch).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_rolling_writer_with_rolling() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Create schema
        let schema = make_test_schema()?;

        // Create writer builders
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema));

        // Set a very small target size to trigger rolling
        let rolling_writer_builder = RollingFileWriterBuilder::new(
            parquet_writer_builder,
            1024,
            file_io,
            location_gen,
            file_name_gen,
        );

        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);

        // Create writer
        let mut writer = data_file_writer_builder.build(None).await?;

        // Create test data
        let arrow_schema = make_test_arrow_schema();
        let arrow_schema_ref = Arc::new(arrow_schema.clone());

        let names = vec![
            "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
            "Kelly", "Larry", "Mallory", "Shawn",
        ];

        let mut rng = rand::rng();
        let batch_num = 10;
        let batch_rows = 100;
        let expected_rows = batch_num * batch_rows;

        for i in 0..batch_num {
            let int_values: Vec<i32> = (0..batch_rows).map(|row| i * batch_rows + row).collect();
            let str_values: Vec<&str> = (0..batch_rows)
                .map(|_| *names.iter().choose(&mut rng).unwrap())
                .collect();

            let int_array = Arc::new(Int32Array::from(int_values)) as ArrayRef;
            let str_array = Arc::new(StringArray::from(str_values)) as ArrayRef;

            let batch =
                RecordBatch::try_new(Arc::clone(&arrow_schema_ref), vec![int_array, str_array])
                    .expect("Failed to create RecordBatch");

            writer.write(batch).await?;
        }

        // Close writer and get data files
        let data_files = writer.close().await?;

        // Verify multiple files were created (at least 4)
        assert!(
            data_files.len() > 4,
            "Expected at least 4 data files to be created, but got {}",
            data_files.len()
        );

        // Verify total record count across all files
        let total_records: u64 = data_files.iter().map(|file| file.record_count).sum();
        assert_eq!(
            total_records, expected_rows as u64,
            "Expected {expected_rows} total records across all files"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rolling_writer_with_rolling_and_background_close() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema = make_test_schema()?;
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema));

        let rolling_writer_builder = RollingFileWriterBuilder::new(
            parquet_writer_builder,
            1024,
            file_io,
            location_gen,
            file_name_gen,
        )
        .with_max_concurrent_closes(2);

        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);
        let mut writer = data_file_writer_builder.build(None).await?;

        let arrow_schema = make_test_arrow_schema();
        let arrow_schema_ref = Arc::new(arrow_schema.clone());

        let names = vec![
            "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
            "Kelly", "Larry", "Mallory", "Shawn",
        ];

        let mut rng = rand::rng();
        let batch_num = 10;
        let batch_rows = 100;
        let expected_rows = batch_num * batch_rows;

        for i in 0..batch_num {
            let int_values: Vec<i32> = (0..batch_rows).map(|row| i * batch_rows + row).collect();
            let str_values: Vec<&str> = (0..batch_rows)
                .map(|_| *names.iter().choose(&mut rng).unwrap())
                .collect();

            let int_array = Arc::new(Int32Array::from(int_values)) as ArrayRef;
            let str_array = Arc::new(StringArray::from(str_values)) as ArrayRef;

            let batch =
                RecordBatch::try_new(Arc::clone(&arrow_schema_ref), vec![int_array, str_array])
                    .expect("Failed to create RecordBatch");

            writer.write(batch).await?;
        }

        let data_files = writer.close().await?;

        assert!(
            data_files.len() > 4,
            "Expected at least 4 data files to be created, but got {}",
            data_files.len()
        );

        let total_records: u64 = data_files.iter().map(|file| file.record_count).sum();
        assert_eq!(
            total_records, expected_rows as u64,
            "Expected {expected_rows} total records across all files"
        );

        Ok(())
    }

    #[derive(Clone)]
    struct MockFileWriterBuilder {
        close_behaviors: Arc<Mutex<VecDeque<CloseBehavior>>>,
        next_id: Arc<Mutex<usize>>,
        closed_ids: Arc<Mutex<Vec<usize>>>,
    }

    impl MockFileWriterBuilder {
        fn new(close_behaviors: Vec<CloseBehavior>) -> Self {
            Self {
                close_behaviors: Arc::new(Mutex::new(close_behaviors.into())),
                next_id: Arc::new(Mutex::new(0)),
                closed_ids: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn closed_ids(&self) -> Arc<Mutex<Vec<usize>>> {
            Arc::clone(&self.closed_ids)
        }
    }

    enum CloseBehavior {
        Ok,
        Wait(oneshot::Receiver<()>),
        Fail(&'static str),
    }

    struct MockFileWriter {
        id: usize,
        written_size: usize,
        close_behavior: CloseBehavior,
        closed_ids: Arc<Mutex<Vec<usize>>>,
    }

    impl FileWriterBuilder for MockFileWriterBuilder {
        type R = MockFileWriter;

        async fn build(&self, _output_file: OutputFile) -> Result<Self::R> {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;

            let close_behavior = self
                .close_behaviors
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(CloseBehavior::Ok);

            Ok(MockFileWriter {
                id,
                written_size: 0,
                close_behavior,
                closed_ids: Arc::clone(&self.closed_ids),
            })
        }
    }

    impl CurrentFileStatus for MockFileWriter {
        fn current_file_path(&self) -> String {
            format!("mock://{}", self.id)
        }

        fn current_row_num(&self) -> usize {
            self.id + 1
        }

        fn current_written_size(&self) -> usize {
            self.written_size
        }
    }

    impl FileWriter for MockFileWriter {
        async fn write(&mut self, _batch: &RecordBatch) -> Result<()> {
            self.written_size = 32;
            Ok(())
        }

        async fn close(mut self) -> Result<Vec<DataFileBuilder>> {
            self.closed_ids.lock().unwrap().push(self.id);

            match std::mem::replace(&mut self.close_behavior, CloseBehavior::Ok) {
                CloseBehavior::Ok => {}
                CloseBehavior::Wait(receiver) => {
                    receiver.await.expect("close sender dropped");
                }
                CloseBehavior::Fail(message) => {
                    return Err(Error::new(ErrorKind::Unexpected, message));
                }
            }

            let mut builder = DataFileBuilder::default();
            builder.content(DataContentType::Data);
            builder.file_path(format!("mock://{}", self.id));
            builder.file_format(DataFileFormat::Parquet);
            builder.record_count(self.id as u64);
            builder.file_size_in_bytes((self.id + 1) as u64);

            Ok(vec![builder])
        }
    }

    #[tokio::test]
    async fn test_background_close_collects_all_files() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_string_lossy().into_owned(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let (tx0, rx0) = oneshot::channel();
        let (tx1, rx1) = oneshot::channel();

        let builder = RollingFileWriterBuilder::new(
            MockFileWriterBuilder::new(vec![
                CloseBehavior::Wait(rx0),
                CloseBehavior::Wait(rx1),
                CloseBehavior::Ok,
            ]),
            16,
            file_io,
            location_gen,
            file_name_gen,
        )
        .with_max_concurrent_closes(2);

        let mut writer = builder.build();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])?;

        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;

        tx1.send(()).unwrap();
        tx0.send(()).unwrap();

        let data_files = writer.close().await?;
        let mut record_counts: Vec<u64> = data_files
            .into_iter()
            .map(|file| file.build().unwrap().record_count())
            .collect();
        record_counts.sort_unstable();

        assert_eq!(record_counts, vec![0, 1, 2]);
        Ok(())
    }

    #[tokio::test]
    async fn test_waiting_for_pending_close_can_resume_rolling() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_string_lossy().into_owned(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let (tx0, rx0) = oneshot::channel();

        let builder = RollingFileWriterBuilder::new(
            MockFileWriterBuilder::new(vec![
                CloseBehavior::Wait(rx0),
                CloseBehavior::Ok,
                CloseBehavior::Ok,
            ]),
            16,
            file_io,
            location_gen,
            file_name_gen,
        )
        .with_max_concurrent_closes(1);

        let mut writer = builder.build();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])?;

        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;

        let batch_for_task = batch.clone();
        let mut blocked_write = tokio::spawn(async move {
            writer.write(&None, &batch_for_task).await?;
            Ok::<RollingFileWriter<MockFileWriterBuilder, _, _>, Error>(writer)
        });

        assert!(
            timeout(Duration::from_millis(50), &mut blocked_write)
                .await
                .is_err()
        );

        tx0.send(()).unwrap();

        let writer = blocked_write.await.unwrap()?;
        let data_files = writer.close().await?;
        let mut record_counts: Vec<u64> = data_files
            .into_iter()
            .map(|file| file.build().unwrap().record_count())
            .collect();
        record_counts.sort_unstable();

        assert_eq!(record_counts, vec![0, 1, 2]);
        Ok(())
    }

    #[tokio::test]
    async fn test_failed_pending_close_does_not_drop_active_writer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_string_lossy().into_owned(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let mock_builder = MockFileWriterBuilder::new(vec![
            CloseBehavior::Fail("background close failed"),
            CloseBehavior::Ok,
        ]);
        let closed_ids = mock_builder.closed_ids();

        let builder =
            RollingFileWriterBuilder::new(mock_builder, 16, file_io, location_gen, file_name_gen)
                .with_max_concurrent_closes(1);

        let mut writer = builder.build();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])?;

        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;
        assert!(writer.write(&None, &batch).await.is_err());
        let data_files = writer.close().await?;

        let record_counts: Vec<u64> = data_files
            .into_iter()
            .map(|file| file.build().unwrap().record_count())
            .collect();
        assert_eq!(record_counts, vec![1]);

        let mut closed_ids = closed_ids.lock().unwrap().clone();
        closed_ids.sort_unstable();
        assert_eq!(closed_ids, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_close_still_closes_active_writer_after_pending_failure() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_string_lossy().into_owned(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let mock_builder = MockFileWriterBuilder::new(vec![
            CloseBehavior::Fail("background close failed"),
            CloseBehavior::Ok,
        ]);
        let closed_ids = mock_builder.closed_ids();

        let builder =
            RollingFileWriterBuilder::new(mock_builder, 16, file_io, location_gen, file_name_gen)
                .with_max_concurrent_closes(2);

        let mut writer = builder.build();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])?;

        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;
        assert!(writer.close().await.is_err());

        let mut closed_ids = closed_ids.lock().unwrap().clone();
        closed_ids.sort_unstable();
        assert_eq!(closed_ids, vec![0, 1]);
        Ok(())
    }

    /// Collects every observed event in emission order for golden comparison.
    #[derive(Debug, Default)]
    struct RecordingObserver {
        events: Mutex<Vec<RollingWriterEvent>>,
    }

    impl RecordingObserver {
        /// Returns the events observed so far, in emission order.
        fn events(&self) -> Vec<RollingWriterEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl RollingWriterObserver for RecordingObserver {
        fn on_event(&self, event: RollingWriterEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    /// Reduces a settled run to the values that define its physical result.
    ///
    /// Used to prove an observed run and an unobserved run are identical: same
    /// files, same values, same order.
    fn output_signature(files: Vec<DataFileBuilder>) -> Vec<(String, u64, u64)> {
        files
            .into_iter()
            .map(|builder| {
                let file = builder.build().expect("mock builder is complete");
                (
                    file.file_path().to_string(),
                    file.record_count(),
                    file.file_size_in_bytes(),
                )
            })
            .collect()
    }

    /// Builds a rolling writer over the deterministic mock file writer.
    fn mock_rolling_builder(
        temp_dir: &TempDir,
        behaviors: Vec<CloseBehavior>,
        target: usize,
        max_concurrent_closes: usize,
    ) -> RollingFileWriterBuilder<
        MockFileWriterBuilder,
        DefaultLocationGenerator,
        DefaultFileNameGenerator,
    > {
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_string_lossy().into_owned(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        RollingFileWriterBuilder::new(
            MockFileWriterBuilder::new(behaviors),
            target,
            FileIO::new_with_fs(),
            location_gen,
            file_name_gen,
        )
        .with_max_concurrent_closes(max_concurrent_closes)
    }

    /// One single-column batch, enough to make the mock writer report a size.
    fn one_row_batch() -> Result<RecordBatch> {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        Ok(RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        ])?)
    }

    /// Drives three writes through a fresh writer, optionally observed.
    async fn run_three_writes(
        temp_dir: &TempDir,
        behaviors: Vec<CloseBehavior>,
        target: usize,
        max_concurrent_closes: usize,
        observer: Option<Arc<RecordingObserver>>,
        cancel_instead_of_close: bool,
    ) -> Result<Vec<DataFileBuilder>> {
        let mut builder = mock_rolling_builder(temp_dir, behaviors, target, max_concurrent_closes);
        if let Some(observer) = observer {
            builder = builder.with_observer(observer);
        }
        let mut writer = builder.build();
        let batch = one_row_batch()?;
        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;
        writer.write(&None, &batch).await?;
        if cancel_instead_of_close {
            writer.cancel().await
        } else {
            writer.close().await
        }
    }

    #[tokio::test]
    async fn rolling_writer_observer_reports_threshold_and_final_in_logical_order() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let unobserved =
            output_signature(run_three_writes(&temp_dir, vec![], 16, 0, None, false).await?);

        let observer = Arc::new(RecordingObserver::default());
        let observed = output_signature(
            run_three_writes(&temp_dir, vec![], 16, 0, Some(Arc::clone(&observer)), false).await?,
        );

        assert_eq!(
            observed, unobserved,
            "installing an observer must not change the files, their values, or their order"
        );

        // Three outputs: two closed because the estimate crossed the target and
        // one final residue, in strict open order, each fully settled before the
        // next output is opened (synchronous close).
        let events = observer.events();
        let mut reasons = Vec::new();
        for (index, chunk) in events.chunks(3).enumerate() {
            let expected_ordinal = index as u64;
            let [
                RollingWriterEvent::OutputOpened {
                    logical_ordinal: opened,
                    path: opened_path,
                },
                RollingWriterEvent::CloseDecided {
                    logical_ordinal: decided,
                    path: decided_path,
                    reason,
                    target_file_size,
                    written_size_estimate,
                },
                RollingWriterEvent::CloseSettled {
                    logical_ordinal: settled,
                    completion_ordinal,
                    path: settled_path,
                    reason: settled_reason,
                    output_files,
                },
            ] = chunk
            else {
                panic!("expected open/decide/settle per output, got {chunk:?}");
            };
            assert_eq!(*opened, expected_ordinal);
            assert_eq!(*decided, expected_ordinal);
            assert_eq!(*settled, expected_ordinal);
            assert_eq!(*completion_ordinal, expected_ordinal);
            assert_eq!(opened_path, decided_path);
            assert_eq!(opened_path, settled_path);
            assert_eq!(*target_file_size, 16);
            assert_eq!(reason, settled_reason);
            assert_eq!(*output_files, Some(1));
            if *reason == RollingCloseReason::Threshold {
                assert!(
                    written_size_estimate > target_file_size,
                    "a target-triggered close must observe an estimate above the target"
                );
            }
            reasons.push(*reason);
        }
        assert_eq!(reasons, vec![
            RollingCloseReason::Threshold,
            RollingCloseReason::Threshold,
            RollingCloseReason::Final,
        ]);
        Ok(())
    }

    #[tokio::test]
    async fn rolling_writer_observer_preserves_outputs_with_concurrent_closes() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let behaviors = || {
            let (tx0, rx0) = oneshot::channel();
            let (tx1, rx1) = oneshot::channel();
            tx0.send(()).unwrap();
            tx1.send(()).unwrap();
            vec![
                CloseBehavior::Wait(rx0),
                CloseBehavior::Wait(rx1),
                CloseBehavior::Ok,
            ]
        };

        let mut unobserved =
            output_signature(run_three_writes(&temp_dir, behaviors(), 16, 2, None, false).await?);
        let observer = Arc::new(RecordingObserver::default());
        let mut observed = output_signature(
            run_three_writes(
                &temp_dir,
                behaviors(),
                16,
                2,
                Some(Arc::clone(&observer)),
                false,
            )
            .await?,
        );
        unobserved.sort();
        observed.sort();
        assert_eq!(
            observed, unobserved,
            "background closes must produce the same file set with and without an observer"
        );

        let events = observer.events();
        let opened: Vec<u64> = events
            .iter()
            .filter_map(|event| match event {
                RollingWriterEvent::OutputOpened {
                    logical_ordinal, ..
                } => Some(*logical_ordinal),
                _ => None,
            })
            .collect();
        assert_eq!(
            opened,
            vec![0, 1, 2],
            "logical ordinals are assigned in open order before any close is spawned"
        );

        let settled: Vec<(u64, u64)> = events
            .iter()
            .filter_map(|event| match event {
                RollingWriterEvent::CloseSettled {
                    logical_ordinal,
                    completion_ordinal,
                    ..
                } => Some((*logical_ordinal, *completion_ordinal)),
                _ => None,
            })
            .collect();
        assert_eq!(settled.len(), 3, "every opened output settles exactly once");
        let mut logical: Vec<u64> = settled.iter().map(|(logical, _)| *logical).collect();
        let mut completion: Vec<u64> = settled.iter().map(|(_, order)| *order).collect();
        logical.sort_unstable();
        completion.sort_unstable();
        assert_eq!(logical, vec![0, 1, 2]);
        assert_eq!(
            completion,
            vec![0, 1, 2],
            "completion ordinals are a dense permutation of the settled closes"
        );
        assert!(
            settled
                .iter()
                .any(|(logical, completion)| logical != completion),
            "concurrent closes must settle out of open order, proving the two ordinals are tracked separately"
        );
        Ok(())
    }

    #[tokio::test]
    async fn rolling_writer_observer_reports_cancel_and_error_without_semantic_change() -> Result<()>
    {
        let temp_dir = TempDir::new()?;

        // Cancellation drains exactly the same outputs a close would, and is
        // distinguishable only by the recorded reason.
        let closed =
            output_signature(run_three_writes(&temp_dir, vec![], 16, 0, None, false).await?);
        let observer = Arc::new(RecordingObserver::default());
        let cancelled = output_signature(
            run_three_writes(&temp_dir, vec![], 16, 0, Some(Arc::clone(&observer)), true).await?,
        );
        assert_eq!(
            cancelled, closed,
            "cancellation must surface every produced output as evidence, not discard it"
        );
        let terminal_reason = observer
            .events()
            .into_iter()
            .filter_map(|event| match event {
                RollingWriterEvent::CloseSettled { reason, .. } => Some(reason),
                _ => None,
            })
            .next_back()
            .expect("cancelled writer settles its open output");
        assert_eq!(terminal_reason, RollingCloseReason::Cancel);

        // A failing close is reported as an Error settlement with no outputs,
        // and the returned error is unchanged by observation.
        let failing = || vec![CloseBehavior::Fail("mock close failure")];
        let unobserved_error = run_three_writes(&temp_dir, failing(), 16, 0, None, false)
            .await
            .err()
            .expect("the first close fails");
        let observer = Arc::new(RecordingObserver::default());
        let observed_error = run_three_writes(
            &temp_dir,
            failing(),
            16,
            0,
            Some(Arc::clone(&observer)),
            false,
        )
        .await
        .err()
        .expect("the first close fails");
        assert_eq!(observed_error.kind(), unobserved_error.kind());
        assert_eq!(observed_error.message(), unobserved_error.message());

        let settled: Vec<(RollingCloseReason, Option<usize>)> = observer
            .events()
            .into_iter()
            .filter_map(|event| match event {
                RollingWriterEvent::CloseSettled {
                    reason,
                    output_files,
                    ..
                } => Some((reason, output_files)),
                _ => None,
            })
            .collect();
        assert_eq!(settled, vec![(RollingCloseReason::Error, None)]);
        Ok(())
    }
}
