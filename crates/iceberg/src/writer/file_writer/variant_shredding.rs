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

//! Per-file Variant shredding chosen from a bounded prefix of the file's own rows.
//!
//! A writer that shreds cannot open its Parquet encoder until it knows the
//! physical layout, and it cannot know the layout until it has seen rows. A
//! [`VariantPrefix`] holds the file's first rows within a row and byte bound,
//! samples them, and then hands back the inferred [`VariantLayout`] together
//! with the rows to replay once into the opened encoder. It performs no IO, so
//! every writer that shreds shares it and keeps its own IO.
//!
//! Arrow owns the standard encoding: the layout is built with
//! [`ShreddedSchemaBuilder`] and applied with [`shred_variant`]. This module
//! only decides which object fields to shred and as which type. Only top-level
//! Variant columns are shredded; arrays always stay in the residual `value`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::extension::ExtensionType;
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::variant::{
    ShreddedSchemaBuilder, Variant, VariantArray, VariantPath, VariantPathElement, VariantType,
    shred_variant,
};

use super::{FileWriter, FileWriterBuilder, ParquetWriter, ParquetWriterBuilder};
use crate::io::OutputFile;
use crate::spec::DataFileBuilder;
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

/// Internal bounds and thresholds that decide one file's Variant layout.
///
/// The values are supplied by the owning writer at construction. They are
/// policy constants, not table or user configuration. The default has zero
/// bounds: it samples nothing, so files are never shredded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VariantShreddingPolicy {
    /// Rows retained before the layout is inferred.
    pub max_rows: usize,
    /// Retained Arrow memory, in the writer's own measure, before the layout is inferred.
    pub max_bytes: usize,
    /// Minimum share of sampled non-null root values, in percent, that must contain a field.
    pub min_frequency_percent: usize,
    /// Distinct child names tracked per object node; later names are ignored.
    pub max_tracked_children: usize,
    /// Children kept per object node, by frequency then name.
    pub max_emitted_children: usize,
    /// Object depth below which traversal stops.
    pub max_depth: usize,
}

/// One physical Variant layout per top-level Variant column of a file.
///
/// An empty layout leaves every column unshredded. The layout is derived from
/// this file's rows only and is never shared with another file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantLayout {
    /// Column index and the shredding type passed to [`shred_variant`].
    columns: Vec<(usize, DataType)>,
}

impl VariantLayout {
    /// Whether no column is shredded.
    pub fn is_unshredded(&self) -> bool {
        self.columns.is_empty()
    }

    /// The shredding type chosen for column `index`, if it is shredded.
    pub fn shredding_type(&self, index: usize) -> Option<&DataType> {
        self.columns
            .iter()
            .find_map(|(column, data_type)| (*column == index).then_some(data_type))
    }

    /// The shredded physical schema of `logical` under this layout.
    ///
    /// Shreds an empty batch, so the schema is exactly the one
    /// [`Self::shred`] produces for real rows.
    ///
    /// # Errors
    ///
    /// Returns an error when Arrow refuses to shred a named column.
    pub fn physical_schema(&self, logical: &SchemaRef) -> Result<SchemaRef> {
        Ok(self
            .shred(&RecordBatch::new_empty(Arc::clone(logical)))?
            .schema())
    }

    /// Shreds every Variant column this layout names and returns the physical batch.
    ///
    /// Each shredded field keeps its name, nullability, and metadata (field id
    /// and Variant extension); only its storage type becomes the standard
    /// `metadata`/`value`/`typed_value` struct. Values that do not fit the
    /// chosen type are kept in the residual `value`, so the logical values are
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when a named column is not valid Variant storage or
    /// Arrow refuses to shred it.
    pub fn shred(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if self.is_unshredded() {
            return Ok(batch.clone());
        }
        let schema = batch.schema();
        let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (index, shredding_type) in &self.columns {
            let variant = VariantArray::try_new(columns[*index].as_ref()).map_err(variant_error)?;
            let shredded: ArrayRef = shred_variant(&variant, shredding_type)
                .map_err(variant_error)?
                .into();
            fields[*index] = Arc::new(
                fields[*index]
                    .as_ref()
                    .clone()
                    .with_data_type(shredded.data_type().clone()),
            );
            columns[*index] = shredded;
        }
        let physical = Schema::new(fields).with_metadata(schema.metadata().clone());
        RecordBatch::try_new(Arc::new(physical), columns).map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to assemble shredded batch.").with_source(err)
        })
    }
}

/// What a writer does after offering one batch to its [`VariantPrefix`].
#[derive(Debug)]
pub enum PrefixStep {
    /// The whole batch was retained; keep sampling.
    Retained,
    /// A bound was reached: open the encoder with this layout and replay.
    Ready(ReadyPrefix),
}

/// The inferred layout and the rows still to be written, in order.
#[derive(Debug)]
pub struct ReadyPrefix {
    /// Physical layout for this file.
    pub layout: VariantLayout,
    /// Retained rows to write first, exactly once.
    pub replay: Vec<RecordBatch>,
    /// Rows of the offered batch that were not retained, written after `replay`.
    pub remainder: Option<RecordBatch>,
}

/// Retains one file's first rows within the policy bounds and infers its layout.
///
/// The writer offers every batch through [`Self::push`] until it gets
/// [`PrefixStep::Ready`], or calls [`Self::finish`] at close. A batch is
/// measured with the writer's own memory measure before it is retained. The
/// prefix stops before a batch would cross the byte bound; a first batch that
/// alone crosses it is retained as the progress exception. Retained batches
/// are zero-copy views of the offered batches and are handed back for replay,
/// so dropping the prefix releases them.
pub struct VariantPrefix {
    /// Bounds and thresholds for this file.
    policy: VariantShreddingPolicy,
    /// The writer's memory measure for one batch.
    measure: fn(&RecordBatch) -> usize,
    /// Sample statistics per top-level Variant column.
    analyzer: VariantLayoutAnalyzer,
    /// Retained batches in arrival order.
    retained: Vec<RecordBatch>,
    /// Rows retained so far.
    rows: usize,
    /// Measured bytes retained so far.
    bytes: usize,
}

impl VariantPrefix {
    /// Starts an empty prefix for files of `schema`.
    pub fn new(
        schema: &Schema,
        policy: VariantShreddingPolicy,
        measure: fn(&RecordBatch) -> usize,
    ) -> Self {
        Self {
            policy,
            measure,
            analyzer: VariantLayoutAnalyzer::new(schema, policy),
            retained: Vec::new(),
            rows: 0,
            bytes: 0,
        }
    }

    /// Whether the file has no Variant column, so no prefix is needed.
    pub fn is_inert(&self) -> bool {
        self.analyzer.columns.is_empty()
    }

    /// Rows retained so far.
    pub fn retained_rows(&self) -> usize {
        self.rows
    }

    /// Measured bytes retained so far.
    pub fn retained_bytes(&self) -> usize {
        self.bytes
    }

    /// Offers one batch.
    ///
    /// Retains as many leading rows as the row bound allows unless doing so
    /// would cross the byte bound with rows already retained. Returns
    /// [`PrefixStep::Ready`] once either bound is reached; the retained rows
    /// are then drained into the result and the prefix is empty.
    ///
    /// # Errors
    ///
    /// Returns an error when a Variant column cannot be read as Variant storage.
    pub fn push(&mut self, batch: &RecordBatch) -> Result<PrefixStep> {
        let take = batch
            .num_rows()
            .min(self.policy.max_rows.saturating_sub(self.rows));
        let head = batch.slice(0, take);
        let head_bytes = (self.measure)(&head);
        if !self.retained.is_empty()
            && self.bytes.saturating_add(head_bytes) > self.policy.max_bytes
        {
            return Ok(PrefixStep::Ready(self.drain(Some(batch.clone()))));
        }
        self.analyzer.observe(&head)?;
        self.retained.push(head);
        self.rows += take;
        self.bytes = self.bytes.saturating_add(head_bytes);
        if take < batch.num_rows()
            || self.rows >= self.policy.max_rows
            || self.bytes >= self.policy.max_bytes
        {
            let remainder =
                (take < batch.num_rows()).then(|| batch.slice(take, batch.num_rows() - take));
            return Ok(PrefixStep::Ready(self.drain(remainder)));
        }
        Ok(PrefixStep::Retained)
    }

    /// Infers from whatever was retained at close; `None` when nothing was.
    pub fn finish(mut self) -> Option<ReadyPrefix> {
        (!self.retained.is_empty()).then(|| self.drain(None))
    }

    /// Infers the layout and hands back the retained rows for replay.
    fn drain(&mut self, remainder: Option<RecordBatch>) -> ReadyPrefix {
        let layout = self.analyzer.layout();
        self.rows = 0;
        self.bytes = 0;
        ReadyPrefix {
            layout,
            replay: std::mem::take(&mut self.retained),
            remainder,
        }
    }
}

/// Sample statistics for every top-level Variant column of one file.
struct VariantLayoutAnalyzer {
    /// Bounds and thresholds for this file.
    policy: VariantShreddingPolicy,
    /// Column index and its root statistics.
    columns: Vec<(usize, ColumnSample)>,
}

/// Statistics for one top-level Variant column.
#[derive(Default)]
struct ColumnSample {
    /// Non-null root values sampled.
    roots: usize,
    /// Fields of root values that are objects.
    node: ObjectNode,
}

impl VariantLayoutAnalyzer {
    /// Finds the top-level Variant columns of `schema`.
    fn new(schema: &Schema, policy: VariantShreddingPolicy) -> Self {
        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.extension_type_name() == Some(VariantType::NAME))
            .map(|(index, _)| (index, ColumnSample::default()))
            .collect();
        Self { policy, columns }
    }

    /// Adds every non-null root value of `batch` to the sample.
    fn observe(&mut self, batch: &RecordBatch) -> Result<()> {
        for (index, sample) in &mut self.columns {
            let variant =
                VariantArray::try_new(batch.column(*index).as_ref()).map_err(variant_error)?;
            for row in 0..variant.len() {
                if variant.is_null(row) {
                    continue;
                }
                sample.roots += 1;
                if let Variant::Object(object) = variant.value(row) {
                    sample.node.observe_object(&object, 1, &self.policy);
                }
            }
        }
        Ok(())
    }

    /// Chooses each column's shredding type from the sample.
    fn layout(&self) -> VariantLayout {
        let columns = self
            .columns
            .iter()
            .filter_map(|(index, sample)| {
                let min_count = (sample.roots * self.policy.min_frequency_percent).div_ceil(100);
                let mut path = Vec::new();
                let builder = sample.node.emit(
                    ShreddedSchemaBuilder::new(),
                    &mut path,
                    min_count.max(1),
                    &self.policy,
                );
                match builder.build() {
                    DataType::Null => None,
                    shredding_type => Some((*index, shredding_type)),
                }
            })
            .collect();
        VariantLayout { columns }
    }
}

/// Child statistics of one object path.
#[derive(Default)]
struct ObjectNode {
    /// Child name to its statistics, alphabetical by unsigned UTF-8 bytes.
    children: BTreeMap<String, Child>,
}

/// How often one child appeared and which family its values share.
struct Child {
    /// Root values containing this child.
    count: usize,
    /// Merged family of every non-null value seen.
    family: Family,
}

/// The shreddable type family of a path's values, merged across the sample.
enum Family {
    /// Only nulls seen so far; nulls do not choose a type.
    Unknown,
    /// No single compatible family; the path stays residual.
    Residual,
    /// An object with its own children.
    Object(ObjectNode),
    /// A scalar family and its widest observed type.
    Scalar(Scalar),
}

/// Shreddable scalar types; integers and decimals widen within their family.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Boolean,
    /// Integer of this many bytes.
    Int(u8),
    /// Decimal with this many integer digits and this scale.
    Decimal {
        integer_digits: u8,
        scale: u8,
    },
    Float,
    Double,
    Date,
    Time,
    Timestamp {
        nanos: bool,
        utc: bool,
    },
    Binary,
    String,
    Uuid,
}

impl ObjectNode {
    /// Records the children of one object value at `depth`.
    fn observe_object(
        &mut self,
        object: &parquet::variant::VariantObject<'_, '_>,
        depth: usize,
        policy: &VariantShreddingPolicy,
    ) {
        for (name, value) in object.iter() {
            let tracked = self.children.len();
            let child = match self.children.get_mut(name) {
                Some(child) => child,
                None if tracked < policy.max_tracked_children => {
                    self.children.entry(name.to_owned()).or_insert(Child {
                        count: 0,
                        family: Family::Unknown,
                    })
                }
                None => continue,
            };
            if matches!(value, Variant::Null) {
                continue;
            }
            child.count += 1;
            child.family.merge(&value, depth + 1, policy);
        }
    }

    /// Adds this node's kept children to `builder` under `path`.
    ///
    /// Children seen in fewer than `min_count` root values or without one
    /// compatible family are skipped; the remaining ones are capped by
    /// frequency, ties broken by name, and emitted alphabetically.
    fn emit<'a>(
        &'a self,
        mut builder: ShreddedSchemaBuilder,
        path: &mut Vec<&'a str>,
        min_count: usize,
        policy: &VariantShreddingPolicy,
    ) -> ShreddedSchemaBuilder {
        let mut kept: Vec<(&'a String, &'a Child)> = self
            .children
            .iter()
            .filter(|(_, child)| {
                child.count >= min_count
                    && matches!(child.family, Family::Object(_) | Family::Scalar(_))
            })
            .collect();
        kept.sort_by(|(left_name, left), (right_name, right)| {
            right.count.cmp(&left.count).then(left_name.cmp(right_name))
        });
        kept.truncate(policy.max_emitted_children);
        kept.sort_by_key(|(name, _)| *name);
        for (name, child) in kept {
            path.push(name);
            builder = match &child.family {
                Family::Object(node) => node.emit(builder, path, min_count, policy),
                Family::Scalar(scalar) => {
                    let variant_path: VariantPath<'_> = path
                        .iter()
                        .map(|segment| VariantPathElement::field(*segment))
                        .collect();
                    builder
                        .with_path(variant_path, scalar.data_type())
                        .expect("a VariantPath always converts to itself")
                }
                Family::Unknown | Family::Residual => builder,
            };
            path.pop();
        }
        builder
    }
}

impl Family {
    /// Merges one non-null value observed at `depth` into this family.
    fn merge(&mut self, value: &Variant<'_, '_>, depth: usize, policy: &VariantShreddingPolicy) {
        match (&mut *self, value) {
            (Family::Residual, _) => {}
            (_, Variant::List(_)) => *self = Family::Residual,
            (_, Variant::Object(_)) if depth > policy.max_depth => *self = Family::Residual,
            (Family::Unknown, Variant::Object(object)) => {
                let mut node = ObjectNode::default();
                node.observe_object(object, depth, policy);
                *self = Family::Object(node);
            }
            (Family::Object(node), Variant::Object(object)) => {
                node.observe_object(object, depth, policy)
            }
            (Family::Object(_), _) => *self = Family::Residual,
            (Family::Unknown, scalar) => {
                *self = Scalar::of(scalar).map_or(Family::Residual, Family::Scalar)
            }
            (Family::Scalar(current), scalar) => {
                *self = Scalar::of(scalar)
                    .and_then(|next| current.widen(next))
                    .map_or(Family::Residual, Family::Scalar)
            }
        }
    }
}

impl Scalar {
    /// The family of one non-null, non-container value.
    fn of(value: &Variant<'_, '_>) -> Option<Self> {
        Some(match value {
            Variant::BooleanTrue | Variant::BooleanFalse => Scalar::Boolean,
            Variant::Int8(_) => Scalar::Int(1),
            Variant::Int16(_) => Scalar::Int(2),
            Variant::Int32(_) => Scalar::Int(4),
            Variant::Int64(_) => Scalar::Int(8),
            Variant::Decimal4(decimal) => Scalar::decimal(9, decimal.scale()),
            Variant::Decimal8(decimal) => Scalar::decimal(18, decimal.scale()),
            Variant::Decimal16(decimal) => Scalar::decimal(38, decimal.scale()),
            Variant::Float(_) => Scalar::Float,
            Variant::Double(_) => Scalar::Double,
            Variant::Date(_) => Scalar::Date,
            Variant::Time(_) => Scalar::Time,
            Variant::TimestampMicros(_) => Scalar::Timestamp {
                nanos: false,
                utc: true,
            },
            Variant::TimestampNtzMicros(_) => Scalar::Timestamp {
                nanos: false,
                utc: false,
            },
            Variant::TimestampNanos(_) => Scalar::Timestamp {
                nanos: true,
                utc: true,
            },
            Variant::TimestampNtzNanos(_) => Scalar::Timestamp {
                nanos: true,
                utc: false,
            },
            Variant::Binary(_) => Scalar::Binary,
            Variant::String(_) | Variant::ShortString(_) => Scalar::String,
            Variant::Uuid(_) => Scalar::Uuid,
            Variant::Null | Variant::Object(_) | Variant::List(_) => return None,
        })
    }

    /// A decimal of the given storage precision and scale.
    fn decimal(precision: u8, scale: u8) -> Self {
        Scalar::Decimal {
            integer_digits: precision.saturating_sub(scale),
            scale,
        }
    }

    /// The narrowest type of the same family holding both; `None` across families.
    fn widen(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Scalar::Int(left), Scalar::Int(right)) => Some(Scalar::Int(left.max(right))),
            (
                Scalar::Decimal {
                    integer_digits: left_digits,
                    scale: left_scale,
                },
                Scalar::Decimal {
                    integer_digits: right_digits,
                    scale: right_scale,
                },
            ) => {
                let integer_digits = left_digits.max(right_digits);
                let scale = left_scale.max(right_scale);
                (integer_digits + scale <= 38).then_some(Scalar::Decimal {
                    integer_digits,
                    scale,
                })
            }
            (left, right) if left == right => Some(left),
            _ => None,
        }
    }

    /// The Arrow type this family shreds to.
    fn data_type(self) -> DataType {
        match self {
            Scalar::Boolean => DataType::Boolean,
            Scalar::Int(1) => DataType::Int8,
            Scalar::Int(2) => DataType::Int16,
            Scalar::Int(4) => DataType::Int32,
            Scalar::Int(_) => DataType::Int64,
            Scalar::Decimal {
                integer_digits,
                scale,
            } => {
                let precision = integer_digits + scale;
                let scale = scale as i8;
                match precision {
                    0..=9 => DataType::Decimal32(precision, scale),
                    10..=18 => DataType::Decimal64(precision, scale),
                    _ => DataType::Decimal128(precision, scale),
                }
            }
            Scalar::Float => DataType::Float32,
            Scalar::Double => DataType::Float64,
            Scalar::Date => DataType::Date32,
            Scalar::Time => DataType::Time64(TimeUnit::Microsecond),
            Scalar::Timestamp { nanos, utc } => DataType::Timestamp(
                if nanos {
                    TimeUnit::Nanosecond
                } else {
                    TimeUnit::Microsecond
                },
                utc.then(|| "+00:00".into()),
            ),
            Scalar::Binary => DataType::Binary,
            Scalar::String => DataType::Utf8,
            Scalar::Uuid => DataType::FixedSizeBinary(16),
        }
    }
}

/// Wraps an Arrow Variant error as an Iceberg data error.
fn variant_error(err: arrow_schema::ArrowError) -> Error {
    Error::new(ErrorKind::DataInvalid, "Invalid Variant column.").with_source(err)
}

/// Builds Parquet writers that choose each file's Variant layout from that file's first rows.
///
/// Every [`FileWriterBuilder::build`] starts a fresh [`VariantParquetWriter`]
/// with an empty prefix, so each rolled output infers independently. The
/// builder carries only the wrapped Parquet builder and the policy constants.
#[derive(Clone, Debug)]
pub struct VariantParquetWriterBuilder {
    /// Ordinary Parquet builder the deferred writer opens once the layout is known.
    inner: ParquetWriterBuilder,
    /// Bounds and thresholds for every file.
    policy: VariantShreddingPolicy,
}

impl VariantParquetWriterBuilder {
    /// Wraps `inner` so its files shred Variant columns under `policy`.
    pub fn new(inner: ParquetWriterBuilder, policy: VariantShreddingPolicy) -> Self {
        Self { inner, policy }
    }
}

impl FileWriterBuilder for VariantParquetWriterBuilder {
    type R = VariantParquetWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        let logical: SchemaRef = Arc::new(self.inner.schema().as_ref().try_into()?);
        Ok(VariantParquetWriter {
            prefix: Some((
                VariantPrefix::new(&logical, self.policy, RecordBatch::get_array_memory_size),
                output_file,
            )),
            builder: self.inner.clone(),
            logical,
            open: None,
        })
    }
}

/// A Parquet writer that opens its encoder only after its Variant layout is inferred.
///
/// Until a policy bound is reached the writer only retains rows; no file is
/// created. It then opens an ordinary [`ParquetWriter`] with the shredded
/// physical schema, replays the retained rows once, and streams the rest.
pub struct VariantParquetWriter {
    /// Ordinary Parquet builder used to open the encoder.
    builder: ParquetWriterBuilder,
    /// Logical Arrow schema of the Iceberg table.
    logical: SchemaRef,
    /// Rows retained before the encoder opens, and the file they will go to.
    prefix: Option<(VariantPrefix, OutputFile)>,
    /// The open encoder and the layout every later batch is shredded with.
    open: Option<(ParquetWriter, VariantLayout)>,
}

impl VariantParquetWriter {
    /// Opens the encoder for `ready.layout` and writes the retained rows, then the remainder.
    ///
    /// # Errors
    ///
    /// Returns an error when the physical schema is refused, the encoder cannot
    /// be built, or a batch cannot be shredded or written.
    async fn open(&mut self, ready: ReadyPrefix, output_file: OutputFile) -> Result<()> {
        let ReadyPrefix {
            layout,
            replay,
            remainder,
        } = ready;
        let builder = if layout.is_unshredded() {
            self.builder.clone()
        } else {
            self.builder
                .clone()
                .with_physical_schema(layout.physical_schema(&self.logical)?)?
        };
        let mut writer = builder.build(output_file).await?;
        for batch in replay.iter().chain(remainder.as_ref()) {
            writer.write(&layout.shred(batch)?).await?;
        }
        self.open = Some((writer, layout));
        Ok(())
    }
}

impl FileWriter for VariantParquetWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if let Some((writer, layout)) = self.open.as_mut() {
            return writer.write(&layout.shred(batch)?).await;
        }
        let Some((mut prefix, output_file)) = self.prefix.take() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Variant Parquet writer has neither a prefix nor an open encoder.",
            ));
        };
        if prefix.is_inert() {
            let ready = ReadyPrefix {
                layout: VariantLayout::default(),
                replay: Vec::new(),
                remainder: Some(batch.clone()),
            };
            return self.open(ready, output_file).await;
        }
        match prefix.push(batch)? {
            PrefixStep::Retained => {
                self.prefix = Some((prefix, output_file));
                Ok(())
            }
            PrefixStep::Ready(ready) => self.open(ready, output_file).await,
        }
    }

    async fn close(mut self) -> Result<Vec<DataFileBuilder>> {
        if let Some((prefix, output_file)) = self.prefix.take() {
            match prefix.finish() {
                Some(ready) => self.open(ready, output_file).await?,
                None => return Ok(vec![]),
            }
        }
        match self.open.take() {
            Some((writer, _)) => writer.close().await,
            None => Ok(vec![]),
        }
    }
}

impl CurrentFileStatus for VariantParquetWriter {
    fn current_file_path(&self) -> String {
        match (&self.open, &self.prefix) {
            (Some((writer, _)), _) => writer.current_file_path(),
            (None, Some((_, output_file))) => output_file.location().to_string(),
            (None, None) => String::new(),
        }
    }

    fn current_row_num(&self) -> usize {
        match (&self.open, &self.prefix) {
            (Some((writer, _)), _) => writer.current_row_num(),
            (None, Some((prefix, _))) => prefix.retained_rows(),
            (None, None) => 0,
        }
    }

    fn current_written_size(&self) -> usize {
        self.open
            .as_ref()
            .map_or(0, |(writer, _)| writer.current_written_size())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use parquet::variant::json_to_variant;

    use super::*;

    /// The policy values Wyrd passes, with a row bound large enough to sample every test row.
    const POLICY: VariantShreddingPolicy = VariantShreddingPolicy {
        max_rows: 4_096,
        max_bytes: 64 * 1024 * 1024,
        min_frequency_percent: 10,
        max_tracked_children: 1_000,
        max_emitted_children: 300,
        max_depth: 50,
    };

    /// A one-column batch whose Variant values are the given JSON documents (`None` is null).
    fn variant_batch(json: &[Option<&str>]) -> RecordBatch {
        let json = Arc::new(StringArray::from(json.to_vec())) as ArrayRef;
        let variant = json_to_variant(&json).unwrap();
        let field = variant.field("v");
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![variant.into()]).unwrap()
    }

    /// The shredding type inferred for column 0 from all of `json` under `policy`.
    fn infer_with(json: &[Option<&str>], policy: VariantShreddingPolicy) -> Option<DataType> {
        let batch = variant_batch(json);
        let mut prefix =
            VariantPrefix::new(&batch.schema(), policy, RecordBatch::get_array_memory_size);
        assert!(matches!(prefix.push(&batch).unwrap(), PrefixStep::Retained));
        prefix.finish().unwrap().layout.shredding_type(0).cloned()
    }

    /// The `(name, typed_value type)` pairs of an object shredding type, in emitted order.
    fn shredded_fields(shredding_type: Option<DataType>) -> Vec<(String, DataType)> {
        let Some(DataType::Struct(fields)) = shredding_type else {
            return Vec::new();
        };
        fields
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect()
    }

    /// The fields inferred from `json` under the default policy.
    fn infer(json: &[Option<&str>]) -> Vec<(String, DataType)> {
        shredded_fields(infer_with(json, POLICY))
    }

    /// A field present in at least 10% of non-null roots is shredded; one below is not; nulls do not count.
    #[test]
    fn frequency_boundary_ignores_nulls() {
        let mut rows = vec![Some(r#"{"a":1}"#)];
        rows.extend(std::iter::repeat_n(Some(r#"{"b":1}"#), 9));
        rows.extend(std::iter::repeat_n(None, 5));
        assert_eq!(infer(&rows), [
            ("a".to_string(), DataType::Int8),
            ("b".to_string(), DataType::Int8)
        ]);
        rows.push(Some(r#"{"b":2}"#));
        assert_eq!(infer(&rows), [("b".to_string(), DataType::Int8)]);
    }

    /// Integers and decimals widen within their family; mixed families, containers, and arrays stay residual.
    #[test]
    fn families_widen_or_stay_residual() {
        assert_eq!(
            infer(&[
                Some(r#"{"i":1,"f":1.5,"mixed":1,"arr":[1],"shape":1}"#),
                Some(r#"{"i":70000,"f":2.5,"mixed":"x","arr":[2],"shape":{"x":1}}"#),
            ]),
            [
                ("f".to_string(), DataType::Float64),
                ("i".to_string(), DataType::Int32)
            ]
        );

        use parquet::variant::{
            VariantArrayBuilder, VariantBuilderExt, VariantDecimal4, VariantDecimal8,
        };
        let mut builder = VariantArrayBuilder::new(2);
        builder
            .new_object()
            .with_field("d", VariantDecimal4::try_new(15, 1).unwrap())
            .finish();
        builder
            .new_object()
            .with_field("d", VariantDecimal8::try_new(1_234_567_890_123, 3).unwrap())
            .finish();
        let variant = builder.build();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![variant.field("v")])), vec![
            variant.into(),
        ])
        .unwrap();
        let mut prefix =
            VariantPrefix::new(&batch.schema(), POLICY, RecordBatch::get_array_memory_size);
        prefix.push(&batch).unwrap();
        assert_eq!(
            shredded_fields(prefix.finish().unwrap().layout.shredding_type(0).cloned()),
            [("d".to_string(), DataType::Decimal64(18, 3))]
        );
    }

    /// Children are kept by frequency with ties broken by name, emitted alphabetically, and capped.
    #[test]
    fn children_are_capped_by_frequency_then_name() {
        let policy = VariantShreddingPolicy {
            max_emitted_children: 2,
            ..POLICY
        };
        let fields = shredded_fields(infer_with(
            &[
                Some(r#"{"z":1,"c":1,"b":1}"#),
                Some(r#"{"z":1,"c":1}"#),
                Some(r#"{"z":1,"b":1}"#),
            ],
            policy,
        ));
        let names: Vec<_> = fields.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["b", "z"]);
    }

    /// Names past the tracking cap are never candidates.
    #[test]
    fn tracking_cap_ignores_later_names() {
        let policy = VariantShreddingPolicy {
            max_tracked_children: 1,
            ..POLICY
        };
        let fields = shredded_fields(infer_with(
            &[Some(r#"{"b":1}"#), Some(r#"{"a":1,"b":2}"#)],
            policy,
        ));
        let names: Vec<_> = fields.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["b"]);
    }

    /// Objects deeper than the depth bound stay residual; shallower ones nest.
    #[test]
    fn depth_bound_stops_traversal() {
        let policy = VariantShreddingPolicy {
            max_depth: 2,
            ..POLICY
        };
        let fields = shredded_fields(infer_with(
            &[Some(r#"{"a":{"b":{"c":1}},"x":{"y":1}}"#)],
            policy,
        ));
        let names: Vec<_> = fields.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            ["x"],
            "a.b is past depth 2, so `a` has nothing to shred"
        );
    }

    /// The row bound is reached mid-batch: the head is retained and the rest is the remainder.
    #[test]
    fn row_bound_splits_the_batch() {
        let policy = VariantShreddingPolicy {
            max_rows: 2,
            ..POLICY
        };
        let batch = variant_batch(&[Some(r#"{"a":1}"#), Some(r#"{"a":2}"#), Some(r#"{"b":"x"}"#)]);
        let mut prefix =
            VariantPrefix::new(&batch.schema(), policy, RecordBatch::get_array_memory_size);
        let PrefixStep::Ready(ready) = prefix.push(&batch).unwrap() else {
            panic!("the row bound is reached");
        };
        assert_eq!(
            ready
                .replay
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            2
        );
        assert_eq!(ready.remainder.map(|rest| rest.num_rows()), Some(1));
        assert_eq!(shredded_fields(ready.layout.shredding_type(0).cloned()), [
            ("a".to_string(), DataType::Int8)
        ]);
    }

    /// The byte bound stops before a batch that would cross it; that batch is not sampled.
    #[test]
    fn byte_bound_stops_before_crossing() {
        let first = variant_batch(&[Some(r#"{"a":1}"#)]);
        let policy = VariantShreddingPolicy {
            max_bytes: first.get_array_memory_size() + 1,
            ..POLICY
        };
        let mut prefix =
            VariantPrefix::new(&first.schema(), policy, RecordBatch::get_array_memory_size);
        assert!(matches!(prefix.push(&first).unwrap(), PrefixStep::Retained));
        let second = variant_batch(&[Some(r#"{"b":"x"}"#)]);
        let PrefixStep::Ready(ready) = prefix.push(&second).unwrap() else {
            panic!("the second batch would cross the byte bound");
        };
        assert_eq!(ready.replay.len(), 1);
        assert_eq!(ready.remainder.map(|rest| rest.num_rows()), Some(1));
        assert_eq!(shredded_fields(ready.layout.shredding_type(0).cloned()), [
            ("a".to_string(), DataType::Int8)
        ]);
    }

    /// A first batch alone over the byte bound is retained and sampled as the progress exception.
    #[test]
    fn oversized_first_batch_is_retained() {
        let policy = VariantShreddingPolicy {
            max_bytes: 1,
            ..POLICY
        };
        let batch = variant_batch(&[Some(r#"{"a":1}"#)]);
        let mut prefix =
            VariantPrefix::new(&batch.schema(), policy, RecordBatch::get_array_memory_size);
        let PrefixStep::Ready(ready) = prefix.push(&batch).unwrap() else {
            panic!("an oversized first batch is ready at once");
        };
        assert_eq!(ready.replay.len(), 1);
        assert!(ready.remainder.is_none());
        assert!(!ready.layout.is_unshredded());
    }

    /// Close infers from a short prefix, and an empty prefix yields nothing to write.
    #[test]
    fn short_close_infers_and_empty_close_writes_nothing() {
        let batch = variant_batch(&[Some(r#"{"a":1}"#)]);
        let empty = VariantPrefix::new(&batch.schema(), POLICY, RecordBatch::get_array_memory_size);
        assert!(empty.finish().is_none());
        assert_eq!(infer(&[Some(r#"{"a":1}"#)]), [(
            "a".to_string(),
            DataType::Int8
        )]);
    }

    /// The default policy samples nothing, so it writes unshredded files.
    #[test]
    fn default_policy_never_shreds() {
        let batch = variant_batch(&[Some(r#"{"a":1}"#)]);
        let mut prefix = VariantPrefix::new(
            &batch.schema(),
            VariantShreddingPolicy::default(),
            RecordBatch::get_array_memory_size,
        );
        let PrefixStep::Ready(ready) = prefix.push(&batch).unwrap() else {
            panic!("a zero row bound is reached at once");
        };
        assert!(ready.layout.is_unshredded());
        assert_eq!(ready.remainder.map(|rest| rest.num_rows()), Some(1));
    }

    /// Shredding keeps logical values: a later value that does not fit the type goes to the residual.
    #[test]
    fn later_incompatible_values_round_trip() {
        let sample = variant_batch(&[Some(r#"{"a":1}"#)]);
        let mut prefix =
            VariantPrefix::new(&sample.schema(), POLICY, RecordBatch::get_array_memory_size);
        prefix.push(&sample).unwrap();
        let layout = prefix.finish().unwrap().layout;
        let later = variant_batch(&[Some(r#"{"a":"text"}"#), Some(r#"{"a":[1,2]}"#), None]);
        let shredded = layout.shred(&later).unwrap();
        let logical: ArrayRef = parquet::variant::unshred_variant(
            &VariantArray::try_new(shredded.column(0).as_ref()).unwrap(),
        )
        .unwrap()
        .into();
        let json = parquet::variant::variant_to_json(&logical).unwrap();
        assert_eq!(json.iter().collect::<Vec<_>>(), [
            Some(r#"{"a":"text"}"#),
            Some(r#"{"a":[1,2]}"#),
            None
        ]);
    }
}
