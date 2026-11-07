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

//! Variant shredding layouts chosen before a file's writer opens.
//!
//! A shredding writer cannot open its Parquet encoder until it knows the
//! physical layout, so the layout is decided first and no rows are buffered:
//!
//! - [`VariantSampler`] picks a seeded, stratified sample of a whole input and
//!   infers the layout from it. Scribe uses it over a claim's staged runs.
//! - [`VariantLayout::combine`] merges the layouts other files already chose,
//!   from the shredded-leaf counts in their footers. Forge uses it over a
//!   rewrite's source files.
//!
//! Both perform no IO; callers keep their own. Arrow owns the standard
//! encoding: the layout is built with [`ShreddedSchemaBuilder`] and applied
//! with [`shred_variant`]. This module only decides which object fields to
//! shred and as which type. Every Variant field is shredded, at any Struct or
//! List depth, addressed by its name path
//! ([`variant_field_paths`](crate::arrow::variant_field_paths)); arrays
//! inside a Variant value always stay in the residual `value`.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::temporal_conversions::time64us_to_time;
use arrow_array::types::{
    Date32Type, Decimal32Type, Decimal64Type, Decimal128Type, Time64MicrosecondType,
    TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, FieldRef, Fields, Schema, SchemaRef, TimeUnit};
use chrono::{DateTime, Timelike};
use parquet::file::metadata::ParquetMetaData;
#[cfg(doc)]
use parquet::variant::unshred_variant;
use parquet::variant::{
    ShreddedSchemaBuilder, Variant, VariantArray, VariantDecimal4, VariantDecimal8,
    VariantDecimal16, VariantDecimalType, VariantMetadata, VariantObject, VariantPath,
    VariantPathElement, shred_variant,
};

use super::{FileWriter, FileWriterBuilder, ParquetWriter, ParquetWriterBuilder};
use crate::arrow::{map_variant_field, variant_field_paths};
use crate::io::OutputFile;
use crate::spec::DataFileBuilder;
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

/// Internal sampling parameters and caps that decide a Variant layout.
///
/// The values are supplied by the owning writer at construction. They are
/// policy constants, not table or user configuration. The default has a zero
/// margin: it samples nothing, so files are never shredded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VariantShreddingPolicy {
    /// Standard-normal quantile of the sample's confidence level (2.5758 for 99%).
    pub confidence_z: f64,
    /// Margin of error on a sampled field frequency, as a fraction (0.02).
    pub margin: f64,
    /// Strata with fewer rows than this are merged into one shared stratum.
    pub min_stratum_rows: usize,
    /// Share of non-null root values, as a fraction, that must hold a field.
    pub min_frequency: f64,
    /// Distinct child names tracked per object node; later names are ignored.
    pub max_tracked_children: usize,
    /// Children kept per object node, by rows covered then name.
    pub max_emitted_children: usize,
    /// Object depth below which traversal stops.
    pub max_depth: usize,
}

impl VariantShreddingPolicy {
    /// Rows to sample from a stratum of `rows` rows.
    ///
    /// Cochran's sample size for a proportion at the worst case `p = 0.5`,
    /// `n0 = z² · 0.25 / e²`, with the finite-population correction
    /// `n0 / (1 + (n0 − 1) / rows)`, rounded up and never above `rows`. A zero
    /// margin samples nothing.
    pub fn stratum_sample_size(&self, rows: usize) -> usize {
        if self.margin <= 0.0 || rows == 0 {
            return 0;
        }
        let n0 =
            (self.confidence_z * self.confidence_z * 0.25 / (self.margin * self.margin)).ceil();
        if n0 < 1.0 {
            return 0;
        }
        let corrected = n0 / (1.0 + (n0 - 1.0) / rows as f64);
        (corrected.ceil() as usize).min(rows)
    }
}

/// One physical Variant layout per Variant field of a file, at any Struct
/// or List depth.
///
/// An empty layout leaves every field unshredded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantLayout {
    /// Name path of each shredded Variant field and the shredding type passed
    /// to [`shred_variant`].
    fields: Vec<(Vec<String>, DataType)>,
}

impl VariantLayout {
    /// Whether no field is shredded.
    pub fn is_unshredded(&self) -> bool {
        self.fields.is_empty()
    }

    /// The shredding type chosen for the Variant field at name `path`, if it is shredded.
    pub fn shredding_type(&self, path: &[&str]) -> Option<&DataType> {
        self.fields.iter().find_map(|(field, data_type)| {
            field
                .iter()
                .map(String::as_str)
                .eq(path.iter().copied())
                .then_some(data_type)
        })
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

    /// Shreds every Variant field this layout names and returns the physical batch.
    ///
    /// Each shredded field keeps its name, nullability, and metadata (field id
    /// and Variant extension); only its storage type becomes the standard
    /// `metadata`/`value`/`typed_value` struct, and the Structs and Lists
    /// around it are rebuilt with that type. Values that do not fit the chosen
    /// type are kept in the residual `value`, so the logical values are
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when a named field is missing, is not valid Variant
    /// storage, or Arrow refuses to shred it.
    pub fn shred(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if self.is_unshredded() {
            return Ok(batch.clone());
        }
        let schema = batch.schema();
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (path, shredding_type) in &self.fields {
            let index = schema.index_of(&path[0])?;
            (fields[index], columns[index]) =
                map_variant_field(&fields[index], &columns[index], path, &mut |variant| {
                    let variant = VariantArray::try_new(variant.as_ref()).map_err(variant_error)?;
                    Ok(shred_variant(&variant, shredding_type)
                        .map_err(variant_error)?
                        .into())
                })?;
        }
        let physical = Schema::new(fields).with_metadata(schema.metadata().clone());
        RecordBatch::try_new(Arc::new(physical), columns).map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to assemble shredded batch.").with_source(err)
        })
    }

    /// Combines the layouts `footers` already chose into one layout for files of `logical`.
    ///
    /// For each Variant field, at any Struct or List depth, every source's
    /// shredded scalar leaf counts its values minus its null count: the exact
    /// values holding that field with that type (inside a List, each element
    /// counts). A leaf whose footer has no null count counts every value.
    /// Counts are summed over all sources; types of one family widen together,
    /// and across families the type covering more rows wins (the others go to
    /// the residual). A field is kept when its combined count reaches
    /// `min_frequency` of the combined non-null roots, then children are
    /// capped per object node by rows covered, ties by name. This is a count,
    /// not a sample, so the margin does not apply; a field no source shredded
    /// is never shredded.
    ///
    /// # Errors
    ///
    /// Returns an error when a footer's schema cannot be read as Arrow.
    pub fn combine(
        logical: &Schema,
        footers: &[Arc<ParquetMetaData>],
        policy: &VariantShreddingPolicy,
    ) -> Result<Self> {
        let mut fields = Vec::new();
        for field in variant_field_paths(logical.fields()) {
            let mut roots = 0_usize;
            let mut leaves: BTreeMap<Vec<String>, Vec<(Scalar, usize)>> = BTreeMap::new();
            for footer in footers {
                let counts = FooterCounts::new(footer, &field)?;
                roots += counts.metadata_values();
                let Some((first_leaf, typed)) = counts.typed_value_fields() else {
                    continue;
                };
                let mut path = Vec::new();
                counts.collect_leaves(&typed, first_leaf, &mut path, &mut leaves);
            }
            let mut root = ObjectNode::default();
            for (path, typed) in leaves {
                let Some((scalar, count)) = Scalar::winner(typed) else {
                    continue;
                };
                root.insert_counted(&path, scalar, count);
            }
            let rules = [StratumRule {
                min_count: ((roots as f64) * policy.min_frequency).ceil().max(1.0) as usize,
                weight: 1.0,
            }];
            if let Some(shredding_type) = root.shredding_type(&rules, policy) {
                fields.push((field, shredding_type));
            }
        }
        Ok(VariantLayout { fields })
    }
}

/// The Parquet leaf indices of one Variant field in a file footer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantLeaves {
    /// Every leaf the field is stored in.
    pub all: Range<usize>,
    /// The `metadata` leaf.
    pub metadata: Option<usize>,
    /// The root residual `value` leaf, when the file stores one.
    pub value: Option<usize>,
}

/// Locates every Variant field of `logical`, at any Struct or List depth, in
/// `footer`, through the same path walk the footer combine uses. A field the
/// file lacks is skipped.
///
/// # Errors
///
/// Returns an error when the footer's schema cannot be read as Arrow.
pub fn variant_leaves(logical: &Schema, footer: &ParquetMetaData) -> Result<Vec<VariantLeaves>> {
    let mut found = Vec::new();
    for path in variant_field_paths(logical.fields()) {
        let counts = FooterCounts::new(footer, &path)?;
        let Some((DataType::Struct(storage), first_leaf)) = &counts.storage else {
            continue;
        };
        let leaves = storage
            .iter()
            .map(|child| leaf_count(child.data_type()))
            .sum::<usize>();
        let leaf = |name| child_leaf(storage, *first_leaf, name).map(|(leaf, _)| leaf);
        found.push(VariantLeaves {
            all: *first_leaf..first_leaf + leaves,
            metadata: leaf("metadata"),
            value: leaf("value"),
        });
    }
    Ok(found)
}

/// Number of Parquet leaf columns a field of `data_type` is stored in.
fn leaf_count(data_type: &DataType) -> usize {
    match data_type {
        DataType::Struct(children) => children
            .iter()
            .map(|child| leaf_count(child.data_type()))
            .sum(),
        DataType::List(element)
        | DataType::LargeList(element)
        | DataType::FixedSizeList(element, _)
        | DataType::Map(element, _) => leaf_count(element.data_type()),
        _ => 1,
    }
}

/// The child named `name` of `fields`, whose leaves start at `first_leaf`,
/// with the index of the child's own first leaf.
fn child_leaf<'f>(
    fields: &'f Fields,
    first_leaf: usize,
    name: &str,
) -> Option<(usize, &'f FieldRef)> {
    let mut leaf = first_leaf;
    for field in fields {
        if field.name() == name {
            return Some((leaf, field));
        }
        leaf += leaf_count(field.data_type());
    }
    None
}

/// Shredded-leaf counts of one Variant field in one source footer.
///
/// Parquet leaves are numbered in schema order, so a leaf's index is found by
/// counting the leaves of the fields before it.
struct FooterCounts<'a> {
    /// The source file's metadata.
    footer: &'a ParquetMetaData,
    /// The field's Arrow storage type in this file and its first leaf index,
    /// or `None` when the file lacks the field.
    storage: Option<(DataType, usize)>,
}

impl<'a> FooterCounts<'a> {
    /// Finds the Variant field at name `path` in `footer`.
    ///
    /// # Errors
    ///
    /// Returns an error when the footer's schema cannot be read as Arrow.
    fn new(footer: &'a ParquetMetaData, path: &[String]) -> Result<Self> {
        let descr = footer.file_metadata().schema_descr();
        let arrow = parquet::arrow::parquet_to_arrow_schema(descr, None).map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                "Invalid Parquet schema in a source footer.",
            )
            .with_source(err)
        })?;
        Ok(Self {
            footer,
            storage: Self::locate(arrow.fields(), 0, path),
        })
    }

    /// The type and first leaf index of the field at `path` among `fields`,
    /// whose leaves start at `first_leaf`.
    fn locate(fields: &Fields, first_leaf: usize, path: &[String]) -> Option<(DataType, usize)> {
        let (name, rest) = path.split_first()?;
        let (leaf, field) = child_leaf(fields, first_leaf, name)?;
        Self::descend(field.data_type(), leaf, rest)
    }

    /// The type and first leaf index at the rest of a path below a field of
    /// `data_type` whose leaves start at `first_leaf`; a List step enters the
    /// element, a Struct step names a child.
    fn descend(
        data_type: &DataType,
        first_leaf: usize,
        rest: &[String],
    ) -> Option<(DataType, usize)> {
        let Some((_, tail)) = rest.split_first() else {
            return Some((data_type.clone(), first_leaf));
        };
        match data_type {
            DataType::List(element) => Self::descend(element.data_type(), first_leaf, tail),
            DataType::Struct(children) => Self::locate(children, first_leaf, rest),
            _ => None,
        }
    }

    /// Non-null values of leaf `leaf`, summed over row groups.
    ///
    /// A row group without a null count counts every value.
    fn present(&self, leaf: usize) -> usize {
        self.footer
            .row_groups()
            .iter()
            .map(|group| {
                let column = group.column(leaf);
                let values = usize::try_from(column.num_values()).unwrap_or(0);
                let nulls = column
                    .statistics()
                    .and_then(|stats| stats.null_count_opt())
                    .map_or(0, |nulls| usize::try_from(nulls).unwrap_or(values));
                values.saturating_sub(nulls)
            })
            .sum()
    }

    /// Non-null Variant values of the field: its non-null `metadata` leaf.
    fn metadata_values(&self) -> usize {
        let Some((DataType::Struct(storage), first_leaf)) = &self.storage else {
            return 0;
        };
        child_leaf(storage, *first_leaf, "metadata").map_or(0, |(leaf, _)| self.present(leaf))
    }

    /// The fields of the field's top-level `typed_value` object, if it is
    /// shredded as one, with the object's first leaf index.
    fn typed_value_fields(&self) -> Option<(usize, Fields)> {
        let (DataType::Struct(storage), first_leaf) = self.storage.as_ref()? else {
            return None;
        };
        let (leaf, typed) = child_leaf(storage, *first_leaf, "typed_value")?;
        match typed.data_type() {
            DataType::Struct(fields) => Some((leaf, fields.clone())),
            _ => None,
        }
    }

    /// Adds every shredded scalar leaf under `fields`, whose leaves start at
    /// `first_leaf`, at `path` to `leaves`.
    ///
    /// A shredded field is a struct of `value` and `typed_value`; its
    /// `typed_value` is either a scalar leaf or a nested object. Anything else,
    /// such as a shredded array, is skipped.
    fn collect_leaves(
        &self,
        fields: &Fields,
        first_leaf: usize,
        path: &mut Vec<String>,
        leaves: &mut BTreeMap<Vec<String>, Vec<(Scalar, usize)>>,
    ) {
        let mut next_leaf = first_leaf;
        for field in fields {
            let element_leaf = next_leaf;
            next_leaf += leaf_count(field.data_type());
            let DataType::Struct(element) = field.data_type() else {
                continue;
            };
            let Some((typed_leaf, typed)) = child_leaf(element, element_leaf, "typed_value") else {
                continue;
            };
            path.push(field.name().clone());
            match typed.data_type() {
                DataType::Struct(nested) => self.collect_leaves(nested, typed_leaf, path, leaves),
                data_type => {
                    if let Some(scalar) = Scalar::from_data_type(data_type) {
                        leaves
                            .entry(path.clone())
                            .or_default()
                            .push((scalar, self.present(typed_leaf)));
                    }
                }
            }
            path.pop();
        }
    }
}

/// Picks a seeded, stratified sample of one input and infers its Variant layout.
///
/// The caller counts the rows of every stratum first, then offers every row
/// once with its stratum and its position in the input. Strata with fewer
/// than the policy's minimum rows share one stratum. Each stratum keeps a row
/// when a hash of the seed and the row's position falls below its Cochran
/// sample size divided by its rows, so the same seed and positions always
/// select the same rows. Only per-stratum counters are kept, never rows.
pub struct VariantSampler {
    /// Seed of the row-selection hash.
    seed: u64,
    /// Sampling stratum of each caller stratum.
    stratum_of: Vec<usize>,
    /// Probability of keeping a row, per sampling stratum.
    rates: Vec<f64>,
    /// Rows per sampling stratum.
    rows: Vec<usize>,
    /// Rows kept per sampling stratum.
    kept: Vec<usize>,
    /// Counters for every Variant field.
    analyzer: VariantLayoutAnalyzer,
}

impl VariantSampler {
    /// Plans the sample of an input of `logical` rows whose caller strata hold `stratum_rows` rows.
    pub fn new(
        logical: &Schema,
        policy: VariantShreddingPolicy,
        seed: u64,
        stratum_rows: &[usize],
    ) -> Self {
        let mut stratum_of = Vec::with_capacity(stratum_rows.len());
        let mut rows = Vec::new();
        let mut shared = None;
        for &stratum in stratum_rows {
            let index = if stratum >= policy.min_stratum_rows {
                rows.push(0);
                rows.len() - 1
            } else {
                *shared.get_or_insert_with(|| {
                    rows.push(0);
                    rows.len() - 1
                })
            };
            rows[index] += stratum;
            stratum_of.push(index);
        }
        let rates = rows
            .iter()
            .map(|&count| {
                if count == 0 {
                    0.0
                } else {
                    policy.stratum_sample_size(count) as f64 / count as f64
                }
            })
            .collect();
        let kept = vec![0; rows.len()];
        Self {
            seed,
            stratum_of,
            rates,
            rows,
            kept,
            analyzer: VariantLayoutAnalyzer::new(logical, policy),
        }
    }

    /// Whether the input has no Variant field, so nothing needs sampling.
    pub fn is_inert(&self) -> bool {
        self.analyzer.columns.is_empty()
    }

    /// Names of the top-level columns holding the Variant fields [`Self::offer`]
    /// reads, each once; a caller projects only these.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        let mut seen = BTreeSet::new();
        self.analyzer
            .columns
            .iter()
            .map(|sample| sample.path[0].as_str())
            .filter(move |name| seen.insert(*name))
    }

    /// Offers every row of `batch`; row `i` is in caller stratum `strata[i]` at position `first_position + i`.
    ///
    /// `batch` needs only the columns [`Self::column_names`] names, found by
    /// name; other columns are ignored. A Variant field may be shredded in any
    /// layout: only the values the sample keeps are reconstructed, so they are
    /// counted exactly as their unshredded values would be.
    ///
    /// # Errors
    ///
    /// Returns an error when `strata` does not have one entry per row, names an
    /// unknown stratum, a Variant field is missing from `batch`, or a Variant
    /// field cannot be read as Variant storage.
    pub fn offer(
        &mut self,
        batch: &RecordBatch,
        strata: &[usize],
        first_position: u64,
    ) -> Result<()> {
        if strata.len() != batch.num_rows() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Variant sample needs one stratum per row.",
            ));
        }
        let mut chosen = Vec::with_capacity(strata.len());
        for (row, &stratum) in strata.iter().enumerate() {
            let Some(&index) = self.stratum_of.get(stratum) else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Variant sample has no stratum {stratum}."),
                ));
            };
            if unit_hash(self.seed, first_position + row as u64) < self.rates[index] {
                self.kept[index] += 1;
                chosen.push((row, index));
            }
        }
        self.analyzer.observe(batch, &chosen)
    }

    /// Rows the sample has kept so far, over every stratum.
    pub fn sampled_rows(&self) -> usize {
        self.kept.iter().sum()
    }

    /// The layout the sample supports.
    ///
    /// A field is eligible when its sampled frequency among a stratum's
    /// non-null roots is at least `min_frequency − margin` in any stratum.
    /// Eligible fields are ranked by estimated rows covered, the sum over
    /// strata of rows × sampled frequency, with ties broken by name.
    pub fn layout(&self) -> VariantLayout {
        let weights: Vec<f64> = self
            .rows
            .iter()
            .zip(&self.kept)
            .map(|(&rows, &kept)| {
                if kept == 0 {
                    0.0
                } else {
                    rows as f64 / kept as f64
                }
            })
            .collect();
        self.analyzer.layout(&weights)
    }
}

/// A uniform value in `[0, 1)` from `seed` and `position`, stable across platforms and releases.
///
/// One SplitMix64 step: the seed advanced by `position` golden-ratio
/// increments, then the standard finalizer.
fn unit_hash(seed: u64, position: u64) -> f64 {
    let mut z = seed.wrapping_add(position.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// What one stratum requires of a field and how much each of its sampled rows stands for.
#[derive(Clone, Copy)]
struct StratumRule {
    /// Sampled roots holding a field for the field to be eligible in this stratum.
    min_count: usize,
    /// Input rows each sampled row of this stratum represents.
    weight: f64,
}

/// Sample counters for every Variant field of one input.
struct VariantLayoutAnalyzer {
    /// Parameters and caps.
    policy: VariantShreddingPolicy,
    /// Counters per Variant field.
    columns: Vec<ColumnSample>,
}

/// Counters for one Variant field, at any Struct or List depth.
struct ColumnSample {
    /// Name path of the field; its first name is the top-level column found
    /// in offered batches.
    path: Vec<String>,
    /// Sampled non-null roots per sampling stratum; inside a List, each
    /// non-null element value is one root.
    roots: BTreeMap<usize, usize>,
    /// Fields of root values that are objects.
    node: ObjectNode,
}

impl VariantLayoutAnalyzer {
    /// Finds every Variant field of `schema`, at any Struct or List depth.
    fn new(schema: &Schema, policy: VariantShreddingPolicy) -> Self {
        let columns = variant_field_paths(schema.fields())
            .into_iter()
            .map(|path| ColumnSample {
                path,
                roots: BTreeMap::new(),
                node: ObjectNode::default(),
            })
            .collect();
        Self { policy, columns }
    }

    /// Adds the `chosen` rows of `batch`, each with its sampling stratum, to the counters.
    ///
    /// Each chosen value is read in place through a [`Reader`], so a shredded
    /// value is counted exactly as its reconstructed value would be without
    /// copying or encoding its payload.
    ///
    /// # Errors
    ///
    /// Returns an error when a Variant field is missing from `batch`, is not
    /// Variant storage, or a chosen value is not valid Variant data.
    fn observe(&mut self, batch: &RecordBatch, chosen: &[(usize, usize)]) -> Result<()> {
        for sample in &mut self.columns {
            let (array, values) = sample.values(batch, chosen)?;
            let variant = VariantArray::try_new(array.as_ref()).map_err(variant_error)?;
            let reader = Reader::try_new(variant.inner())?;
            for (index, stratum) in values {
                if variant.is_null(index) {
                    continue;
                }
                let metadata = binary(variant.metadata_column().as_ref(), index)?;
                let metadata = VariantMetadata::try_new(metadata).map_err(variant_error)?;
                *sample.roots.entry(stratum).or_default() += 1;
                match reader.read(metadata, index)? {
                    Logical::Object(object) => {
                        sample
                            .node
                            .observe_object(&object, 1, stratum, &self.policy)?
                    }
                    value => value.check()?,
                }
            }
        }
        Ok(())
    }

    /// Chooses each field's shredding type; `weights[h]` is the input rows per sampled row of stratum `h`.
    fn layout(&self, weights: &[f64]) -> VariantLayout {
        let fraction = (self.policy.min_frequency - self.policy.margin).max(0.0);
        let fields = self
            .columns
            .iter()
            .filter_map(|sample| {
                let rules: Vec<StratumRule> = weights
                    .iter()
                    .enumerate()
                    .map(|(stratum, &weight)| {
                        let roots = sample.roots.get(&stratum).copied().unwrap_or(0);
                        StratumRule {
                            min_count: ((roots as f64) * fraction).ceil().max(1.0) as usize,
                            weight,
                        }
                    })
                    .collect();
                sample
                    .node
                    .shredding_type(&rules, &self.policy)
                    .map(|shredding_type| (sample.path.clone(), shredding_type))
            })
            .collect();
        VariantLayout { fields }
    }
}

impl ColumnSample {
    /// The field's Variant storage in `batch` with the index and stratum of
    /// each of its values in the `chosen` rows.
    ///
    /// The walk follows the path from the top-level column: a Struct step
    /// keeps the values whose Struct is valid, and a List step expands each
    /// valid List into its elements, so a row inside a List holds one value
    /// per element. No value is copied.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is missing from `batch` or crosses a
    /// type other than Struct or List.
    fn values(
        &self,
        batch: &RecordBatch,
        chosen: &[(usize, usize)],
    ) -> Result<(ArrayRef, Vec<(usize, usize)>)> {
        let missing = || {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Variant sample batch has no field {}.", self.path.join(".")),
            )
        };
        let mut array = Arc::clone(batch.column_by_name(&self.path[0]).ok_or_else(missing)?);
        let mut values: Vec<(usize, usize)> = chosen.to_vec();
        for step in &self.path[1..] {
            array = match array.data_type() {
                DataType::List(_) => {
                    let list = array.as_list::<i32>();
                    values = values
                        .into_iter()
                        .filter(|&(index, _)| list.is_valid(index))
                        .flat_map(|(index, stratum)| {
                            {
                                list.value_offsets()[index] as usize
                                    ..list.value_offsets()[index + 1] as usize
                            }
                            .map(move |element| (element, stratum))
                        })
                        .collect();
                    Arc::clone(list.values())
                }
                DataType::Struct(_) => {
                    let parent = array.as_struct();
                    values.retain(|&(index, _)| parent.is_valid(index));
                    Arc::clone(parent.column_by_name(step).ok_or_else(missing)?)
                }
                _ => return Err(missing()),
            };
        }
        Ok((array, values))
    }
}

/// Reads logical Variant values in place from one `value`/`typed_value`
/// pair, at the top level or of a shredded object field or list element.
///
/// It follows [`unshred_variant`]'s reconstruction exactly, built once per
/// array from the physical type: it accepts the same shredding types, rejects
/// the same per-value contents, and gives each value the type its
/// reconstruction would have. A value is reported only as sampling counts it,
/// so no payload is copied or encoded: missing, null, a scalar family, a list,
/// or an object's fields in name order.
enum Reader<'a> {
    /// Neither column, so every value is missing.
    Absent,
    /// Only the residual `value` column.
    Value(&'a dyn Array),
    /// A `typed_value` column, and the residual `value` column when present.
    Typed {
        /// Residual Variant bytes.
        value: Option<&'a dyn Array>,
        /// Shredded values.
        typed: Typed<'a>,
    },
}

/// The `typed_value` column of a [`Reader`].
enum Typed<'a> {
    /// A shredded scalar.
    Leaf(&'a dyn Array, Leaf),
    /// A shredded object; its fields sorted by name, as a reconstructed
    /// object orders them.
    Object(&'a StructArray, Vec<(&'a str, Reader<'a>)>),
    /// A shredded array whose elements `element` reads; `elements` gives the
    /// element range of a value.
    List(&'a dyn Array, Box<Reader<'a>>),
}

/// How a shredded scalar column reconstructs.
#[derive(Clone, Copy)]
enum Leaf {
    /// Every value has this family.
    Fixed(Scalar),
    /// Days since the epoch, which must be a valid date.
    Date,
    /// Microseconds since midnight, which must be a valid time.
    Time,
    /// Instants in microseconds or `nanos`, adjusted to UTC when `utc`.
    Timestamp { nanos: bool, utc: bool },
    /// Decimals of this many bytes and signed scale.
    Decimal { bytes: u8, scale: i8 },
}

/// One logical Variant value as the sampler counts it.
enum Logical<'r, 'a> {
    /// No value: an object field that is absent.
    Missing,
    /// A Variant null.
    Null,
    /// A scalar of this family.
    Scalar(Scalar),
    /// An array; its elements are never counted.
    List,
    /// An object.
    Object(LogicalObject<'r, 'a>),
}

/// The fields of one logical object: shredded fields merged with the fields
/// of its residual object.
struct LogicalObject<'r, 'a> {
    /// Metadata of the value's row.
    metadata: VariantMetadata<'a>,
    /// Shredded fields sorted by name; empty for a residual-only object.
    typed: &'r [(&'a str, Reader<'a>)],
    /// Index of the value in the shredded field columns.
    index: usize,
    /// Fully validated residual fields, sorted by name.
    residual: Option<VariantObject<'a, 'a>>,
}

impl<'a> Reader<'a> {
    /// A reader of the `value`/`typed_value` pair `pair`.
    ///
    /// # Errors
    ///
    /// Returns an error for a shredding type [`unshred_variant`] rejects or a
    /// residual `value` column that is not binary.
    fn try_new(pair: &'a StructArray) -> Result<Self> {
        let value = match pair.column_by_name("value") {
            Some(value) => {
                binary_type(value.data_type())?;
                Some(value.as_ref())
            }
            None => None,
        };
        let Some(typed) = pair.column_by_name("typed_value") else {
            return Ok(value.map_or(Reader::Absent, Reader::Value));
        };
        let invalid = |what: String| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("{what} is not a valid Variant shredding type."),
            )
        };
        let fixed = |scalar| Typed::Leaf(typed.as_ref(), Leaf::Fixed(scalar));
        let leaf = |leaf| Typed::Leaf(typed.as_ref(), leaf);
        let decimal = |bytes, max: u8, precision: u8, scale: i8| {
            if precision == 0 || precision > max || scale.unsigned_abs() > precision {
                return Err(invalid(typed.data_type().to_string()));
            }
            Ok(leaf(Leaf::Decimal { bytes, scale }))
        };
        let typed = match typed.data_type() {
            DataType::Boolean => fixed(Scalar::Boolean),
            DataType::Int8 => fixed(Scalar::Int(1)),
            DataType::Int16 => fixed(Scalar::Int(2)),
            DataType::Int32 => fixed(Scalar::Int(4)),
            DataType::Int64 => fixed(Scalar::Int(8)),
            DataType::Float32 => fixed(Scalar::Float),
            DataType::Float64 => fixed(Scalar::Double),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => fixed(Scalar::String),
            DataType::Binary | DataType::BinaryView | DataType::LargeBinary => {
                fixed(Scalar::Binary)
            }
            DataType::FixedSizeBinary(16) => fixed(Scalar::Uuid),
            DataType::Date32 => leaf(Leaf::Date),
            DataType::Time64(TimeUnit::Microsecond) => leaf(Leaf::Time),
            DataType::Timestamp(unit @ (TimeUnit::Microsecond | TimeUnit::Nanosecond), zone) => {
                leaf(Leaf::Timestamp {
                    nanos: *unit == TimeUnit::Nanosecond,
                    utc: zone.is_some(),
                })
            }
            DataType::Decimal32(precision, scale) => decimal(4, 9, *precision, *scale)?,
            DataType::Decimal64(precision, scale) => decimal(8, 18, *precision, *scale)?,
            DataType::Decimal128(precision, scale) => decimal(16, 38, *precision, *scale)?,
            DataType::Struct(_) => {
                let object = typed.as_struct();
                let mut fields = object
                    .fields()
                    .iter()
                    .zip(object.columns())
                    .map(|(field, column)| {
                        let Some(column) = column.as_struct_opt() else {
                            return Err(invalid(format!("Object field {}", field.name())));
                        };
                        Ok((field.name().as_str(), Reader::try_new(column)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                fields.sort_by_key(|(name, _)| *name);
                Typed::Object(object, fields)
            }
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::FixedSizeList(..) => {
                let Some(element) = list_values(typed.as_ref()).as_struct_opt() else {
                    return Err(invalid("List element".to_owned()));
                };
                Typed::List(typed.as_ref(), Box::new(Reader::try_new(element)?))
            }
            other => return Err(invalid(other.to_string())),
        };
        Ok(Reader::Typed { value, typed })
    }

    /// The value at `index`, whose row has `metadata`.
    ///
    /// The typed value wins over the residual one; with neither the value is
    /// missing. A scalar or list is fully checked here, an object's fields as
    /// they are read.
    ///
    /// # Errors
    ///
    /// Returns an error when residual bytes are not valid Variant, a typed
    /// scalar is out of range, a scalar or list has both a typed and a
    /// residual value, or a partially shredded object's residual value is not
    /// an object.
    fn read<'r>(&'r self, metadata: VariantMetadata<'a>, index: usize) -> Result<Logical<'r, 'a>> {
        let residual = |value: Option<&'a dyn Array>| -> Result<Option<Variant<'a, 'a>>> {
            match value {
                Some(value) if value.is_valid(index) => Ok(Some(
                    Variant::try_new_with_metadata(metadata.clone(), binary(value, index)?)
                        .map_err(variant_error)?,
                )),
                _ => Ok(None),
            }
        };
        let (value, typed) = match self {
            Reader::Absent => return Ok(Logical::Missing),
            Reader::Value(value) => {
                return Ok(residual(Some(*value))?.map_or(Logical::Missing, Logical::of));
            }
            Reader::Typed { value, typed } => (residual(*value)?, typed),
        };
        let (Typed::Leaf(array, _) | Typed::List(array, _)) = typed else {
            let Typed::Object(array, fields) = typed else {
                unreachable!("every typed column is a leaf, list, or object");
            };
            if array.is_null(index) {
                return Ok(value.map_or(Logical::Missing, Logical::of));
            }
            let residual = match value {
                None => None,
                Some(Variant::Object(object)) => Some(object),
                Some(_) => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "A partially shredded Variant object has a residual value that is not an object.",
                    ));
                }
            };
            return Ok(Logical::Object(LogicalObject {
                metadata,
                typed: fields,
                index,
                residual,
            }));
        };
        if array.is_null(index) {
            return Ok(value.map_or(Logical::Missing, Logical::of));
        }
        if value.is_some() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "A shredded Variant value has both a typed and a residual value.",
            ));
        }
        match typed {
            Typed::Leaf(array, leaf) => Ok(Logical::Scalar(leaf.scalar(*array, index)?)),
            Typed::List(array, element) => {
                for element_index in list_range(*array, index) {
                    match element.read(metadata.clone(), element_index)? {
                        Logical::Missing => {}
                        value => value.check()?,
                    }
                }
                Ok(Logical::List)
            }
            Typed::Object(..) => unreachable!("objects return above"),
        }
    }
}

impl Leaf {
    /// The family of the value at `index` of `array`, which is not null.
    ///
    /// # Errors
    ///
    /// Returns an error when the value cannot be represented in Variant.
    fn scalar(self, array: &dyn Array, index: usize) -> Result<Scalar> {
        let out_of_range = |what: &str| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid shredded Variant {what} value."),
            )
        };
        Ok(match self {
            Leaf::Fixed(scalar) => scalar,
            Leaf::Date => {
                let days = array.as_primitive::<Date32Type>().value(index);
                Date32Type::to_naive_date_opt(days).ok_or_else(|| out_of_range("date"))?;
                Scalar::Date
            }
            Leaf::Time => {
                let micros = array.as_primitive::<Time64MicrosecondType>().value(index);
                time64us_to_time(micros).ok_or_else(|| out_of_range("time"))?;
                Scalar::Time
            }
            Leaf::Timestamp { nanos: false, utc } => {
                let micros = array
                    .as_primitive::<TimestampMicrosecondType>()
                    .value(index);
                DateTime::from_timestamp_micros(micros).ok_or_else(|| out_of_range("timestamp"))?;
                Scalar::Timestamp { nanos: false, utc }
            }
            Leaf::Timestamp { nanos: true, utc } => {
                let nanos = array.as_primitive::<TimestampNanosecondType>().value(index);
                Scalar::Timestamp {
                    nanos: !DateTime::from_timestamp_nanos(nanos)
                        .nanosecond()
                        .is_multiple_of(1000),
                    utc,
                }
            }
            Leaf::Decimal { bytes, scale } => {
                let scale = match bytes {
                    4 => VariantDecimal4::try_new_with_signed_scale(
                        array.as_primitive::<Decimal32Type>().value(index),
                        scale,
                    )
                    .map(|decimal| decimal.scale()),
                    8 => VariantDecimal8::try_new_with_signed_scale(
                        array.as_primitive::<Decimal64Type>().value(index),
                        scale,
                    )
                    .map(|decimal| decimal.scale()),
                    _ => VariantDecimal16::try_new_with_signed_scale(
                        array.as_primitive::<Decimal128Type>().value(index),
                        scale,
                    )
                    .map(|decimal| decimal.scale()),
                }
                .map_err(variant_error)?;
                Scalar::decimal(bytes_precision(bytes), scale)
            }
        })
    }
}

/// The largest precision of a Variant decimal of `bytes` bytes.
const fn bytes_precision(bytes: u8) -> u8 {
    match bytes {
        4 => 9,
        8 => 18,
        _ => 38,
    }
}

impl<'r, 'a> Logical<'r, 'a> {
    /// The logical form of a validated residual `value`.
    fn of(value: Variant<'a, 'a>) -> Self {
        match value {
            Variant::Null => Logical::Null,
            Variant::List(_) => Logical::List,
            Variant::Object(object) => Logical::Object(LogicalObject {
                metadata: object.metadata.clone(),
                typed: &[],
                index: 0,
                residual: Some(object),
            }),
            scalar => Scalar::of(&scalar).map_or(Logical::Null, Logical::Scalar),
        }
    }

    /// Checks the parts of this value not yet read, which the sampler does
    /// not count: the shredded fields of an object, at any depth.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Reader::read`] and [`LogicalObject::fields`].
    fn check(self) -> Result<()> {
        match self {
            Logical::Object(object) if !object.typed.is_empty() => {
                object.fields(|_, value| value.check())
            }
            _ => Ok(()),
        }
    }
}

impl<'r, 'a> LogicalObject<'r, 'a> {
    /// Calls `visit` with each present field and its value, in name order.
    ///
    /// # Errors
    ///
    /// Returns the first error of `visit` or [`Reader::read`], or an error
    /// when a residual field repeats a shredded field's name.
    fn fields(&self, mut visit: impl FnMut(&'a str, Logical<'r, 'a>) -> Result<()>) -> Result<()> {
        let mut residual = self
            .residual
            .iter()
            .flat_map(|object| object.iter())
            .peekable();
        for (name, reader) in self.typed {
            while let Some((field, value)) = residual.next_if(|(field, _)| field < name) {
                visit(field, Logical::of(value))?;
            }
            if residual.next_if(|(field, _)| field == name).is_some() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Variant field '{name}' is both shredded and residual."),
                ));
            }
            match reader.read(self.metadata.clone(), self.index)? {
                Logical::Missing => {}
                value => visit(name, value)?,
            }
        }
        residual.try_for_each(|(field, value)| visit(field, Logical::of(value)))
    }
}

/// Fails unless `data_type` holds Variant bytes.
///
/// # Errors
///
/// Returns an error for any type but Binary, LargeBinary, or BinaryView.
fn binary_type(data_type: &DataType) -> Result<()> {
    match data_type {
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Ok(()),
        // ponytail: dictionary or run-end encoded Variant bytes, which
        // `unshred_variant` decodes, are rejected; Wyrd never writes them.
        other => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Variant bytes must be binary, not {other}."),
        )),
    }
}

/// The Variant bytes at `index` of `array`, which is not null.
///
/// # Errors
///
/// Returns the error of [`binary_type`].
fn binary(array: &dyn Array, index: usize) -> Result<&[u8]> {
    binary_type(array.data_type())?;
    Ok(match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().value(index),
        DataType::LargeBinary => array.as_binary::<i64>().value(index),
        _ => array.as_binary_view().value(index),
    })
}

/// The element values of a list-like `array`.
fn list_values(array: &dyn Array) -> &ArrayRef {
    match array.data_type() {
        DataType::List(_) => array.as_list::<i32>().values(),
        DataType::LargeList(_) => array.as_list::<i64>().values(),
        DataType::ListView(_) => array.as_list_view::<i32>().values(),
        DataType::LargeListView(_) => array.as_list_view::<i64>().values(),
        _ => array.as_fixed_size_list().values(),
    }
}

/// The element range of the list at `index` of a list-like `array`.
fn list_range(array: &dyn Array, index: usize) -> Range<usize> {
    let span = |offset: usize, len: usize| offset..offset + len;
    match array.data_type() {
        DataType::List(_) => {
            let offsets = array.as_list::<i32>().value_offsets();
            offsets[index] as usize..offsets[index + 1] as usize
        }
        DataType::LargeList(_) => {
            let offsets = array.as_list::<i64>().value_offsets();
            offsets[index] as usize..offsets[index + 1] as usize
        }
        DataType::ListView(_) => {
            let list = array.as_list_view::<i32>();
            span(
                list.value_offsets()[index] as usize,
                list.value_sizes()[index] as usize,
            )
        }
        DataType::LargeListView(_) => {
            let list = array.as_list_view::<i64>();
            span(
                list.value_offsets()[index] as usize,
                list.value_sizes()[index] as usize,
            )
        }
        _ => {
            let list = array.as_fixed_size_list();
            let len = list.value_length() as usize;
            span(index * len, len)
        }
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
    /// Roots holding this child, per stratum.
    counts: BTreeMap<usize, usize>,
    /// Merged family of every non-null value seen.
    family: Family,
}

impl Child {
    /// A child not yet seen with a value.
    fn new() -> Self {
        Child {
            counts: BTreeMap::new(),
            family: Family::Unknown,
        }
    }

    /// Whether any stratum holds this child often enough under `rules`.
    fn eligible(&self, rules: &[StratumRule]) -> bool {
        self.counts.iter().any(|(&stratum, &count)| {
            rules
                .get(stratum)
                .is_some_and(|rule| count >= rule.min_count)
        })
    }

    /// Estimated input rows holding this child under `rules`.
    fn covered(&self, rules: &[StratumRule]) -> f64 {
        self.counts
            .iter()
            .map(|(&stratum, &count)| {
                rules
                    .get(stratum)
                    .map_or(0.0, |rule| count as f64 * rule.weight)
            })
            .sum()
    }
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
    /// Records the children of one object value at `depth`, sampled in `stratum`.
    ///
    /// # Errors
    ///
    /// Returns the errors of reading the object's fields.
    fn observe_object(
        &mut self,
        object: &LogicalObject<'_, '_>,
        depth: usize,
        stratum: usize,
        policy: &VariantShreddingPolicy,
    ) -> Result<()> {
        object.fields(|name, value| {
            let tracked = self.children.len();
            let child = match self.children.get_mut(name) {
                Some(child) => child,
                None if tracked < policy.max_tracked_children => self
                    .children
                    .entry(name.to_owned())
                    .or_insert_with(Child::new),
                None => return value.check(),
            };
            if matches!(value, Logical::Null) {
                return Ok(());
            }
            *child.counts.entry(stratum).or_default() += 1;
            child.family.merge(value, depth + 1, stratum, policy)
        })
    }

    /// Records `count` rows holding the scalar leaf at `path` with type `scalar`.
    ///
    /// Used when counts come from footers rather than a sample: there is one
    /// stratum, and each object on the path counts the most rows any of its
    /// leaves holds, since an object is present wherever one of its leaves is.
    fn insert_counted(&mut self, path: &[String], scalar: Scalar, count: usize) {
        let Some((name, rest)) = path.split_first() else {
            return;
        };
        let child = self.children.entry(name.clone()).or_insert_with(Child::new);
        let counted = child.counts.entry(0).or_default();
        *counted = (*counted).max(count);
        if rest.is_empty() {
            child.family = Family::Scalar(scalar);
            return;
        }
        if !matches!(child.family, Family::Object(_)) {
            child.family = Family::Object(ObjectNode::default());
        }
        if let Family::Object(node) = &mut child.family {
            node.insert_counted(rest, scalar, count);
        }
    }

    /// The shredding type of this root node under `rules`; `None` when nothing is kept.
    fn shredding_type(
        &self,
        rules: &[StratumRule],
        policy: &VariantShreddingPolicy,
    ) -> Option<DataType> {
        let mut path = Vec::new();
        match self
            .emit(ShreddedSchemaBuilder::new(), &mut path, rules, policy)
            .build()
        {
            DataType::Null => None,
            shredding_type => Some(shredding_type),
        }
    }

    /// Adds this node's kept children to `builder` under `path`.
    ///
    /// Children not eligible in any stratum under `rules`, or without one
    /// compatible family, are skipped; the remaining ones are capped by
    /// estimated rows covered, ties broken by name, and emitted alphabetically.
    fn emit<'a>(
        &'a self,
        mut builder: ShreddedSchemaBuilder,
        path: &mut Vec<&'a str>,
        rules: &[StratumRule],
        policy: &VariantShreddingPolicy,
    ) -> ShreddedSchemaBuilder {
        let mut kept: Vec<(&'a String, &'a Child, f64)> = self
            .children
            .iter()
            .filter(|(_, child)| {
                child.eligible(rules)
                    && matches!(child.family, Family::Object(_) | Family::Scalar(_))
            })
            .map(|(name, child)| (name, child, child.covered(rules)))
            .collect();
        kept.sort_by(|(left_name, _, left), (right_name, _, right)| {
            right.total_cmp(left).then(left_name.cmp(right_name))
        });
        kept.truncate(policy.max_emitted_children);
        kept.sort_by_key(|(name, _, _)| *name);
        for (name, child, _) in kept {
            path.push(name);
            builder = match &child.family {
                Family::Object(node) => node.emit(builder, path, rules, policy),
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
    /// Merges one non-null value observed at `depth` in `stratum` into this family.
    ///
    /// # Errors
    ///
    /// Returns the errors of reading the value's fields.
    fn merge(
        &mut self,
        value: Logical<'_, '_>,
        depth: usize,
        stratum: usize,
        policy: &VariantShreddingPolicy,
    ) -> Result<()> {
        match (&mut *self, value) {
            (Family::Residual, value) => return value.check(),
            (_, Logical::List) => *self = Family::Residual,
            (_, value @ Logical::Object(_)) if depth > policy.max_depth => {
                *self = Family::Residual;
                return value.check();
            }
            (Family::Unknown, Logical::Object(object)) => {
                let mut node = ObjectNode::default();
                node.observe_object(&object, depth, stratum, policy)?;
                *self = Family::Object(node);
            }
            (Family::Object(node), Logical::Object(object)) => {
                node.observe_object(&object, depth, stratum, policy)?
            }
            (Family::Object(_), Logical::Scalar(_)) => *self = Family::Residual,
            (Family::Scalar(_), value @ Logical::Object(_)) => {
                *self = Family::Residual;
                return value.check();
            }
            (Family::Unknown, Logical::Scalar(scalar)) => *self = Family::Scalar(scalar),
            (Family::Scalar(current), Logical::Scalar(scalar)) => {
                *self = current
                    .widen(scalar)
                    .map_or(Family::Residual, Family::Scalar)
            }
            (_, Logical::Null | Logical::Missing) => {}
        }
        Ok(())
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

    /// The family a shredded leaf of `data_type` was written for; `None` for non-scalar types.
    fn from_data_type(data_type: &DataType) -> Option<Self> {
        Some(match data_type {
            DataType::Boolean => Scalar::Boolean,
            DataType::Int8 => Scalar::Int(1),
            DataType::Int16 => Scalar::Int(2),
            DataType::Int32 => Scalar::Int(4),
            DataType::Int64 => Scalar::Int(8),
            DataType::Decimal32(precision, scale)
            | DataType::Decimal64(precision, scale)
            | DataType::Decimal128(precision, scale) => {
                let scale = u8::try_from(*scale).ok()?;
                Scalar::decimal(*precision, scale)
            }
            DataType::Float32 => Scalar::Float,
            DataType::Float64 => Scalar::Double,
            DataType::Date32 => Scalar::Date,
            DataType::Time64(TimeUnit::Microsecond) => Scalar::Time,
            DataType::Timestamp(unit, zone) => Scalar::Timestamp {
                nanos: *unit == TimeUnit::Nanosecond,
                utc: zone.is_some(),
            },
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Scalar::Binary,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Scalar::String,
            DataType::FixedSizeBinary(16) => Scalar::Uuid,
            _ => return None,
        })
    }

    /// The type covering the most rows of one path across sources, with its rows.
    ///
    /// Types of one family widen together and add their rows; across families
    /// the one with more rows wins, ties going to the first seen.
    fn winner(typed: Vec<(Scalar, usize)>) -> Option<(Scalar, usize)> {
        let mut families: Vec<(Scalar, usize)> = Vec::new();
        for (scalar, count) in typed {
            match families
                .iter_mut()
                .find_map(|(seen, rows)| seen.widen(scalar).map(|widened| (seen, rows, widened)))
            {
                Some((seen, rows, widened)) => {
                    *seen = widened;
                    *rows += count;
                }
                None => families.push((scalar, count)),
            }
        }
        families
            .into_iter()
            .reduce(|best, next| if next.1 > best.1 { next } else { best })
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

/// Builds Parquet writers that shred Variant columns with one layout chosen in advance.
///
/// Every file the builder opens uses the same layout, so all outputs of one
/// rewrite share it. The builder carries only the wrapped Parquet builder and
/// the layout.
#[derive(Clone, Debug)]
pub struct VariantParquetWriterBuilder {
    /// Ordinary Parquet builder each writer is opened from.
    inner: ParquetWriterBuilder,
    /// Layout every file is shredded with.
    layout: VariantLayout,
}

impl VariantParquetWriterBuilder {
    /// Wraps `inner` so its files shred Variant columns with `layout`.
    pub fn new(inner: ParquetWriterBuilder, layout: VariantLayout) -> Self {
        Self { inner, layout }
    }
}

impl FileWriterBuilder for VariantParquetWriterBuilder {
    type R = VariantParquetWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        let builder = if self.layout.is_unshredded() {
            self.inner.clone()
        } else {
            let logical: SchemaRef = Arc::new(self.inner.schema().as_ref().try_into()?);
            self.inner
                .clone()
                .with_physical_schema(self.layout.physical_schema(&logical)?)?
        };
        Ok(VariantParquetWriter {
            writer: builder.build(output_file).await?,
            layout: self.layout.clone(),
        })
    }
}

/// A Parquet writer that shreds every batch with its layout before encoding it.
///
/// The wrapped [`ParquetWriter`] opens its file on the first write, so an
/// output that receives no rows creates no file.
pub struct VariantParquetWriter {
    /// Ordinary Parquet writer for the physical schema.
    writer: ParquetWriter,
    /// Layout every batch is shredded with.
    layout: VariantLayout,
}

impl FileWriter for VariantParquetWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(&self.layout.shred(batch)?).await
    }

    async fn close(self) -> Result<Vec<DataFileBuilder>> {
        self.writer.close().await
    }
}

impl CurrentFileStatus for VariantParquetWriter {
    fn current_file_path(&self) -> String {
        self.writer.current_file_path()
    }

    fn current_row_num(&self) -> usize {
        self.writer.current_row_num()
    }

    fn current_written_size(&self) -> usize {
        self.writer.current_written_size()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use bytes::Bytes;
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::ParquetMetaDataReader;
    use parquet::variant::{json_to_variant, unshred_variant};

    use super::*;

    /// The policy values Wyrd passes.
    const POLICY: VariantShreddingPolicy = VariantShreddingPolicy {
        confidence_z: 2.5758,
        margin: 0.02,
        min_stratum_rows: 30,
        min_frequency: 0.10,
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

    /// The layout sampled from `batch` with row `i` in stratum `strata[i]`.
    fn sample(
        batch: &RecordBatch,
        strata: &[usize],
        policy: VariantShreddingPolicy,
        seed: u64,
    ) -> VariantLayout {
        let mut rows = Vec::new();
        for &stratum in strata {
            if rows.len() <= stratum {
                rows.resize(stratum + 1, 0);
            }
            rows[stratum] += 1;
        }
        let mut sampler = VariantSampler::new(&batch.schema(), policy, seed, &rows);
        sampler.offer(batch, strata, 0).unwrap();
        sampler.layout()
    }

    /// The shredding type of column 0 sampled from all of `json`, as one stratum.
    fn infer_with(json: &[Option<&str>], policy: VariantShreddingPolicy) -> Option<DataType> {
        let batch = variant_batch(json);
        sample(&batch, &vec![0; json.len()], policy, 7)
            .shredding_type(&["v"])
            .cloned()
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

    /// The field names shredded in `shredding_type`, in emitted order.
    fn names(shredding_type: Option<&DataType>) -> Vec<String> {
        shredded_fields(shredding_type.cloned())
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    /// The fields inferred from `json` under the default policy.
    fn infer(json: &[Option<&str>]) -> Vec<(String, DataType)> {
        shredded_fields(infer_with(json, POLICY))
    }

    /// The footer of an in-memory Parquet file holding `batch` shredded as sampled from itself.
    fn shredded_footer(batch: &RecordBatch) -> Arc<ParquetMetaData> {
        let layout = sample(batch, &vec![0; batch.num_rows()], POLICY, 7);
        let physical = layout.shred(batch).unwrap();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, physical.schema(), None).unwrap();
        writer.write(&physical).unwrap();
        writer.close().unwrap();
        Arc::new(
            ParquetMetaDataReader::new()
                .parse_and_finish(&Bytes::from(bytes))
                .unwrap(),
        )
    }

    /// Cochran's size is 4147 for a large stratum at 99% / ±0.02, shrinks with the
    /// finite-population correction, and is zero for an empty stratum or the default policy.
    #[test]
    fn stratum_sample_size_follows_cochran() {
        assert_eq!(POLICY.stratum_sample_size(1_000_000_000), 4147);
        assert_eq!(POLICY.stratum_sample_size(100), 98);
        assert_eq!(POLICY.stratum_sample_size(10), 10);
        assert_eq!(POLICY.stratum_sample_size(0), 0);
        assert_eq!(
            VariantShreddingPolicy::default().stratum_sample_size(1_000),
            0
        );
    }

    /// A field is eligible at 8% (10% minus the margin) of sampled non-null roots; nulls do not count.
    #[test]
    fn frequency_boundary_ignores_nulls() {
        let mut rows = vec![Some(r#"{"a":1}"#)];
        rows.extend(std::iter::repeat_n(Some(r#"{"b":1}"#), 11));
        rows.extend(std::iter::repeat_n(None, 5));
        assert_eq!(infer(&rows), [
            ("a".to_string(), DataType::Int8),
            ("b".to_string(), DataType::Int8)
        ]);
        rows.push(Some(r#"{"b":2}"#));
        assert_eq!(infer(&rows), [("b".to_string(), DataType::Int8)]);
    }

    /// A field common in one writer's rows but rare overall is shredded, whether
    /// that writer has its own stratum or shares the merged small-writer stratum.
    #[test]
    fn small_writers_keep_their_fields() {
        let mut json = vec![Some(r#"{"rare":1}"#); 4];
        json.extend(std::iter::repeat_n(Some(r#"{"x":1}"#), 36));
        json.extend(std::iter::repeat_n(Some(r#"{"common":1}"#), 60));
        let mut strata = vec![0; 40];
        strata.extend(std::iter::repeat_n(1, 60));
        let batch = variant_batch(&json);
        let own = sample(&batch, &strata, POLICY, 7);
        assert!(names(own.shredding_type(&["v"])).contains(&"rare".to_string()));
        let one = sample(&batch, &vec![0; 100], POLICY, 7);
        assert!(
            !names(one.shredding_type(&["v"])).contains(&"rare".to_string()),
            "4 of 100 rows is below 8% without strata"
        );

        let mut json = vec![Some(r#"{"tiny":1}"#); 5];
        json.extend(std::iter::repeat_n(Some(r#"{"x":1}"#), 20));
        json.extend(std::iter::repeat_n(Some(r#"{"common":1}"#), 75));
        let mut strata = vec![0; 5];
        strata.extend(std::iter::repeat_n(1, 20));
        strata.extend(std::iter::repeat_n(2, 75));
        let merged = sample(&variant_batch(&json), &strata, POLICY, 7);
        assert!(
            names(merged.shredding_type(&["v"])).contains(&"tiny".to_string()),
            "5 of the 25 rows in the merged small-writer stratum"
        );
    }

    /// The same seed and positions select the same rows and layout; the rate
    /// matches Cochran's size over the stratum's rows.
    #[test]
    fn sampling_is_seeded_and_reproducible() {
        let rows = 20_000;
        let schema = variant_batch(&[None]).schema();
        let run = |seed: u64| {
            let mut sampler = VariantSampler::new(&schema, POLICY, seed, &[rows]);
            let documents: Vec<Option<String>> = (0..rows)
                .map(|row| Some(format!(r#"{{"k{}":1}}"#, row % 7)))
                .collect();
            let json: Vec<Option<&str>> = documents.iter().map(Option::as_deref).collect();
            for (chunk, start) in json.chunks(4_096).zip((0..).step_by(4_096)) {
                let batch = variant_batch(chunk);
                sampler.offer(&batch, &vec![0; chunk.len()], start).unwrap();
            }
            (sampler.kept[0], sampler.layout())
        };
        let (kept, layout) = run(42);
        assert_eq!(run(42), (kept, layout.clone()));
        let expected = POLICY.stratum_sample_size(rows);
        assert!(
            kept.abs_diff(expected) < expected / 10,
            "kept {kept}, expected about {expected}"
        );
        assert_eq!(names(layout.shredding_type(&["v"])).len(), 7);
    }

    /// A shredded input samples the same rows and layout as its logical values,
    /// in a top-level column and inside a List, and keeps residual-only fields
    /// discoverable.
    #[test]
    fn shredded_input_samples_like_its_logical_values() {
        let documents: Vec<String> = (0..20_000)
            .map(|row| match row % 4 {
                0 => format!(r#"{{"a":{row},"b":"x{row}"}}"#),
                1 => format!(r#"{{"a":{row},"c":{{"d":[1,2]}}}}"#),
                2 => r#"{"a":"text","b":null}"#.to_owned(),
                _ => "null".to_owned(),
            })
            .collect();
        let json: Vec<Option<&str>> = documents
            .iter()
            .enumerate()
            .map(|(row, document)| (row % 9 != 0).then_some(document.as_str()))
            .collect();
        let logical = variant_batch(&json);
        let strata = vec![0; logical.num_rows()];
        let only_a = sample(&variant_batch(&[Some(r#"{"a":1}"#)]), &[0], POLICY, 7);
        let shredded = only_a.shred(&logical).unwrap();
        assert_ne!(shredded.schema(), logical.schema());
        let expected = sample(&logical, &strata, POLICY, 11);
        assert_eq!(sample(&shredded, &strata, POLICY, 11), expected);
        assert!(names(expected.shredding_type(&["v"])).contains(&"b".to_owned()));

        let row: &[&str] = &[r#"{"k":1,"s":"a"}"#, r#"{"k":2,"s":"b","rare":true}"#];
        let events = events_batch(&vec![row; 40]);
        let path = ["events", "element", "attributes"];
        let events_strata = vec![0; events.num_rows()];
        let only_k = sample(&events_batch(&[&[r#"{"k":1}"#]]), &[0], POLICY, 7);
        assert_eq!(names(only_k.shredding_type(&path)), ["k"]);
        let shredded = only_k.shred(&events).unwrap();
        assert_eq!(
            sample(&shredded, &events_strata, POLICY, 7),
            sample(&events, &events_strata, POLICY, 7)
        );
    }

    /// At the cap, children are kept by estimated rows covered, not raw sampled counts.
    #[test]
    fn cap_ranks_by_rows_covered() {
        let policy = VariantShreddingPolicy {
            max_emitted_children: 2,
            ..POLICY
        };
        let mut node = ObjectNode::default();
        let observe = |node: &mut ObjectNode, json: &str, stratum: usize| {
            let batch = variant_batch(&[Some(json)]);
            let variant = VariantArray::try_new(batch.column(0).as_ref()).unwrap();
            if let Logical::Object(object) = Logical::of(variant.value(0)) {
                node.observe_object(&object, 1, stratum, &policy).unwrap();
            }
        };
        observe(&mut node, r#"{"x":1}"#, 0);
        for _ in 0..5 {
            observe(&mut node, r#"{"y":1}"#, 1);
        }
        for _ in 0..3 {
            observe(&mut node, r#"{"z":1}"#, 1);
        }
        let rules = [
            StratumRule {
                min_count: 1,
                weight: 100.0,
            },
            StratumRule {
                min_count: 1,
                weight: 1.0,
            },
        ];
        assert_eq!(names(node.shredding_type(&rules, &policy).as_ref()), [
            "x", "y"
        ]);
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
        assert_eq!(
            shredded_fields(
                sample(&batch, &[0, 0], POLICY, 7)
                    .shredding_type(&["v"])
                    .cloned()
            ),
            [("d".to_string(), DataType::Decimal64(18, 3))]
        );
    }

    /// Children are kept by count with ties broken by name, emitted alphabetically, and capped.
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

    /// The default policy samples nothing, so it writes unshredded files.
    #[test]
    fn default_policy_never_shreds() {
        assert!(infer_with(&[Some(r#"{"a":1}"#)], VariantShreddingPolicy::default()).is_none());
    }

    /// Shredding keeps logical values: a later value that does not fit the type goes to the residual.
    #[test]
    fn later_incompatible_values_round_trip() {
        let layout = sample(&variant_batch(&[Some(r#"{"a":1}"#)]), &[0], POLICY, 7);
        let later = variant_batch(&[Some(r#"{"a":"text"}"#), Some(r#"{"a":[1,2]}"#), None]);
        let shredded = layout.shred(&later).unwrap();
        let logical: ArrayRef =
            unshred_variant(&VariantArray::try_new(shredded.column(0).as_ref()).unwrap())
                .unwrap()
                .into();
        let json = parquet::variant::variant_to_json(&logical).unwrap();
        assert_eq!(json.iter().collect::<Vec<_>>(), [
            Some(r#"{"a":"text"}"#),
            Some(r#"{"a":[1,2]}"#),
            None
        ]);
    }

    /// A batch with one `events: List<Struct<name, attributes: Variant>>`
    /// column; row `r` holds one event per JSON document in `rows[r]`.
    fn events_batch(rows: &[&[&str]]) -> RecordBatch {
        let documents: Vec<&str> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        let json = Arc::new(StringArray::from(documents.clone())) as ArrayRef;
        let attributes = json_to_variant(&json).unwrap();
        let element = Fields::from(vec![
            Arc::new(arrow_schema::Field::new("name", DataType::Utf8, true)),
            Arc::new(attributes.field("attributes")),
        ]);
        let structs = StructArray::try_new(
            element.clone(),
            vec![
                Arc::new(StringArray::from(vec!["e"; documents.len()])),
                attributes.into(),
            ],
            None,
        )
        .unwrap();
        let element = Arc::new(arrow_schema::Field::new(
            "element",
            DataType::Struct(element),
            true,
        ));
        let events = arrow_array::ListArray::try_new(
            Arc::clone(&element),
            arrow_buffer::OffsetBuffer::from_lengths(rows.iter().map(|row| row.len())),
            Arc::new(structs),
            None,
        )
        .unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![arrow_schema::Field::new(
                "events",
                DataType::List(element),
                true,
            )])),
            vec![Arc::new(events)],
        )
        .unwrap()
    }

    /// The JSON of every event's attributes in an `events` batch, logical or shredded.
    fn events_json(batch: &RecordBatch) -> Vec<Option<String>> {
        let (_, events) = crate::arrow::unshred_variants(
            batch.schema().fields().first().unwrap(),
            batch.column(0),
        )
        .unwrap();
        let structs = events.as_list::<i32>().values().as_struct();
        parquet::variant::variant_to_json(structs.column_by_name("attributes").unwrap())
            .unwrap()
            .iter()
            .map(|json| json.map(str::to_owned))
            .collect()
    }

    /// Span event attributes, a Variant inside a List of Structs, are
    /// sampled with each event as one root, shredded in place, read back
    /// unchanged, and combined from footers like a top-level column.
    #[test]
    fn variants_inside_lists_are_sampled_shredded_and_combined() {
        let row: &[&str] = &[r#"{"k":1,"s":"a"}"#, r#"{"k":2,"s":"b","rare":true}"#];
        let mut rows = vec![row; 20];
        let tail: &[&str] = &[r#"{"k":"text"}"#];
        rows.push(tail);
        let batch = events_batch(&rows);
        let path = ["events", "element", "attributes"];
        let layout = sample(&batch, &vec![0; batch.num_rows()], POLICY, 7);
        // `k` holds integers and one string, so it stays residual.
        assert_eq!(names(layout.shredding_type(&path)), ["rare", "s"]);

        let shredded = layout.shred(&batch).unwrap();
        assert_ne!(shredded.schema(), batch.schema());
        assert_eq!(events_json(&shredded), events_json(&batch));

        let footer = shredded_footer(&batch);
        let [leaves] =
            <[VariantLeaves; 1]>::try_from(variant_leaves(&batch.schema(), &footer).unwrap())
                .unwrap();
        let descr = footer.file_metadata().schema_descr();
        let at = |leaf: usize| descr.column(leaf).path().string();
        assert_eq!(leaves.all.clone().map(at).collect::<Vec<_>>(), [
            "events.list.element.attributes.metadata",
            "events.list.element.attributes.value",
            "events.list.element.attributes.typed_value.rare.value",
            "events.list.element.attributes.typed_value.rare.typed_value",
            "events.list.element.attributes.typed_value.s.value",
            "events.list.element.attributes.typed_value.s.typed_value",
        ]);
        assert_eq!(
            leaves.metadata.map(at).unwrap(),
            "events.list.element.attributes.metadata"
        );
        assert_eq!(
            leaves.value.map(at).unwrap(),
            "events.list.element.attributes.value"
        );
        let combined = VariantLayout::combine(&batch.schema(), &[footer], &POLICY).unwrap();
        assert_eq!(names(combined.shredding_type(&path)), ["rare", "s"]);
    }

    /// Footer counts combine across sources: same-family types widen, a field
    /// below 10% of the combined roots is dropped, and nested leaves survive.
    #[test]
    fn combine_counts_footer_leaves() {
        let mut first = vec![Some(r#"{"a":1,"o":{"x":"p"}}"#); 9];
        first.push(Some(r#"{"a":2,"o":{"x":"q"},"b":"rare"}"#));
        let second = vec![Some(r#"{"a":70000,"c":"s"}"#); 10];
        let footers = [
            shredded_footer(&variant_batch(&first)),
            shredded_footer(&variant_batch(&second)),
        ];
        let schema = variant_batch(&[None]).schema();
        let layout = VariantLayout::combine(&schema, &footers, &POLICY).unwrap();
        let fields = shredded_fields(layout.shredding_type(&["v"]).cloned());
        let by_name: BTreeMap<_, _> = fields.into_iter().collect();
        assert_eq!(
            by_name.keys().map(String::as_str).collect::<Vec<_>>(),
            ["a", "c", "o"],
            "b is 1 of 20 roots"
        );
        assert_eq!(by_name["a"], DataType::Int32);
    }

    /// Across type families the type covering more rows wins; no footers means no shredding.
    #[test]
    fn combine_keeps_the_type_covering_more_rows() {
        let footers = [
            shredded_footer(&variant_batch(&[Some(r#"{"x":1}"#); 5])),
            shredded_footer(&variant_batch(&[Some(r#"{"x":"s"}"#); 8])),
        ];
        let schema = variant_batch(&[None]).schema();
        let layout = VariantLayout::combine(&schema, &footers, &POLICY).unwrap();
        assert_eq!(shredded_fields(layout.shredding_type(&["v"]).cloned()), [(
            "x".to_string(),
            DataType::Utf8
        )]);
        assert!(
            VariantLayout::combine(&schema, &[], &POLICY)
                .unwrap()
                .is_unshredded()
        );
    }

    /// Counts the bytes this thread allocates, so a test can tell what one
    /// call allocates without other threads' noise.
    struct CountingAllocator;

    thread_local! {
        static ALLOCATED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Adds `bytes` to this thread's count; a thread being torn down is not counted.
    fn count(bytes: usize) {
        let _ = ALLOCATED.try_with(|allocated| allocated.set(allocated.get() + bytes));
    }

    // SAFETY: every call forwards unchanged to the system allocator.
    unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            count(layout.size());
            // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            count(layout.size());
            // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
            unsafe { std::alloc::System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(
            &self,
            ptr: *mut u8,
            layout: std::alloc::Layout,
            new_size: usize,
        ) -> *mut u8 {
            count(new_size);
            // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
            unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// Rows kept per stratum, sampled roots per field, and the layout of one
    /// sampler offered `batches`, each a batch and its first position, as one
    /// stratum of `rows` rows.
    fn counters(
        batches: &[(&RecordBatch, u64)],
        rows: usize,
        policy: VariantShreddingPolicy,
    ) -> Result<(Vec<usize>, Vec<BTreeMap<usize, usize>>, VariantLayout)> {
        let mut sampler = VariantSampler::new(&batches[0].0.schema(), policy, 5, &[rows]);
        for (batch, first) in batches {
            sampler.offer(batch, &vec![0; batch.num_rows()], *first)?;
        }
        let roots = sampler
            .analyzer
            .columns
            .iter()
            .map(|column| column.roots.clone())
            .collect();
        Ok((sampler.kept.clone(), roots, sampler.layout()))
    }

    /// A shredded or mixed input reads its chosen values in place: it selects
    /// the same rows and counts the same fields as its logical values, through
    /// typed/residual overlap, residual-only fields, nulls, missing fields,
    /// type conflicts, nested objects, and lists; malformed storage still
    /// fails; and longer payloads allocate nothing more.
    #[test]
    fn physical_sample_counts_without_payload_reconstruction() {
        // Every field seen once is kept, so the layout shows every count.
        let every = VariantShreddingPolicy {
            min_frequency: 0.0,
            ..POLICY
        };
        let documents = [
            r#"{"a":1,"o":{"x":2,"y":"r"},"new":"t"}"#,
            r#"{"a":"conflict","o":{"x":"s"}}"#,
            r#"{"a":null,"o":null}"#,
            r#"{"o":{"x":1.5}}"#,
            r#"{"a":300,"o":[1,2]}"#,
            r#"{"z":{"deep":{"k":true}},"o":{}}"#,
            r#""scalar""#,
            r#"[1,{"a":1}]"#,
            "null",
        ];
        let json: Vec<Option<&str>> = (0..400)
            .map(|row| (row % 10 != 9).then(|| documents[row % documents.len()]))
            .collect();
        let logical = variant_batch(&json);
        let seed = variant_batch(&[Some(r#"{"a":1,"o":{"x":1}}"#)]);
        let layout = sample(&seed, &[0], POLICY, 7);
        assert_eq!(names(layout.shredding_type(&["v"])), ["a", "o"]);
        let shredded = layout.shred(&logical).unwrap();
        assert_ne!(shredded.schema(), logical.schema());
        let expected = counters(&[(&logical, 0)], 400, every).unwrap();
        assert!(expected.0[0] > 300, "{:?}", expected.0);
        assert!(names(expected.2.shredding_type(&["v"])).contains(&"new".to_owned()));
        assert_eq!(counters(&[(&shredded, 0)], 400, every).unwrap(), expected);
        let (head, tail) = (logical.slice(0, 200), shredded.slice(200, 200));
        let mixed = layout.shred(&head).unwrap();
        assert_eq!(
            counters(&[(&head, 0), (&tail, 200)], 400, every).unwrap(),
            expected,
            "canonical then shredded"
        );
        assert_eq!(
            counters(&[(&mixed, 0), (&logical.slice(200, 200), 200)], 400, every).unwrap(),
            expected,
            "shredded then canonical"
        );

        let row: &[&str] = &[r#"{"k":1,"s":"a"}"#, r#"{"k":"two","rare":[true]}"#];
        let events = events_batch(&vec![row; 40]);
        let only_k = sample(&events_batch(&[&[r#"{"k":1}"#]]), &[0], POLICY, 7);
        let rows = events.num_rows();
        assert_eq!(
            counters(&[(&only_k.shred(&events).unwrap(), 0)], rows, every).unwrap(),
            counters(&[(&events, 0)], rows, every).unwrap()
        );

        // Malformed storage fails as reconstruction does: a residual that is
        // not an object, and a residual repeating a shredded field.
        let only_a = sample(&variant_batch(&[Some(r#"{"a":1}"#)]), &[0], POLICY, 7);
        let valid = only_a.shred(&variant_batch(&[Some(r#"{"a":1}"#)])).unwrap();
        for residual in [r#"5"#, r#"{"a":2}"#] {
            let residual = variant_batch(&[Some(residual)]);
            let residual = residual
                .column(0)
                .as_struct()
                .column_by_name("value")
                .unwrap();
            let storage = valid.column(0).as_struct();
            let columns = storage
                .fields()
                .iter()
                .zip(storage.columns())
                .map(|(field, column)| match field.name().as_str() {
                    "value" => Arc::clone(residual),
                    _ => Arc::clone(column),
                })
                .collect();
            let fields = storage
                .fields()
                .iter()
                .map(|field| match field.name().as_str() {
                    "value" => Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(residual.data_type().clone()),
                    ),
                    _ => Arc::clone(field),
                })
                .collect::<Fields>();
            let storage = StructArray::try_new(fields, columns, None).unwrap();
            let schema = Schema::new(vec![
                valid
                    .schema()
                    .field(0)
                    .clone()
                    .with_data_type(storage.data_type().clone()),
            ]);
            let malformed =
                RecordBatch::try_new(Arc::new(schema), vec![Arc::new(storage)]).unwrap();
            let variant = VariantArray::try_new(malformed.column(0).as_ref()).unwrap();
            assert!(
                unshred_variant(&variant).is_err(),
                "reconstruction rejects it"
            );
            assert!(
                counters(&[(&malformed, 0)], 1, every).is_err(),
                "sampling rejects it"
            );
        }

        // Reading a chosen value allocates nothing for its typed or residual payload.
        let offered = |payload: usize| {
            let text = "p".repeat(payload);
            let documents: Vec<String> = (0..100)
                .map(|row| format!(r#"{{"s":"{text}","r":"{text}","n":{row}}}"#))
                .collect();
            let json: Vec<Option<&str>> = documents.iter().map(|doc| Some(doc.as_str())).collect();
            let only_s = sample(&variant_batch(&[Some(r#"{"s":"x"}"#)]), &[0], POLICY, 7);
            let batch = only_s.shred(&variant_batch(&json)).unwrap();
            let mut sampler = VariantSampler::new(&batch.schema(), POLICY, 5, &[100]);
            let before = ALLOCATED.with(std::cell::Cell::get);
            sampler.offer(&batch, &[0; 100], 0).unwrap();
            let allocated = ALLOCATED.with(std::cell::Cell::get) - before;
            assert!(sampler.sampled_rows() > 90, "{}", sampler.sampled_rows());
            allocated
        };
        let (short, long) = (offered(16), offered(64 * 1024));
        assert!(
            long.saturating_sub(short) < 64 * 1024,
            "a longer payload allocated {short} then {long} bytes"
        );
    }
}
