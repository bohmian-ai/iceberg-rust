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

//! Evaluates predicates against a Parquet Page Index

use std::collections::HashMap;

use fnv::FnvHashSet;
use ordered_float::OrderedFloat;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;

use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::decimal_utils::i128_from_be_bytes;
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType, Schema};
use crate::{Error, ErrorKind, Result};

const IN_PREDICATE_LIMIT: usize = 200;

enum MissingColBehavior {
    CantMatch,
    MightMatch,
}

enum PageNullCount {
    AllNull,
    NoneNull,
    SomeNull,
    Unknown,
}

impl PageNullCount {
    fn from_row_and_null_counts(num_rows: usize, null_count: Option<i64>) -> Self {
        match (num_rows, null_count) {
            (x, Some(y)) if x == y as usize => PageNullCount::AllNull,
            (_, Some(0)) => PageNullCount::NoneNull,
            (_, Some(_)) => PageNullCount::SomeNull,
            _ => PageNullCount::Unknown,
        }
    }
}

/// One page's physical min or max value as stored in a Parquet column index.
///
/// Bounds are carried in their physical form so that a single place,
/// [`PageBound::to_datum`], decides how each physical type is read under an
/// Iceberg type.
#[derive(Clone, Copy)]
enum PageBound<'b> {
    /// `BOOLEAN` bound.
    Boolean(bool),
    /// `INT32` bound.
    Int32(i32),
    /// `INT64` bound.
    Int64(i64),
    /// `FLOAT` bound.
    Float(f32),
    /// `DOUBLE` bound.
    Double(f64),
    /// `BYTE_ARRAY` bound; writers may truncate it and it need not be UTF-8.
    ByteArray(&'b [u8]),
    /// `FIXED_LEN_BYTE_ARRAY` bound; writers may truncate it.
    FixedLenByteArray(&'b [u8]),
}

impl PageBound<'_> {
    /// Interprets an optional page bound under `field_type`.
    ///
    /// Returns `None` when the physical/Iceberg pairing is unsupported, so
    /// the caller keeps every page of the column. Returns `Some(None)` when
    /// the page has no bound (all-null page) or the bound is malformed for
    /// the type, which predicates treat as "might match".
    fn option_to_datum(bound: Option<Self>, field_type: &PrimitiveType) -> Option<Option<Datum>> {
        match bound {
            None => Some(None),
            Some(bound) => bound.to_datum(field_type),
        }
    }

    /// Interprets this physical bound as an Iceberg [`Datum`] of `field_type`.
    ///
    /// Byte bounds keep their unsigned byte order: `binary` and `fixed`
    /// compare as raw bytes (never decoded as UTF-8), `uuid` requires exactly
    /// 16 bytes, `string` requires valid UTF-8, and `decimal` decodes the
    /// untruncated big-endian two's-complement `FIXED_LEN_BYTE_ARRAY`.
    /// Integer-backed decimals widen to the decimal's unscaled value, and
    /// `INT32`/`FLOAT` widen to `long`/`double` for promoted columns.
    ///
    /// Returns `None` for an unsupported pairing and `Some(None)` for a
    /// bound that is malformed for an otherwise supported pairing.
    fn to_datum(self, field_type: &PrimitiveType) -> Option<Option<Datum>> {
        let literal = match (field_type, self) {
            (PrimitiveType::Boolean, Self::Boolean(v)) => PrimitiveLiteral::Boolean(v),
            (PrimitiveType::Int | PrimitiveType::Date, Self::Int32(v)) => PrimitiveLiteral::Int(v),
            (PrimitiveType::Long, Self::Int32(v)) => PrimitiveLiteral::Long(i64::from(v)),
            (
                PrimitiveType::Long
                | PrimitiveType::Time
                | PrimitiveType::Timestamp
                | PrimitiveType::Timestamptz
                | PrimitiveType::TimestampNs
                | PrimitiveType::TimestamptzNs,
                Self::Int64(v),
            ) => PrimitiveLiteral::Long(v),
            (PrimitiveType::Decimal { .. }, Self::Int32(v)) => {
                PrimitiveLiteral::Int128(i128::from(v))
            }
            (PrimitiveType::Decimal { .. }, Self::Int64(v)) => {
                PrimitiveLiteral::Int128(i128::from(v))
            }
            (PrimitiveType::Float, Self::Float(v)) => PrimitiveLiteral::Float(OrderedFloat(v)),
            (PrimitiveType::Double, Self::Float(v)) => {
                PrimitiveLiteral::Double(OrderedFloat(f64::from(v)))
            }
            (PrimitiveType::Double, Self::Double(v)) => PrimitiveLiteral::Double(OrderedFloat(v)),
            (PrimitiveType::String, Self::ByteArray(bytes)) => match std::str::from_utf8(bytes) {
                Ok(value) => PrimitiveLiteral::String(value.to_owned()),
                Err(_) => return Some(None),
            },
            (PrimitiveType::Binary, Self::ByteArray(bytes))
            | (PrimitiveType::Fixed(_), Self::FixedLenByteArray(bytes)) => {
                PrimitiveLiteral::Binary(bytes.to_vec())
            }
            (PrimitiveType::Uuid, Self::FixedLenByteArray(bytes)) => {
                match <[u8; 16]>::try_from(bytes) {
                    Ok(bytes) => PrimitiveLiteral::UInt128(u128::from_be_bytes(bytes)),
                    Err(_) => return Some(None),
                }
            }
            (PrimitiveType::Decimal { .. }, Self::FixedLenByteArray(bytes)) => {
                match i128_from_be_bytes(bytes) {
                    Some(value) => PrimitiveLiteral::Int128(value),
                    None => return Some(None),
                }
            }
            _ => return None,
        };
        Some(Some(Datum::new(field_type.clone(), literal)))
    }
}

pub(crate) struct PageIndexEvaluator<'a> {
    column_index: &'a [Option<ColumnIndexMetaData>],
    offset_index: &'a [Option<OffsetIndexMetaData>],
    row_group_metadata: &'a RowGroupMetaData,
    iceberg_field_id_to_parquet_column_index: &'a HashMap<i32, usize>,
    snapshot_schema: &'a Schema,
    row_count_cache: HashMap<usize, Vec<usize>>,
}

impl<'a> PageIndexEvaluator<'a> {
    pub(crate) fn new(
        column_index: &'a [Option<ColumnIndexMetaData>],
        offset_index: &'a [Option<OffsetIndexMetaData>],
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
        snapshot_schema: &'a Schema,
    ) -> Self {
        Self {
            column_index,
            offset_index,
            row_group_metadata,
            iceberg_field_id_to_parquet_column_index: field_id_map,
            snapshot_schema,
            row_count_cache: HashMap::new(),
        }
    }

    /// Evaluate this `PageIndexEvaluator`'s filter predicate against a
    /// specific page's column index entry in a parquet file's page index.
    /// [`ArrowReader`] uses the resulting [`RowSelection`] to reject
    /// pages within a parquet file's row group that cannot contain rows
    /// matching the filter predicate.
    pub(crate) fn eval(
        filter: &'a BoundPredicate,
        column_index: &'a [Option<ColumnIndexMetaData>],
        offset_index: &'a [Option<OffsetIndexMetaData>],
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
        snapshot_schema: &'a Schema,
    ) -> Result<Vec<RowSelector>> {
        if row_group_metadata.num_rows() == 0 {
            return Ok(vec![]);
        }

        let mut evaluator = Self::new(
            column_index,
            offset_index,
            row_group_metadata,
            field_id_map,
            snapshot_schema,
        );

        Ok(visit(&mut evaluator, filter)?.iter().copied().collect())
    }

    fn select_all_rows(&self) -> Result<RowSelection> {
        Ok(vec![RowSelector::select(
            self.row_group_metadata.num_rows() as usize
        )]
        .into())
    }

    fn skip_all_rows(&self) -> Result<RowSelection> {
        Ok(vec![RowSelector::skip(
            self.row_group_metadata.num_rows() as usize
        )]
        .into())
    }

    fn calc_row_selection<F>(
        &mut self,
        field_id: i32,
        predicate: F,
        missing_col_behavior: MissingColBehavior,
    ) -> Result<RowSelection>
    where
        F: Fn(Option<Datum>, Option<Datum>, PageNullCount) -> Result<bool>,
    {
        let Some(&parquet_column_index) =
            self.iceberg_field_id_to_parquet_column_index.get(&field_id)
        else {
            // if the snapshot's column is not present in the row group,
            // exit early
            return match missing_col_behavior {
                MissingColBehavior::CantMatch => self.skip_all_rows(),
                MissingColBehavior::MightMatch => self.select_all_rows(),
            };
        };

        let Some(field) = self.snapshot_schema.field_by_id(field_id) else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Field with id {field_id} missing from snapshot schema"),
            ));
        };

        let Some(field_type) = field.field_type.as_primitive_type() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Field with id {field_id} not convertible to primitive type"),
            ));
        };

        let Some(column_index) = self
            .column_index
            .get(parquet_column_index)
            .and_then(Option::as_ref)
        else {
            // This should not happen, but we fail soft anyway so that the scan is still
            // successful, just a bit slower
            return self.select_all_rows();
        };

        let row_counts = {
            // Caches row count calculations for columns that appear multiple times in
            // the predicate
            match self.row_count_cache.get(&parquet_column_index) {
                Some(count) => count.clone(),
                None => {
                    let Some(offset_index) = self
                        .offset_index
                        .get(parquet_column_index)
                        .and_then(Option::as_ref)
                    else {
                        // A page index provider can load column and offset indexes
                        // independently. Without offsets, we cannot construct a row
                        // selection, so preserve correctness by skipping page pruning.
                        return self.select_all_rows();
                    };

                    let count = self.calc_row_counts(offset_index);
                    self.row_count_cache
                        .insert(parquet_column_index, count.clone());

                    count
                }
            }
        };

        let Some(page_filter) = Self::apply_predicate_to_column_index(
            predicate,
            field_type,
            column_index,
            &row_counts,
        )?
        else {
            return self.select_all_rows();
        };

        let row_selectors: Vec<_> = row_counts
            .iter()
            .zip(page_filter.iter())
            .map(|(&row_count, &is_selected)| {
                if is_selected {
                    RowSelector::select(row_count)
                } else {
                    RowSelector::skip(row_count)
                }
            })
            .collect();

        Ok(row_selectors.into())
    }

    /// Returns a list of row counts per page
    fn calc_row_counts(&self, offset_index: &OffsetIndexMetaData) -> Vec<usize> {
        let mut remaining_rows = self.row_group_metadata.num_rows() as usize;
        let mut row_counts = Vec::with_capacity(self.offset_index.len());

        let page_locations = offset_index.page_locations();
        for (idx, page_location) in page_locations.iter().enumerate() {
            let row_count = if idx < page_locations.len() - 1 {
                let row_count = (page_locations[idx + 1].first_row_index
                    - page_location.first_row_index) as usize;
                remaining_rows -= row_count;
                row_count
            } else {
                remaining_rows
            };

            row_counts.push(row_count);
        }

        row_counts
    }

    /// Evaluates `predicate` against every page of one column index and
    /// returns the per-page keep decision.
    ///
    /// Each page's physical min/max is first read as a [`PageBound`] and then
    /// interpreted under the Iceberg `field_type` by [`PageBound::to_datum`].
    /// Returns `Ok(None)` — keep every page — when the column carries no
    /// index (`NONE`), uses a physical type with no comparable bound
    /// (`INT96`), or pairs a physical type with an Iceberg type that has no
    /// defined reading. Malformed individual bounds become unknown bounds,
    /// which predicates treat as "might match".
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `predicate`.
    fn apply_predicate_to_column_index<F>(
        predicate: F,
        field_type: &PrimitiveType,
        column_index: &ColumnIndexMetaData,
        row_counts: &[usize],
    ) -> Result<Option<Vec<bool>>>
    where
        F: Fn(Option<Datum>, Option<Datum>, PageNullCount) -> Result<bool>,
    {
        fn pages<'b, T: 'b>(
            mins: impl Iterator<Item = Option<&'b T>>,
            maxs: impl Iterator<Item = Option<&'b T>>,
            to_bound: impl Fn(&'b T) -> PageBound<'b>,
        ) -> Vec<(Option<PageBound<'b>>, Option<PageBound<'b>>)> {
            mins.zip(maxs)
                .map(|(min, max)| (min.map(&to_bound), max.map(&to_bound)))
                .collect()
        }

        let bounds = match column_index {
            ColumnIndexMetaData::INT96(_) => return Ok(None),
            ColumnIndexMetaData::BOOLEAN(idx) => {
                pages(idx.min_values_iter(), idx.max_values_iter(), |&v| {
                    PageBound::Boolean(v)
                })
            }
            ColumnIndexMetaData::INT32(idx) => {
                pages(idx.min_values_iter(), idx.max_values_iter(), |&v| {
                    PageBound::Int32(v)
                })
            }
            ColumnIndexMetaData::INT64(idx) => {
                pages(idx.min_values_iter(), idx.max_values_iter(), |&v| {
                    PageBound::Int64(v)
                })
            }
            ColumnIndexMetaData::FLOAT(idx) => {
                pages(idx.min_values_iter(), idx.max_values_iter(), |&v| {
                    PageBound::Float(v)
                })
            }
            ColumnIndexMetaData::DOUBLE(idx) => {
                pages(idx.min_values_iter(), idx.max_values_iter(), |&v| {
                    PageBound::Double(v)
                })
            }
            ColumnIndexMetaData::BYTE_ARRAY(idx) => idx
                .min_values_iter()
                .zip(idx.max_values_iter())
                .map(|(min, max)| (min.map(PageBound::ByteArray), max.map(PageBound::ByteArray)))
                .collect(),
            ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(idx) => idx
                .min_values_iter()
                .zip(idx.max_values_iter())
                .map(|(min, max)| {
                    (
                        min.map(PageBound::FixedLenByteArray),
                        max.map(PageBound::FixedLenByteArray),
                    )
                })
                .collect(),
        };

        let mut selected = Vec::with_capacity(bounds.len());
        for (i, ((min, max), &row_count)) in bounds.into_iter().zip(row_counts).enumerate() {
            let (Some(min), Some(max)) = (
                PageBound::option_to_datum(min, field_type),
                PageBound::option_to_datum(max, field_type),
            ) else {
                return Ok(None);
            };
            selected.push(predicate(
                min,
                max,
                PageNullCount::from_row_and_null_counts(row_count, column_index.null_count(i)),
            )?);
        }

        Ok(Some(selected))
    }

    fn visit_inequality(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        cmp_fn: fn(&Datum, &Datum) -> bool,
        use_lower_bound: bool,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |min, max, null_count| {
                if matches!(null_count, PageNullCount::AllNull) {
                    return Ok(false);
                }

                if datum.is_nan() {
                    // NaN indicates unreliable bounds.
                    return Ok(true);
                }

                let bound = if use_lower_bound { min } else { max };

                if let Some(bound) = bound {
                    if cmp_fn(&bound, datum) {
                        return Ok(true);
                    }

                    return Ok(false);
                }

                Ok(true)
            },
            MissingColBehavior::MightMatch,
        )
    }
}

impl BoundPredicateVisitor for PageIndexEvaluator<'_> {
    type T = RowSelection;

    fn always_true(&mut self) -> Result<RowSelection> {
        self.select_all_rows()
    }

    fn always_false(&mut self) -> Result<RowSelection> {
        self.skip_all_rows()
    }

    fn and(&mut self, lhs: RowSelection, rhs: RowSelection) -> Result<RowSelection> {
        Ok(lhs.intersection(&rhs))
    }

    fn or(&mut self, lhs: RowSelection, rhs: RowSelection) -> Result<RowSelection> {
        Ok(lhs.union(&rhs))
    }

    fn not(&mut self, _: RowSelection) -> Result<RowSelection> {
        Err(Error::new(
            ErrorKind::Unexpected,
            "NOT unsupported at this point. NOT-rewrite should be performed first",
        ))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |_max, _min, null_count| Ok(!matches!(null_count, PageNullCount::NoneNull)),
            MissingColBehavior::MightMatch,
        )
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |_max, _min, null_count| Ok(!matches!(null_count, PageNullCount::AllNull)),
            MissingColBehavior::CantMatch,
        )
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // NaN counts not present in ColumnChunkMetadata Statistics.
        // Only float columns can be NaN.
        if reference.field().field_type.is_floating_type() {
            self.select_all_rows()
        } else {
            self.skip_all_rows()
        }
    }

    fn not_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // NaN counts not present in ColumnChunkMetadata Statistics
        self.select_all_rows()
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::lt, true)
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::le, true)
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::gt, false)
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        self.visit_inequality(reference, datum, PartialOrd::ge, false)
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if matches!(nulls, PageNullCount::AllNull) {
                    return Ok(false);
                }

                if let Some(min) = min
                    && min.gt(datum)
                {
                    return Ok(false);
                }

                if let Some(max) = max
                    && max.lt(datum)
                {
                    return Ok(false);
                }

                Ok(true)
            },
            MissingColBehavior::CantMatch,
        )
    }

    fn not_eq(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // Because the bounds are not necessarily a min or max value,
        // this cannot be answered using them. notEq(col, X) with (X, Y)
        // doesn't guarantee that X is a value in col.
        self.select_all_rows()
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        let PrimitiveLiteral::String(datum) = datum.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use StartsWith operator on non-string values",
            ));
        };

        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if matches!(nulls, PageNullCount::AllNull) {
                    return Ok(false);
                }

                if let Some(lower_bound) = min {
                    let PrimitiveLiteral::String(lower_bound) = lower_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use StartsWith operator on non-string lower_bound value",
                        ));
                    };

                    let prefix_length = lower_bound.chars().count().min(datum.chars().count());

                    // truncate lower bound so that its length
                    // is not greater than the length of prefix
                    let truncated_lower_bound =
                        lower_bound.chars().take(prefix_length).collect::<String>();
                    if datum < &truncated_lower_bound {
                        return Ok(false);
                    }
                }

                if let Some(upper_bound) = max {
                    let PrimitiveLiteral::String(upper_bound) = upper_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use StartsWith operator on non-string upper_bound value",
                        ));
                    };

                    let prefix_length = upper_bound.chars().count().min(datum.chars().count());

                    // truncate upper bound so that its length
                    // is not greater than the length of prefix
                    let truncated_upper_bound =
                        upper_bound.chars().take(prefix_length).collect::<String>();
                    if datum > &truncated_upper_bound {
                        return Ok(false);
                    }
                }

                Ok(true)
            },
            MissingColBehavior::CantMatch,
        )
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        // notStartsWith will match unless all values must start with the prefix.
        // This happens when the lower and upper bounds both start with the prefix.

        let PrimitiveLiteral::String(prefix) = datum.literal() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Cannot use StartsWith operator on non-string values",
            ));
        };

        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if !matches!(nulls, PageNullCount::NoneNull) {
                    return Ok(true);
                }

                let Some(lower_bound) = min else {
                    return Ok(true);
                };

                let PrimitiveLiteral::String(lower_bound_str) = lower_bound.literal() else {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        "Cannot use NotStartsWith operator on non-string lower_bound value",
                    ));
                };

                if lower_bound_str < prefix {
                    // if lower is shorter than the prefix then lower doesn't start with the prefix
                    return Ok(true);
                }

                let prefix_len = prefix.chars().count();

                if lower_bound_str.chars().take(prefix_len).collect::<String>() == *prefix {
                    // lower bound matches the prefix

                    let Some(upper_bound) = max else {
                        return Ok(true);
                    };

                    let PrimitiveLiteral::String(upper_bound) = upper_bound.literal() else {
                        return Err(Error::new(
                            ErrorKind::Unexpected,
                            "Cannot use NotStartsWith operator on non-string upper_bound value",
                        ));
                    };

                    // if upper is shorter than the prefix then upper can't start with the prefix
                    if upper_bound.chars().count() < prefix_len {
                        return Ok(true);
                    }

                    if upper_bound.chars().take(prefix_len).collect::<String>() == *prefix {
                        // both bounds match the prefix, so all rows must match the
                        // prefix and therefore do not satisfy the predicate
                        return Ok(false);
                    }
                }

                Ok(true)
            },
            MissingColBehavior::MightMatch,
        )
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        let field_id = reference.field().id;

        if literals.len() > IN_PREDICATE_LIMIT {
            // skip evaluating the predicate if the number of values is too big
            return self.select_all_rows();
        }
        self.calc_row_selection(
            field_id,
            |min, max, nulls| {
                if matches!(nulls, PageNullCount::AllNull) {
                    return Ok(false);
                }

                match (min, max) {
                    (Some(min), Some(max))
                        if literals
                            .iter()
                            .all(|datum| datum.lt(&min) || datum.gt(&max)) =>
                    {
                        // if all values are outside the bounds, no rows can match
                        return Ok(false);
                    }
                    (Some(min), _) if !literals.iter().any(|datum| datum.ge(&min)) => {
                        // if no values are within the min bound, no rows can match
                        return Ok(false);
                    }
                    (_, Some(max)) if !literals.iter().any(|datum| datum.le(&max)) => {
                        // if no values are within the max bound, no rows can match
                        return Ok(false);
                    }
                    _ => {}
                }

                Ok(true)
            },
            MissingColBehavior::CantMatch,
        )
    }

    fn not_in(
        &mut self,
        _reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<RowSelection> {
        // Because the bounds are not necessarily a min or max value,
        // this cannot be answered using them. notIn(col, {X, ...})
        // with (X, Y) doesn't guarantee that X is a value in col.
        self.select_all_rows()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BinaryArray, FixedSizeBinaryArray, Float32Array, RecordBatch, StringArray,
        TimestampMicrosecondArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{
        ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelector,
    };
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
    use parquet::file::page_index::column_index::ColumnIndexMetaData;
    use parquet::file::properties::WriterProperties;
    use rand::Rng;
    use tempfile::NamedTempFile;

    use super::PageIndexEvaluator;
    use crate::expr::{Bind, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, Type};
    use crate::{ErrorKind, Result};

    /// Helper function to create a test parquet file with page indexes
    /// and return the metadata needed for testing
    fn create_test_parquet_file() -> Result<(Arc<ParquetMetaData>, NamedTempFile)> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col_float", DataType::Float32, true),
            Field::new("col_string", DataType::Utf8, true),
        ]));

        let temp_file = NamedTempFile::new().unwrap();
        let file = temp_file.reopen().unwrap();

        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(1024)
            .set_write_batch_size(512)
            .build();

        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

        let mut batches = vec![];

        // Batch 1: 1024 rows - strings with AARDVARK, BEAR, BISON
        let float_vals: Vec<Option<f32>> = vec![None; 1024];
        let mut string_vals = vec![];
        string_vals.push(Some("AARDVARK".to_string()));
        for _ in 1..1023 {
            string_vals.push(Some("BEAR".to_string()));
        }
        string_vals.push(Some("BISON".to_string()));

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        // Batch 2: 1024 rows - all DEER
        let float_vals: Vec<Option<f32>> = vec![None; 1024];
        let string_vals = vec![Some("DEER".to_string()); 1024];

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        // Batch 3: 1024 rows - float 0-10
        let mut float_vals = vec![];
        for i in 0..1024 {
            float_vals.push(Some(i as f32 * 10.0 / 1024.0));
        }
        let mut string_vals = vec![];
        string_vals.push(Some("GIRAFFE".to_string()));
        string_vals.push(None);
        for _ in 2..1024 {
            string_vals.push(Some("HIPPO".to_string()));
        }

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        // Batch 4: 1024 rows - float 10-20
        let mut float_vals = vec![None];
        for i in 1..1024 {
            float_vals.push(Some(10.0 + i as f32 * 10.0 / 1024.0));
        }
        let string_vals = vec![Some("HIPPO".to_string()); 1024];

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        // Write rows one at a time to give the writer a chance to split into pages
        for batch in &batches {
            for i in 0..batch.num_rows() {
                writer.write(&batch.slice(i, 1)).unwrap();
            }
        }

        writer.close().unwrap();

        let file = temp_file.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options).unwrap();
        let metadata = reader.metadata().clone();

        Ok((metadata, temp_file))
    }

    /// Get the test metadata components for testing
    fn get_test_metadata(
        metadata: &ParquetMetaData,
    ) -> (
        Vec<Option<ColumnIndexMetaData>>,
        Vec<Option<parquet::file::page_index::offset_index::OffsetIndexMetaData>>,
        &parquet::file::metadata::RowGroupMetaData,
    ) {
        let row_group_metadata = metadata.row_group(0);
        let page_index = metadata.page_index_for_row_group(0);
        let column_count = row_group_metadata.columns().len();
        let column_index = (0..column_count)
            .map(|column_idx| page_index.column_index(column_idx).cloned())
            .collect();
        let offset_index = (0..column_count)
            .map(|column_idx| page_index.offset_index(column_idx).cloned())
            .collect();
        (column_index, offset_index, row_group_metadata)
    }

    #[test]
    fn eval_matches_no_rows_for_empty_row_group() -> Result<()> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col_float", DataType::Float32, true),
            Field::new("col_string", DataType::Utf8, true),
        ]));

        let empty_float: ArrayRef = Arc::new(Float32Array::from(Vec::<Option<f32>>::new()));
        let empty_string: ArrayRef = Arc::new(StringArray::from(Vec::<Option<String>>::new()));
        let empty_batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![empty_float, empty_string]).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        let file = temp_file.reopen().unwrap();

        let mut writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
        writer.write(&empty_batch).unwrap();
        writer.close().unwrap();

        let file = temp_file.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options).unwrap();
        let metadata = reader.metadata();

        if metadata.num_row_groups() == 0 || metadata.row_group(0).num_rows() == 0 {
            return Ok(());
        }

        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .greater_than(Datum::float(1.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let row_group_metadata = metadata.row_group(0);
        let page_index = metadata.page_index_for_row_group(0);
        let column_count = row_group_metadata.columns().len();
        let column_index = (0..column_count)
            .map(|column_idx| page_index.column_index(column_idx).cloned())
            .collect::<Vec<_>>();
        let offset_index = (0..column_count)
            .map(|column_idx| page_index.offset_index(column_idx).cloned())
            .collect::<Vec<_>>();

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        assert_eq!(result.len(), 0);

        Ok(())
    }

    #[test]
    fn eval_is_null_select_only_pages_with_nulls() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_null()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::select(2048),
            RowSelector::skip(1024),
            RowSelector::select(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_is_not_null_dont_select_pages_with_all_nulls() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_not_null()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::skip(2048), RowSelector::select(2048)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_is_nan_select_all() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_nan()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_nan_select_all() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_not_nan()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_inequality_nan_datum_all_rows_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .less_than(Datum::float(f32::NAN))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::skip(2048), RowSelector::select(2048)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_inequality_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .less_than(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::skip(2048),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_eq_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .equal_to(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        // Pages 0-1: all null (skip)
        // Page 2: 0-10 (select, might contain 5.0)
        // Page 3: 10-20 (skip, min > 5.0)
        let expected = vec![
            RowSelector::skip(2048),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_eq_all_rows() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .not_equal_to(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_starts_with_error_float_col() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .starts_with(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        );

        assert_eq!(result.unwrap_err().kind(), ErrorKind::Unexpected);

        Ok(())
    }

    #[test]
    fn eval_starts_with_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        // Test starts_with on string column where only some pages match
        // Our file has 4 pages: ["AARDVARK".."BISON"], ["DEER"], ["GIRAFFE".."HIPPO"], ["HIPPO"]
        // Testing starts_with("B") should select only page 0
        let filter = Reference::new("col_string")
            .starts_with(Datum::string("B"))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        // Page 0 has "BEAR" and "BISON" (starts with B), rest don't
        let expected = vec![RowSelector::select(1024), RowSelector::skip(3072)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_starts_with_pages_containing_value_except_pages_with_min_and_max_equal_to_prefix_and_all_null_pages()
    -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        // Test not_starts_with where one page has ALL values starting with prefix
        // Our file has page 1 with all "DEER" (min="DEER", max="DEER")
        // Testing not_starts_with("DE") should skip page 1 where all values start with "DE"
        let filter = Reference::new("col_string")
            .not_starts_with(Datum::string("DE"))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        // Page 0: mixed values (select)
        // Page 1: all "DEER" starting with "DE" (skip)
        // Pages 2-3: other values not all starting with "DE" (select)
        let expected = vec![
            RowSelector::select(1024),
            RowSelector::skip(1024),
            RowSelector::select(2048),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_in_length_of_set_above_limit_all_rows() -> Result<()> {
        let mut rng = rand::rng();
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_in(std::iter::repeat_with(|| Datum::float(rng.random_range(0.0..10.0))).take(1000))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_in_valid_set_size_some_rows() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        // Test is_in with multiple values using min/max bounds
        // Our file has 4 pages: ["AARDVARK".."BISON"], ["DEER"], ["GIRAFFE".."HIPPO"], ["HIPPO"]
        // Testing is_in(["AARDVARK", "GIRAFFE"]) - both are in different pages
        let filter = Reference::new("col_string")
            .is_in([Datum::string("AARDVARK"), Datum::string("GIRAFFE")])
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        // Page 0 contains "AARDVARK", page 1 doesn't contain either, page 2 contains "GIRAFFE", page 3 doesn't
        let expected = vec![
            RowSelector::select(1024),
            RowSelector::skip(1024),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    /// Rows written per data page by [`create_binary_bounds_parquet_file`].
    const BINARY_PAGE_ROWS: usize = 1024;

    /// Number of data pages written by [`create_binary_bounds_parquet_file`].
    const BINARY_PAGES: usize = 4;

    /// Returns the 16-byte identifier written at `row` of `page`.
    ///
    /// The first byte is the page number, so identifiers sort by page and
    /// every page's `FIXED_LEN_BYTE_ARRAY` bounds are disjoint.
    fn binary_bounds_id(page: usize, row: usize) -> Vec<u8> {
        let mut id = vec![0u8; 16];
        id[0] = page as u8;
        id[14..].copy_from_slice(&(row as u16).to_be_bytes());
        id
    }

    /// Returns the non-UTF-8 binary payload written on every row of `page`.
    fn binary_bounds_payload(page: usize) -> Vec<u8> {
        vec![0xFF, 0xFE, page as u8]
    }

    /// Writes one row group of [`BINARY_PAGES`] data pages holding a 16-byte
    /// fixed-size id, a non-UTF-8 binary payload, and a microsecond
    /// timestamp equal to `page * 1_000_000 + row`, and returns its metadata
    /// with the page index loaded.
    fn create_binary_bounds_parquet_file() -> Result<(Arc<ParquetMetaData>, NamedTempFile)> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::FixedSizeBinary(16), false),
            Field::new("payload", DataType::Binary, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));

        let temp_file = NamedTempFile::new().unwrap();
        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(BINARY_PAGE_ROWS)
            .set_write_batch_size(512)
            .build();
        let mut writer = ArrowWriter::try_new(
            temp_file.reopen().unwrap(),
            arrow_schema.clone(),
            Some(props),
        )
        .unwrap();

        for page in 0..BINARY_PAGES {
            let ids: Vec<Vec<u8>> = (0..BINARY_PAGE_ROWS)
                .map(|row| binary_bounds_id(page, row))
                .collect();
            let payloads = vec![binary_bounds_payload(page); BINARY_PAGE_ROWS];
            let timestamps: Vec<i64> = (0..BINARY_PAGE_ROWS)
                .map(|row| (page * 1_000_000 + row) as i64)
                .collect();
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(FixedSizeBinaryArray::try_from_iter(ids.into_iter()).unwrap()),
                Arc::new(BinaryArray::from_iter_values(payloads)),
                Arc::new(TimestampMicrosecondArray::from(timestamps)),
            ])
            .unwrap();
            // Write rows one at a time so the writer honours the page row limit.
            for row in 0..batch.num_rows() {
                writer.write(&batch.slice(row, 1)).unwrap();
            }
        }
        writer.close().unwrap();

        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(
            temp_file.reopen().unwrap(),
            options,
        )
        .unwrap();
        Ok((reader.metadata().clone(), temp_file))
    }

    /// Proves that binary page bounds keep real page selection: 16-byte
    /// fixed-size ids select only the page holding the value, non-UTF-8
    /// binary bounds are compared as bytes, a binary `IN` intersected with a
    /// selective timestamp bound narrows to the single matching page, and an
    /// Iceberg type with no defined reading of the physical bound keeps every
    /// page instead of failing.
    #[test]
    fn binary_bounds_preserve_mixed_pruning() -> Result<()> {
        let (metadata, _temp_file) = create_binary_bounds_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        assert!(
            matches!(
                column_index[0],
                Some(ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(_))
            ),
            "id must carry a FIXED_LEN_BYTE_ARRAY column index"
        );
        assert_eq!(
            column_index[0].as_ref().map(ColumnIndexMetaData::num_pages),
            Some(BINARY_PAGES as u64)
        );

        let schema = Arc::new(
            Schema::builder()
                .with_fields([
                    Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Fixed(16)),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "payload",
                        Type::Primitive(PrimitiveType::Binary),
                    )),
                    Arc::new(NestedField::required(
                        3,
                        "ts",
                        Type::Primitive(PrimitiveType::Timestamp),
                    )),
                    Arc::new(NestedField::required(
                        4,
                        "id_as_int",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                ])
                .build()?,
        );
        let field_id_map = HashMap::from_iter([(1, 0), (2, 1), (3, 2), (4, 0)]);
        let page = |selectors: &[(bool, usize)]| -> Vec<RowSelector> {
            selectors
                .iter()
                .map(|&(select, pages)| {
                    let rows = pages * BINARY_PAGE_ROWS;
                    if select {
                        RowSelector::select(rows)
                    } else {
                        RowSelector::skip(rows)
                    }
                })
                .collect()
        };
        let eval = |filter: crate::expr::Predicate| -> Result<Vec<RowSelector>> {
            let bound = filter.bind(schema.clone(), false)?;
            PageIndexEvaluator::eval(
                &bound,
                &column_index,
                &offset_index,
                row_group_metadata,
                &field_id_map,
                schema.as_ref(),
            )
        };

        let fixed_eq = eval(Reference::new("id").equal_to(Datum::fixed(binary_bounds_id(2, 5))))?;
        assert_eq!(fixed_eq, page(&[(false, 2), (true, 1), (false, 1)]));

        let fixed_absent =
            eval(Reference::new("id").equal_to(Datum::fixed(binary_bounds_id(9, 0))))?;
        assert_eq!(fixed_absent, page(&[(false, 4)]));

        let binary_eq =
            eval(Reference::new("payload").equal_to(Datum::binary(binary_bounds_payload(1))))?;
        assert_eq!(binary_eq, page(&[(false, 1), (true, 1), (false, 2)]));

        let mixed = eval(
            Reference::new("id")
                .is_in([
                    Datum::fixed(binary_bounds_id(0, 7)),
                    Datum::fixed(binary_bounds_id(2, 7)),
                ])
                .and(
                    Reference::new("ts")
                        .greater_than_or_equal_to(Datum::timestamp_micros(2_000_000)),
                ),
        )?;
        assert_eq!(mixed, page(&[(false, 2), (true, 1), (false, 1)]));

        let unsupported = eval(Reference::new("id_as_int").equal_to(Datum::int(5)))?;
        assert_eq!(unsupported, page(&[(true, 4)]));

        Ok(())
    }

    fn build_iceberg_schema_and_field_map() -> Result<(Arc<Schema>, HashMap<i32, usize>)> {
        let iceberg_schema = Schema::builder()
            .with_fields([
                Arc::new(NestedField::new(
                    1,
                    "col_float",
                    Type::Primitive(PrimitiveType::Float),
                    false,
                )),
                Arc::new(NestedField::new(
                    2,
                    "col_string",
                    Type::Primitive(PrimitiveType::String),
                    false,
                )),
            ])
            .build()?;
        let iceberg_schema_ref = Arc::new(iceberg_schema);

        let field_id_map = HashMap::from_iter([(1, 0), (2, 1)]);

        Ok((iceberg_schema_ref, field_id_map))
    }
}
