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

//! Evaluates equality and `IN` predicates against the Parquet split-block
//! Bloom filters of one row group.

use std::collections::{BTreeSet, HashMap};

use fnv::FnvHashSet;
use parquet::basic::{LogicalType, TimeUnit, Type as PhysicalType};
use parquet::bloom_filter::Sbbf;
use parquet::file::metadata::RowGroupMetaData;
use parquet::schema::types::ColumnDescriptor;

use crate::Result;
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType};

/// Decides whether a row group might contain rows matching a predicate,
/// using only the Bloom filters of the columns its equality and `IN` terms
/// reference.
///
/// The evaluator answers "might match": `false` means the loaded Bloom
/// filters prove that no row of the row group can satisfy the predicate.
/// Every term it cannot decide — ranges, negations, nulls, missing columns,
/// columns without a Bloom filter, unsupported types — answers `true`, so a
/// matching row is never excluded. The predicate must already have `NOT`
/// rewritten away; any remaining `NOT` is treated as "might match".
pub(crate) struct BloomFilterEvaluator<'a> {
    /// Metadata of the row group being evaluated.
    row_group_metadata: &'a RowGroupMetaData,
    /// Iceberg field id to Parquet leaf column index.
    field_id_map: &'a HashMap<i32, usize>,
    /// Bloom filters loaded for this row group, keyed by leaf column index.
    bloom_filters: &'a HashMap<usize, Sbbf>,
    /// Leaf columns whose Bloom filter an equality or `IN` term consulted.
    probed_columns: BTreeSet<usize>,
}

impl<'a> BloomFilterEvaluator<'a> {
    /// Returns the leaf columns of `row_group_metadata` whose Bloom filters
    /// could decide `filter`: columns referenced by an equality or `IN`
    /// term that declare a Bloom filter in the column chunk metadata.
    ///
    /// The caller loads exactly these filters before calling [`Self::eval`],
    /// so columns that appear only in range or null terms cost no IO.
    ///
    /// # Errors
    ///
    /// Returns an error only if visiting the bound predicate fails.
    pub(crate) fn probe_columns(
        filter: &BoundPredicate,
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
    ) -> Result<Vec<usize>> {
        let no_filters = HashMap::new();
        let mut evaluator = BloomFilterEvaluator {
            row_group_metadata,
            field_id_map,
            bloom_filters: &no_filters,
            probed_columns: BTreeSet::new(),
        };
        visit(&mut evaluator, filter)?;
        Ok(evaluator.probed_columns.into_iter().collect())
    }

    /// Returns `false` when `bloom_filters` prove that no row of the row
    /// group can satisfy `filter`, and `true` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error only if visiting the bound predicate fails.
    pub(crate) fn eval(
        filter: &BoundPredicate,
        row_group_metadata: &'a RowGroupMetaData,
        field_id_map: &'a HashMap<i32, usize>,
        bloom_filters: &'a HashMap<usize, Sbbf>,
    ) -> Result<bool> {
        let mut evaluator = BloomFilterEvaluator {
            row_group_metadata,
            field_id_map,
            bloom_filters,
            probed_columns: BTreeSet::new(),
        };
        visit(&mut evaluator, filter)
    }

    /// Returns `false` only when the referenced column's Bloom filter proves
    /// every one of `datums` absent; records the column as probed whenever
    /// its chunk declares a Bloom filter.
    fn might_contain_any<'d>(
        &mut self,
        reference: &BoundReference,
        datums: impl IntoIterator<Item = &'d Datum>,
    ) -> bool {
        let Some(&column) = self.field_id_map.get(&reference.field().id) else {
            return true;
        };
        let Some(chunk) = self.row_group_metadata.columns().get(column) else {
            return true;
        };
        if chunk.bloom_filter_offset().is_none() {
            return true;
        }
        self.probed_columns.insert(column);
        let Some(bloom_filter) = self.bloom_filters.get(&column) else {
            return true;
        };
        let descriptor = chunk.column_descr();
        datums
            .into_iter()
            .any(|datum| might_contain(bloom_filter, descriptor, datum))
    }
}

/// Returns `false` only when `bloom_filter` proves `datum` absent from the
/// column described by `column`.
///
/// The probe hashes the same bytes the Parquet writer hashed for the
/// physical value: native `i32`/`i64` bytes for `INT32`/`INT64`, UTF-8
/// bytes for strings, raw bytes for binary and fixed, and the 16 big-endian
/// bytes for a uuid. `INT32` columns read as Iceberg `long` (type
/// promotion) are probed with the narrowed `i32`. Time and timestamp probes
/// require the column's logical unit to match the Iceberg type, because a
/// unit mismatch would hash a different value. Every other pairing —
/// floating point, decimal, boolean, `INT96`, or a length mismatch — answers
/// `true`.
fn might_contain(bloom_filter: &Sbbf, column: &ColumnDescriptor, datum: &Datum) -> bool {
    let unit = || match column.logical_type_ref() {
        Some(LogicalType::Timestamp(timestamp)) => Some(&timestamp.unit),
        Some(LogicalType::Time(time)) => Some(&time.unit),
        _ => None,
    };
    let is_micros = || matches!(unit(), Some(TimeUnit::MICROS));
    let is_nanos = || matches!(unit(), Some(TimeUnit::NANOS));

    match (column.physical_type(), datum.data_type(), datum.literal()) {
        (
            PhysicalType::INT32,
            PrimitiveType::Int | PrimitiveType::Date,
            PrimitiveLiteral::Int(v),
        ) => bloom_filter.check(v),
        (PhysicalType::INT32, PrimitiveType::Long, PrimitiveLiteral::Long(v)) => {
            i32::try_from(*v).map_or(true, |v| bloom_filter.check(&v))
        }
        (PhysicalType::INT64, PrimitiveType::Long, PrimitiveLiteral::Long(v)) => {
            bloom_filter.check(v)
        }
        (
            PhysicalType::INT64,
            PrimitiveType::Time | PrimitiveType::Timestamp | PrimitiveType::Timestamptz,
            PrimitiveLiteral::Long(v),
        ) if is_micros() => bloom_filter.check(v),
        (
            PhysicalType::INT64,
            PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs,
            PrimitiveLiteral::Long(v),
        ) if is_nanos() => bloom_filter.check(v),
        (PhysicalType::BYTE_ARRAY, PrimitiveType::String, PrimitiveLiteral::String(v)) => {
            bloom_filter.check(v.as_str())
        }
        (PhysicalType::BYTE_ARRAY, PrimitiveType::Binary, PrimitiveLiteral::Binary(v)) => {
            bloom_filter.check(v.as_slice())
        }
        (
            PhysicalType::FIXED_LEN_BYTE_ARRAY,
            PrimitiveType::Fixed(_),
            PrimitiveLiteral::Binary(v),
        ) if usize::try_from(column.type_length()).ok() == Some(v.len()) => {
            bloom_filter.check(v.as_slice())
        }
        (PhysicalType::FIXED_LEN_BYTE_ARRAY, PrimitiveType::Uuid, PrimitiveLiteral::UInt128(v))
            if column.type_length() == 16 =>
        {
            bloom_filter.check(v.to_be_bytes().as_slice())
        }
        _ => true,
    }
}

impl BoundPredicateVisitor for BloomFilterEvaluator<'_> {
    type T = bool;

    fn always_true(&mut self) -> Result<bool> {
        Ok(true)
    }

    fn always_false(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn and(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs && rhs)
    }

    fn or(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs || rhs)
    }

    fn not(&mut self, _inner: bool) -> Result<bool> {
        // A Bloom filter cannot prove that every row matches the inner term.
        Ok(true)
    }

    fn is_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn not_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn is_nan(&mut self, _reference: &BoundReference, _predicate: &BoundPredicate) -> Result<bool> {
        Ok(true)
    }

    fn not_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn less_than(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn less_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn greater_than(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn greater_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(self.might_contain_any(reference, [literal]))
    }

    fn not_eq(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn starts_with(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn not_starts_with(
        &mut self,
        _reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(self.might_contain_any(reference, literals))
    }

    fn not_in(
        &mut self,
        _reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, Date32Array, FixedSizeBinaryArray, Int32Array, Int64Array, RecordBatch,
        TimestampMicrosecondArray, TimestampMillisecondArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use parquet::bloom_filter::Sbbf;
    use parquet::file::properties::{ReaderProperties, WriterProperties};
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::BloomFilterEvaluator;
    use crate::expr::{Bind, Predicate, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, Type};

    /// Microsecond timestamp base written to the timestamp columns.
    const TS_BASE: i64 = 1_700_000_000_000_000;

    /// Returns the uuid written on row `row`.
    fn row_uuid(row: u128) -> Uuid {
        Uuid::from_u128((row + 1).wrapping_mul(0x0123_4567_89ab_cdef_0011_2233_4455_6677))
    }

    /// Proves the Bloom probe hashes each Iceberg value exactly as the
    /// Parquet writer hashed the physical value: present `int`, `long`,
    /// `date`, microsecond `timestamp`, `uuid`, and promoted `int`→`long`
    /// values never prune, Bloom-negative values do, and a millisecond
    /// column read as a microsecond Iceberg timestamp always keeps the row
    /// group because its hashed values are in a different unit.
    #[test]
    fn probes_hash_physical_values_like_the_writer() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("int", DataType::Int32, false),
            Field::new("long", DataType::Int64, false),
            Field::new("date", DataType::Date32, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "ts_ms",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
            Field::new("uuid", DataType::FixedSizeBinary(16), false),
        ]));
        let rows = 0..10i64;
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from_iter_values(
                rows.clone().map(|i| i as i32 * 10),
            )) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(
                rows.clone().map(|i| i * 1_000_000_007),
            )),
            Arc::new(Date32Array::from_iter_values(
                rows.clone().map(|i| 19_000 + i as i32),
            )),
            Arc::new(TimestampMicrosecondArray::from_iter_values(
                rows.clone().map(|i| TS_BASE + i),
            )),
            Arc::new(TimestampMillisecondArray::from_iter_values(
                rows.clone().map(|i| TS_BASE + i),
            )),
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    rows.clone().map(|i| row_uuid(i as u128).into_bytes()),
                )
                .unwrap(),
            ),
        ])
        .unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bloom.parquet");
        let props = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), arrow_schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let options = ReadOptionsBuilder::new()
            .with_reader_properties(
                ReaderProperties::builder()
                    .set_read_bloom_filter(true)
                    .build(),
            )
            .build();
        let reader =
            SerializedFileReader::new_with_options(File::open(&path).unwrap(), options).unwrap();
        let row_group = reader.get_row_group(0).unwrap();
        let bloom_filters: HashMap<usize, Sbbf> = (0..6)
            .map(|column| {
                let filter = row_group
                    .get_column_bloom_filter(column)
                    .expect("Bloom filter written for every column")
                    .clone();
                (column, filter)
            })
            .collect();
        let row_group_metadata = reader.metadata().row_group(0);

        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "int", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "long", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(3, "date", Type::Primitive(PrimitiveType::Date)).into(),
                    NestedField::required(4, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(5, "ts_ms", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(6, "uuid", Type::Primitive(PrimitiveType::Uuid)).into(),
                    NestedField::required(7, "promoted", Type::Primitive(PrimitiveType::Long))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let field_id_map = HashMap::from([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 0)]);
        let might_match = |predicate: Predicate| {
            let bound = predicate.bind(schema.clone(), true).unwrap();
            BloomFilterEvaluator::eval(&bound, row_group_metadata, &field_id_map, &bloom_filters)
                .unwrap()
        };
        let eq = |column: &str, datum: Datum| might_match(Reference::new(column).equal_to(datum));

        // Present values must never be pruned.
        assert!(eq("int", Datum::int(30)));
        assert!(eq("long", Datum::long(3 * 1_000_000_007i64)));
        assert!(eq("date", Datum::date(19_003)));
        assert!(eq("ts", Datum::timestamp_micros(TS_BASE + 3)));
        assert!(eq("uuid", Datum::uuid(row_uuid(3))));
        assert!(eq("promoted", Datum::long(30)));

        // Bloom-negative values are pruned once the writer-side check agrees.
        assert!(!bloom_filters[&0].check(&35i32));
        assert!(!eq("int", Datum::int(35)));
        assert!(!eq("promoted", Datum::long(35)));
        assert!(!bloom_filters[&1].check(&1i64));
        assert!(!eq("long", Datum::long(1)));
        assert!(!bloom_filters[&2].check(&18_000i32));
        assert!(!eq("date", Datum::date(18_000)));
        assert!(!bloom_filters[&3].check(&(TS_BASE + 100)));
        assert!(!eq("ts", Datum::timestamp_micros(TS_BASE + 100)));
        let absent_uuid = Uuid::from_u128(42);
        assert!(!bloom_filters[&5].check(absent_uuid.as_bytes().as_slice()));
        assert!(!eq("uuid", Datum::uuid(absent_uuid)));
        assert!(!might_match(
            Reference::new("int").is_in([Datum::int(35), Datum::int(36)])
        ));
        assert!(might_match(
            Reference::new("int").is_in([Datum::int(35), Datum::int(40)])
        ));

        // A long that cannot be an INT32 value keeps the row group rather than guessing.
        assert!(eq("promoted", Datum::long(i64::from(i32::MAX) + 1)));
        // A unit mismatch keeps the row group even for a Bloom-negative probe.
        assert!(!bloom_filters[&4].check(&(TS_BASE + 100)));
        assert!(eq("ts_ms", Datum::timestamp_micros(TS_BASE + 100)));
        // Negations and non-equality terms are never decided by a Bloom filter.
        assert!(might_match(
            Reference::new("int").not_equal_to(Datum::int(35))
        ));
        assert!(might_match(
            Reference::new("int")
                .equal_to(Datum::int(35))
                .or(Reference::new("long").greater_than(Datum::long(0)))
        ));
    }
}
